#!/usr/bin/env python3
"""
ScienceWorld LLM Evaluation (Pure ICL Baseline - Deterministic)
No action prioritization, no truncation, deterministic decoding
"""

import json
import random
import numpy as np
import argparse
import hashlib
import os
import re
import requests
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime
from scienceworld import ScienceWorldEnv
from collections import defaultdict, Counter

# ===== EXPERIMENT SWITCH (EDIT HERE ONLY) =====
# Manifest and examples
_DEFAULT_MANIFEST = r"C:\Users\ylili\Desktop\ASL\abschlussarbeit\sciworld\agenttraj-l-train-0.sciworld_only.json"
_DEFAULT_EXAMPLES = r"C:\Users\ylili\Desktop\ASL\abschlussarbeit\sciworld\eval_results\selector_corpus@20251103_073850.full.json"
_RUN_TRIALS_LIMIT = 2120  # 全量任务数

# Shots to run in one go (no need to change via terminal)
# Change this to run different K values: [2], [4], [8], [2,4], [2,4,8], etc.
_SHOT_LIST = [0, 2, 4, 8]  # Run 0-shot, 2-shot, 4-shot, 8-shot

# Baseline mode settings (deterministic by default)
_DETERMINISTIC = True  # True = temp=0, top_p=1, top_k=1 (fair baseline)


# ===== END EXPERIMENT SWITCH =====


class ICLExampleManager:
    """Manage and select ICL examples"""

    def __init__(self, examples_file: str = None, max_examples: int = 3):
        self.max_examples = max_examples
        self.examples = defaultdict(list)
        self.loaded = False

        if examples_file and os.path.exists(examples_file):
            self.load_examples(examples_file)

    def load_examples(self, filepath: str):
        """Load successful trajectories as examples"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

                # Handle different formats
                if isinstance(data, dict):
                    if 'trials' in data:
                        for trial in data['trials']:
                            if trial.get('success', False) or trial.get('cumulative_reward', 0) > 0:
                                self.add_example_from_trial(trial)
                    else:
                        for task_name, examples in data.items():
                            if isinstance(examples, list):
                                for ex in examples:
                                    self.add_formatted_example(task_name, ex)
                elif isinstance(data, list):
                    for example in data:
                        if example.get('success', False) or example.get('cumulative_reward', 0) > 0:
                            self.add_example_from_trial(example)

            self.loaded = True
            print(f"Loaded ICL examples for {len(self.examples)} tasks")
        except Exception as e:
            print(f"Warning: Could not load examples from {filepath}: {e}")

    def add_example_from_trial(self, trial: Dict):
        """Add an example from a trial result"""
        task_name = trial.get('task_name', '') or trial.get('task', '') or trial.get('env_name', '')
        if not task_name:
            return

        goal = trial.get('goal') or trial.get('task_description') or f"Complete {task_name.replace('-', ' ')}"

        raw_steps = trial.get('action_history') or trial.get('trajectory', [])
        steps = []
        if isinstance(raw_steps, list):
            for item in raw_steps[:5]:
                if isinstance(item, dict):
                    act = item.get('action', '')
                    obs = item.get('observation', '') or item.get('llm_output', '')
                else:
                    act = str(item)
                    obs = ''
                steps.append({'action': act, 'observation': obs[:100]})

        if steps:
            self.examples[task_name].append({
                'goal': goal,
                'steps': steps,
                'reward': trial.get('cumulative_reward', 1.0)
            })

    def add_formatted_example(self, task_name: str, example: Dict):
        """Add a pre-formatted example"""
        self.examples[task_name].append(example)

    def get_examples(self, task_name: str, n: int = None) -> List[Dict]:
        """Get ICL examples for a task"""
        n = n or self.max_examples

        if task_name in self.examples:
            examples = self.examples[task_name][:n]
            if examples:
                return examples

        # Try similar tasks
        similar_examples = []
        task_prefix = task_name.split('-')[0]
        for other_task, exs in self.examples.items():
            if task_prefix in other_task:
                similar_examples.extend(exs)

        return similar_examples[:n]


class LLMICLEvaluator:
    """ICL-based LLM evaluation for ScienceWorld with RA-ICL"""

    def __init__(self,
                 jar_path: str = None,
                 seed: int = 42,
                 step_limit: int = 30,
                 output_dir: str = "eval_results",
                 model_tag: str = "llama3.2:3b-instruct-q4_K_M",
                 ollama_host: str = "http://localhost:11434",
                 max_vars_per_task: int = None,
                 num_predict: int = 50,
                 request_timeout: int = 120,
                 num_ctx: int = 4096,

                 # ICL specific
                 examples_file: str = None,
                 num_examples: int = 2,
                 dynamic_examples: bool = True,
                 use_simple_prompt: bool = False,

                 # RA-ICL specific
                 consistency_n: int = 3,
                 icl_retrieval: bool = False,
                 icl_mmr_lambda: float = 0.3,

                 # Speed knobs
                 num_shards: int = 1,
                 shard_index: int = 0,
                 quiet: bool = False,
                 ollama_num_thread: int = None,
                 ollama_num_batch: int = None,
                 ollama_use_mmap: bool = None,
                 ollama_f16_kv: bool = None,

                 # Manifest config
                 manifest_json: str = None,
                 limit_total: int = None,

                 # Pure ICL baseline
                 deterministic: bool = False,
                 ):

        self.jar_path = jar_path
        self.seed = seed
        self.step_limit = step_limit
        self.output_dir = output_dir
        self.model_tag = model_tag
        self.ollama_host = ollama_host
        self.max_vars_per_task = max_vars_per_task
        self.num_predict = max(1, int(num_predict))
        self.request_timeout = max(30, int(request_timeout))
        self.num_ctx = int(num_ctx)

        # ICL config
        self.num_examples = num_examples
        self.dynamic_examples = dynamic_examples
        self.use_simple_prompt = use_simple_prompt
        self.example_manager = ICLExampleManager(examples_file, num_examples)
        self.successful_trajectories = []

        # RA-ICL config
        self.consistency_n = consistency_n
        self.icl_retrieval = icl_retrieval
        self.icl_mmr_lambda = icl_mmr_lambda
        self.temperature = 0.4

        # Speed config
        self.num_shards = max(1, int(num_shards))
        self.shard_index = max(0, int(shard_index))
        self.quiet = bool(quiet)
        self.ollama_num_thread = ollama_num_thread
        self.ollama_num_batch = ollama_num_batch
        self.ollama_use_mmap = ollama_use_mmap
        self.ollama_f16_kv = ollama_f16_kv

        # Manifest config
        self.manifest_json = manifest_json
        self.limit_total = limit_total

        # Pure ICL baseline config
        self.deterministic = bool(deterministic)

        # Session
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json", "Connection": "keep-alive"})

        # Seeds
        random.seed(seed)
        np.random.seed(seed)

        # Env
        try:
            self.env = ScienceWorldEnv("", jar_path, envStepLimit=step_limit)
        except TypeError:
            self.env = ScienceWorldEnv("", envStepLimit=step_limit)
        if hasattr(self.env, "seed"):
            self.env.seed(seed)

        # Ollama
        self.test_ollama()

        # Out dir
        os.makedirs(output_dir, exist_ok=True)

        # ===== BEGIN PATCH: 记住代码内开关的实际值（默认读 selector corpus） =====
        # manifest 优先用 examples_file（通常就是 selector_corpus），否则退回默认
        self._manifest_json = manifest_json if (manifest_json and os.path.exists(manifest_json)) else _DEFAULT_MANIFEST
        self._limit_total = int(limit_total) if limit_total else int(_RUN_TRIALS_LIMIT)

        # Override deterministic setting from code config
        try:
            self.deterministic = bool(_DETERMINISTIC)
        except NameError:
            pass  # Use parameter value if code config not available

        # 不再用常量覆盖 K；尊重命令行/外层循环设置的 self.num_examples

        # 若没成功加载 examples（极端容错），再尝试默认 examples
        if not self.example_manager.loaded and os.path.exists(_DEFAULT_EXAMPLES):
            try:
                self.example_manager.load_examples(_DEFAULT_EXAMPLES)
            except Exception:
                pass
        # ===== END PATCH =====

    def test_ollama(self):
        """Test Ollama connection"""
        try:
            r = self.session.get(f"{self.ollama_host}/api/tags", timeout=10)
            r.raise_for_status()
            names = [m.get("name") for m in r.json().get("models", []) if isinstance(m, dict)]

            if self.model_tag not in names:
                print(f"Model {self.model_tag} not found. Pulling...")
                with self.session.post(f"{self.ollama_host}/api/pull",
                                       json={"name": self.model_tag},
                                       stream=True, timeout=None) as pr:
                    pr.raise_for_status()
                    for line in pr.iter_lines(decode_unicode=True):
                        if not line:
                            continue
                        try:
                            evt = json.loads(line)
                        except:
                            continue
                        status = str(evt.get("status", "")).lower()
                        if status:
                            print(f"  {status}")
                        if evt.get("done") is True:
                            break

            print(f"✓ Ollama ready: {self.model_tag}")
            self.llm_available = True
        except Exception as e:
            print(f"WARNING: Ollama not ready ({e})")
            self.llm_available = False

    # ===== RA-ICL methods =====
    def _cos_sim(self, a: str, b: str) -> float:
        """Lightweight cosine similarity"""
        from collections import Counter
        ta, tb = [w for w in a.lower().split() if w], [w for w in b.lower().split() if w]
        ca, cb = Counter(ta), Counter(tb)
        keys = set(ca) | set(cb)
        na = sum(ca[k] * ca[k] for k in keys) ** 0.5 or 1.0
        nb = sum(cb[k] * cb[k] for k in keys) ** 0.5 or 1.0
        dot = sum(ca[k] * cb[k] for k in keys)
        return dot / (na * nb)

    def _mmr(self, query: str, cands: list, get_text, k: int, lam: float = 0.3):
        """Maximal Marginal Relevance"""
        selected, rest = [], list(cands)
        sims_q = [self._cos_sim(query, get_text(c)) for c in rest]
        for _ in range(min(k, len(rest))):
            if not rest:
                break
            if not selected:
                j = max(range(len(rest)), key=lambda i: sims_q[i])
            else:
                def score(i):
                    sim_q = sims_q[i]
                    sim_d = max(self._cos_sim(get_text(rest[i]), get_text(s)) for s in selected)
                    return lam * sim_q - (1 - lam) * sim_d

                j = max(range(len(rest)), key=score)
            selected.append(rest.pop(j))
            sims_q.pop(j)
        return selected

    def _map_action_text_to_current(self, action_text: str, acts_now: list) -> Optional[int]:
        """Map example action text to current action set (returns 1-based id)"""
        t = (action_text or "").strip().lower()
        if not t or not acts_now:
            return None

        texts = [a.get("action", "").strip().lower() for a in acts_now]

        # Exact match
        for j, s in enumerate(texts, 1):
            if t == s:
                return j

        # Prefix/substring match
        for j, s in enumerate(texts, 1):
            if t.startswith(s) or s.startswith(t) or t in s or s in t:
                return j

        # Verb+noun rough match
        import re
        head = re.split(r"[^\w]+", t)[:2]
        for j, s in enumerate(texts, 1):
            if all(w and w in s for w in head):
                return j

        return None

    def retrieve_icl_examples(self, goal: str, obs: str, acts_now: list, k: int) -> list:
        """Retrieve most similar steps from example pool using MMR"""
        if not self.example_manager.examples:
            return []

        query = (goal or "") + "\n" + (obs or "")
        pool = []

        # Expand to step-level candidates
        for tn, exs in self.example_manager.examples.items():
            for ex in exs:
                g = ex.get("goal", "")
                for st in ex.get("steps", []):
                    pool.append({
                        "task_name": tn,
                        "goal": g,
                        "observation": st.get("observation", ""),
                        "action_text": st.get("action", "")
                    })

        if not pool:
            return []

        # MMR selection
        picked = self._mmr(query, pool,
                           get_text=lambda r: r["goal"] + "\n" + r["observation"],
                           k=k, lam=self.icl_mmr_lambda)

        out = []
        for p in picked:
            cid = self._map_action_text_to_current(p["action_text"], acts_now)
            if cid is None:  # Skip if mapping fails
                continue
            out.append({
                "task_name": p["task_name"],
                "goal": p["goal"],
                "observation": p["observation"],
                "valid_actions": [a["action"] for a in acts_now],
                "chosen_id": cid
            })

        return out

    # ===== BEGIN PATCH: 外部 2120 清单加载（与 zero-shot 同步） =====
    def _load_manifest_from_json(self, path: str) -> List[Dict]:
        """Load external manifest .json/.jsonl -> [{'task_name':..., 'variation_id':...}, ...]"""

        def _norm_one(rec):
            task_name = rec.get('task_name') or rec.get('task') or rec.get('env_name')
            var = (rec.get('variation_id') or rec.get('variation') or
                   rec.get('var_id') or rec.get('seed') or 0)
            try:
                var = int(var)
            except Exception:
                var = 0
            if not task_name:
                return None
            return {"task_name": str(task_name), "variation_id": var}

        manifest = []
        if not os.path.exists(path):
            print(f"[warn] manifest-json not found: {path}")
            return manifest

        if path.endswith(".jsonl"):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        item = _norm_one(rec)
                        if item:
                            manifest.append(item)
                    except Exception:
                        continue
        else:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "trials" in data:
                data = data["trials"]
            if isinstance(data, list):
                for rec in data:
                    item = _norm_one(rec)
                    if item:
                        manifest.append(item)

        # 过滤非法任务名 + 去重
        try:
            all_task_names = set(self.env.get_task_names())
            manifest = [m for m in manifest if m["task_name"] in all_task_names]
        except Exception:
            pass

        seen, out = set(), []
        for m in manifest:
            key = (m["task_name"], m["variation_id"])
            if key in seen:
                continue
            seen.add(key)
            out.append(m)
        return out

    # ===== END PATCH =====

    def generate_manifest(self) -> List[Dict]:
        """Generate task manifest"""
        SELECTED_TASKS = [
            "boil", "change-the-state-of-matter-of", "chemistry-mix",
            "chemistry-mix-paint-secondary-color", "chemistry-mix-paint-tertiary-color",
            "find-animal", "find-living-thing", "find-non-living-thing", "find-plant",
            "freeze", "grow-fruit", "grow-plant", "identify-life-stages-1",
            "identify-life-stages-2", "inclined-plane-determine-angle",
            "inclined-plane-friction-named-surfaces", "inclined-plane-friction-unnamed-surfaces",
            "lifespan-longest-lived", "lifespan-longest-lived-then-shortest-lived",
            "lifespan-shortest-lived", "measure-melting-point-known-substance",
            "measure-melting-point-unknown-substance", "melt",
            "mendelian-genetics-known-plant", "mendelian-genetics-unknown-plant",
            "power-component", "power-component-renewable-vs-nonrenewable-energy",
            "test-conductivity", "test-conductivity-of-unknown-substances", "use-thermometer",
        ]
        manifest = []
        all_task_names = self.env.get_task_names()
        selected_task_names = [t for t in all_task_names if t in SELECTED_TASKS]
        selected_task_names.sort(key=lambda x: all_task_names.index(x))

        for task_name in selected_task_names:
            self.env.load(task_name, 0, "")
            max_vars = self.env.get_max_variations(task_name)
            num_vars = min(max_vars, self.max_vars_per_task) if self.max_vars_per_task else max_vars
            for var_id in range(num_vars):
                manifest.append({"task_name": task_name, "variation_id": var_id})
        return manifest

    def canonicalize_actions(self, actions: List[Dict]) -> List[Dict]:
        return sorted(actions, key=lambda x: (x.get("action", ""), str(x)))

    def build_icl_prompt(self, obs: str, goal: str, acts: list, task_name: str, step_id: int) -> str:
        """用传入的 acts 构建 ICL 提示；不在这里再取环境动作。"""
        primer = [
            # "FORMAT DEMO",
            # "Valid actions:",
            # "[1] look around",
            # "[2] examine beaker",
            # ""  # 不给固定的 Chosen 示例，避免模型被锚到 1
        ]

        # —— 取示例：优先检索，回退例库 ——
        k = self.num_examples
        exs, source = [], "none"
        if self.icl_retrieval:
            exs = self.retrieve_icl_examples(goal, obs, acts, k=k)
            if exs: source = "retrieval"
        if not exs and self.example_manager:
            fb = self.example_manager.get_examples(task_name, k) or []
            for ex in fb[:k]:
                # 回退时无法精准映射，保底成 1（通常是 look around）
                vas = [a["action"] for a in acts]
                try:
                    look_idx = next(i for i, a in enumerate(vas, 1) if a.lower().startswith("look"))
                except StopIteration:
                    look_idx = 1

                exs.append({
                    "task_name": task_name,
                    "goal": ex.get("goal", ""),
                    "observation": (ex.get("observation") or (ex.get("steps", [{}])[0].get("observation", ""))),
                    "valid_actions": vas,
                    "chosen_id": look_idx
                })

            if exs: source = "example_manager"

        # —— 示例块（编号空间与当前 acts 一致）——
        ex_blocks = []
        for i, ex in enumerate(exs[:k], 1):

            lines = [f"EXAMPLE {i}",
                     f"Task: {ex.get('task_name', '')}",
                     f"Goal: {ex.get('goal', '')}",
                     f"Observation: {ex.get('observation', '')}",
                     "Valid actions:"]
            vas = ex.get("valid_actions") or [a["action"] for a in acts]
            for j, a in enumerate(vas, 1):
                lines.append(f"[{j}] {a}")
            lines += ["Chosen:", f"ACTION: {ex.get('chosen_id', 1)}", ""]
            ex_blocks.append("\n".join(lines))

        # —— 当前回合 ——
        now = [f"Task: {task_name}",
               f"Goal: {goal}",
               f"Observation: {obs}",
               "Valid actions:"]
        for j, a in enumerate([x["action"] for x in acts], 1):
            now.append(f"[{j}] {a}")
        now.append("Chosen:\nACTION: ")

        # —— 记录元信息，便于 run_trial 打印一致性 ——
        self._last_icl_meta = {
            "examples_used": len(exs),
            "source": source,
            "valid_actions_len": len(acts),
        }

        parts = []
        parts.extend(primer)
        if ex_blocks: parts.extend(ex_blocks)
        parts.append("\n".join(now))
        return "\n".join(parts)

    def _ollama_options(self, temp_override=None) -> Dict:
        """Build Ollama options (deterministic mode for pure ICL baseline)."""
        if self.deterministic:
            temperature = 0.0
            top_p = 1.0
            top_k = 1
            seed = self.seed
        else:
            temperature = (temp_override if temp_override is not None else self.temperature)
            top_p = 0.95
            top_k = 10
            seed = self.seed  # ← 去掉随机抖动，保证可复现

        opts = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "num_predict": self.num_predict,
            "seed": seed,
            "num_ctx": self.num_ctx,
            "repeat_penalty": 1.0,
            "stop": ["\n"],
        }
        if self.ollama_num_thread is not None:
            opts["num_thread"] = int(self.ollama_num_thread)
        if self.ollama_num_batch is not None:
            opts["num_batch"] = int(self.ollama_num_batch)
        if self.ollama_use_mmap is not None:
            opts["use_mmap"] = bool(self.ollama_use_mmap)
        if self.ollama_f16_kv is not None:
            opts["f16_kv"] = bool(self.ollama_f16_kv)
        return opts

    def ollama_generate(self, prompt: str, temp_override=None) -> str:
        """Single Ollama generation"""
        if not self.llm_available:
            return "1"
        try:
            resp = self.session.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.model_tag,
                    "prompt": prompt,
                    "stream": False,
                    "keep_alive": "10m",
                    "options": self._ollama_options(temp_override),
                },
                timeout=self.request_timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            txt = data.get("response", "1")

            # Extract number
            txt = txt.strip()

            # 1) 优先形式：ACTION: i
            m = re.search(r"action\s*:\s*(\d+)", txt, flags=re.IGNORECASE)
            if m:
                return m.group(1)

            # 2) 兜底：任意数字
            m = re.search(r"\b(\d+)\b", txt)
            if m:
                return m.group(1)

            return "1"


        except Exception as e:
            if not self.quiet:
                print(f"Generate error: {e}")
            return "1"

    def generate_llm_response(self, prompt: str) -> str:
        """Generate with self-consistency voting if n>1"""
        n = max(1, self.consistency_n)
        if n == 1:
            return self.ollama_generate(prompt)

        # Multiple sampling
        votes = []
        for _ in range(n):
            out = self.ollama_generate(prompt, temp_override=0.4)
            votes.append(out)

        # Majority voting
        ids = []
        for v in votes:
            try:
                ids.append(int(v))
            except:
                ids.append(1)

        if not ids:
            return "1"

        cnt = Counter(ids)
        best = max(cnt.items(), key=lambda x: (x[1], -x[0]))[0]
        return str(best)

    def parse_action_id(self, raw_output: str, num_actions: int) -> Optional[int]:
        """Parse action ID from LLM output (returns 0-based index)"""
        try:
            i = int(raw_output)
            if 1 <= i <= num_actions:
                return i - 1
        except:
            pass
        return 0  # Default to first action

    def _get_valid_actions_text(self) -> list[str]:
        """
        返回当前步的自然语言动作列表。
        优先使用标准 API；都没有时，从组合模板里抽取，并做最小过滤。
        """
        # 1) 标准 API（通常存在）
        for name in ("getValidActions", "get_valid_actions", "get_valid_actions_text"):
            if hasattr(self.env, name):
                v = getattr(self.env, name)()
                if isinstance(v, list) and (not v or isinstance(v[0], str)):
                    return v

        # 2) 回退：从组合模板抽取 'action' 字段，并过滤掉明显的图编辑模板
        actions = []
        if hasattr(self.env, "get_valid_action_object_combinations_with_templates"):
            combos = self.env.get_valid_action_object_combinations_with_templates() or []
            for c in combos:
                if isinstance(c, dict) and isinstance(c.get("action"), str):
                    actions.append(c["action"].strip())

        DROP_PREFIXES = ("connect ", "mix ", "move ", "disconnect", "wait1", "reset task", "task")
        actions = [a for a in actions if a and not a.lower().startswith(DROP_PREFIXES)]
        return actions or ["look around"]

    def run_trial(self, task_name: str, variation_id: int) -> Dict:
        """Run single trial (记录每步 observation；返回完整 action_history)"""
        self.env.load(task_name, variation_id, "")
        obs, info = self.env.reset()
        goal = self.env.get_task_description()
        cumulative_reward = 0.0
        steps = 0
        action_history = []

        for t in range(self.step_limit):
            # 记录执行前的观测，供选择器/检索使用
            obs_before = obs

            # 1) 获取环境给的"合法动作"——保持**原始顺序**，不重排、不截断
            acts_raw = None
            if hasattr(self.env, "get_valid_action_object_combinations_with_templates"):
                acts_raw = self.env.get_valid_action_object_combinations_with_templates() or []
            # 回退到"文本动作列表"
            if not acts_raw:
                text_list = self._get_valid_actions_text() or ["look around"]
                acts = [{"action": a} for a in text_list]
            else:
                # env 返回的是包含 'action' 字段的字典列表；保持原样
                acts = []
                for x in acts_raw:
                    if isinstance(x, dict) and isinstance(x.get("action"), str):
                        acts.append({"action": x["action"]})
                if not acts:
                    acts = [{"action": "look around"}]

            if t == 0 and not self.quiet:
                print(f"[DEBUG] valid_actions(total): {len(acts)}, sample: {[a['action'] for a in acts[:5]]}")

            # 2) 构造提示（只用这份 acts），生成并解析编号（按 len(acts)）
            prompt = self.build_icl_prompt(obs_before, goal, acts, task_name, t)
            raw = self.generate_llm_response(prompt)
            idx = self.parse_action_id(raw, len(acts))
            if not (isinstance(idx, int) and 0 <= idx < len(acts)):
                idx = 0

            # 3) 选出并执行（仍然只用这份 acts）
            chosen = acts[idx]
            action_text = chosen.get("action", "look around")
            if t < 2 and not self.quiet:
                print(f"\nStep {t + 1}: LLM chose #{raw} -> action: '{action_text}'")

            # 先把"动作前观测 + 动作 + 原始输出"写入轨迹
            action_history.append({
                "step": t,
                "observation": obs_before,
                "action": action_text,
                "llm_output": raw
            })

            # 再执行动作，推进环境
            obs, reward, done, info = self.env.step(action_text)
            cumulative_reward += reward
            steps += 1

            if done:
                break

        success = cumulative_reward > 0

        # Save successful trajectory（动态小池保留 observation）
        if success and self.dynamic_examples:
            self.successful_trajectories.append({
                'task_name': task_name,
                'goal': goal,
                'steps': [{'action': h['action'], 'observation': h.get('observation', '')}
                          for h in action_history[:3]],
                'reward': cumulative_reward
            })
            if len(self.successful_trajectories) > 20:
                self.successful_trajectories = self.successful_trajectories[-20:]

        return {
            "task_name": task_name,
            "variation_id": variation_id,
            "goal": goal,  # 方便后续导出
            "steps": steps,
            "cumulative_reward": cumulative_reward,
            "success": success,
            "action_history": action_history,  # 不再截断
        }

    def run_evaluation(self) -> Dict:
        """Run full evaluation + checkpoint & selector-corpus export"""
        if self.llm_available:
            _ = self.generate_llm_response("1")  # Warmup

        # ===== BEGIN PATCH: 外部清单优先 + 数量限制（先 limit 再分片） =====
        manifest = self._load_manifest_from_json(self._manifest_json)
        if not manifest:
            # 回退到内部生成（不推荐，但用于容错）
            manifest = self.generate_manifest()

        # 统一在分片前做 limit（冒烟=10；全量=2120）
        if self._limit_total and self._limit_total > 0:
            manifest = manifest[: self._limit_total]
        # ===== END PATCH =====

        if self.num_shards > 1:
            manifest = [e for i, e in enumerate(manifest)
                        if (i % self.num_shards) == self.shard_index]

        print(f"\n{'=' * 60}")
        print("LLM ICL EVALUATION (Ollama) - PURE BASELINE")
        print(f"{'=' * 60}")
        print(f"Mode: ICL-{self.num_examples}shot")
        print(f"Deterministic: {self.deterministic}")
        print(f"Model: {self.model_tag}")
        print(f"Examples loaded: {len(self.example_manager.examples)} tasks")
        print(f"Retrieval-augmented ICL: {self.icl_retrieval}")
        print(f"Self-consistency voting: n={self.consistency_n}")
        print(f"Trials: {len(manifest)}")
        print("-" * 60)

        results = []
        # —— 新增：每多少条保存一个检查点（进度文件 + 选择器语料）——
        CHECKPOINT_EVERY = 200  # 可改为 100/500

        for i, entry in enumerate(manifest):
            do_print = not self.quiet or ((i + 1) % 10 == 0) or (i == 0) or ((i + 1) == len(manifest))
            if do_print:
                print(f"[{i + 1}/{len(manifest)}] {entry['task_name']} var{entry['variation_id']}", end=' ')

            tr = self.run_trial(entry["task_name"], entry["variation_id"])
            results.append(tr)

            if do_print:
                status = "✓" if tr['success'] else "✗"
                print(f"R={tr['cumulative_reward']:.2f} {status}")

            # === 新增：检查点保存（进度 JSON + 选择器语料 JSON），无需等全量结束 ===
            if (i + 1) % CHECKPOINT_EVERY == 0:
                ts_prog = datetime.now().strftime("%Y%m%d_%H%M%S")
                # 1) 进度文件：只含配置 & 当前累计 trials（不做汇总指标）
                progress = {
                    "config": {
                        "backend": "ollama",
                        "model": self.model_tag,
                        "mode": f"icl-{self.num_examples}shot-pure-baseline",
                        "deterministic": self.deterministic,
                        "seed": self.seed,
                        "step_limit": self.step_limit,
                        "max_vars_per_task": self.max_vars_per_task,
                        "num_trials": i + 1,
                        "num_examples": self.num_examples,
                        "dynamic_examples": self.dynamic_examples,
                        "icl_retrieval": self.icl_retrieval,
                        "consistency_n": self.consistency_n,
                        "icl_mmr_lambda": self.icl_mmr_lambda,
                    },
                    "metrics": {},
                    "trials": results,
                }
                fn_prog = f"{self.output_dir}/llm_icl_pure_{self.num_examples}shot_progress_{i + 1}_{ts_prog}.json"
                os.makedirs(self.output_dir, exist_ok=True)
                with open(fn_prog, "w", encoding="utf-8") as f:
                    json.dump(progress, f, indent=2, ensure_ascii=False)

                # 2) 选择器语料：精简字段，选择器可直接读取
                selector_trials = []
                for titem in results:
                    selector_trials.append({
                        "task_name": titem.get("task_name"),
                        "variation_id": titem.get("variation_id"),
                        "goal": titem.get("goal", ""),
                        "success": titem.get("success", False),
                        "cumulative_reward": titem.get("cumulative_reward", 0.0),
                        "action_history": [
                            {
                                "step": h.get("step"),
                                "observation": h.get("observation", ""),
                                "action": h.get("action", ""),
                                "llm_output": h.get("llm_output", "")
                            } for h in (titem.get("action_history", []) or [])
                        ]
                    })
                selector_obj = {"trials": selector_trials}
                fn_sel = fn_prog.replace(".json", ".selector_corpus.json")
                with open(fn_sel, "w", encoding="utf-8") as f:
                    json.dump(selector_obj, f, indent=2, ensure_ascii=False)

                if not self.quiet:
                    print(f"\n[checkpoint] saved: {fn_prog}")
                    print(f"[selector-corpus] saved: {fn_sel}")

        # ===== 计算最终指标（保持原逻辑）=====
        rewards = [r["cumulative_reward"] for r in results]
        mean_reward = np.mean(rewards) if rewards else 0.0
        se_reward = (np.std(rewards) / np.sqrt(len(rewards))) if len(rewards) > 0 else 0.0

        task_rewards = {}
        for r in results:
            task_rewards.setdefault(r["task_name"], []).append(r["cumulative_reward"])
        task_means = [np.mean(v) for v in task_rewards.values()] if task_rewards else []
        macro_mean = np.mean(task_means) if task_means else 0.0
        macro_se = (np.std(task_means) / np.sqrt(len(task_means))) if len(task_means) > 1 else 0.0

        cfg = f"icl_pure_{self.num_examples}shot_{self.model_tag}_{self.seed}"
        config_hash = hashlib.sha256(cfg.encode()).hexdigest()[:16]

        return {
            "config": {
                "backend": "ollama",
                "model": self.model_tag,
                "mode": f"icl-{self.num_examples}shot-pure-baseline",
                "deterministic": self.deterministic,
                "seed": self.seed,
                "step_limit": self.step_limit,
                "max_vars_per_task": self.max_vars_per_task,
                "num_trials": len(manifest),
                "num_examples": self.num_examples,
                "dynamic_examples": self.dynamic_examples,
                "icl_retrieval": self.icl_retrieval,
                "consistency_n": self.consistency_n,
                "icl_mmr_lambda": self.icl_mmr_lambda,
                "config_hash": config_hash,
            },
            "metrics": {
                "mean_reward": mean_reward,
                "mean_se": se_reward,
                "macro_mean": macro_mean,
                "macro_se": macro_se,
                "num_tasks": len(task_rewards),
                "success_rate": sum(1 for r in results if r["success"]) / len(results) if results else 0.0,
            },
            "trials": results,
        }

    def save_results(self, output: Dict):
        """Save results to file"""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(self.output_dir, exist_ok=True)
        fn = f"{self.output_dir}/pure_icl_{self.num_examples}shot_{ts}.json"
        with open(fn, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to: {fn}")
        print("\n" + "=" * 60)
        print("PURE ICL BASELINE EVALUATION COMPLETE")
        print("=" * 60)
        print(f"Mode: ICL-{self.num_examples}shot (Pure Baseline)")
        print(f"Deterministic: {output['config']['deterministic']}")
        print(f"Mean Reward: {output['metrics']['mean_reward']:.2f} ± {output['metrics']['mean_se']:.2f}")
        print(f"Macro Mean: {output['metrics']['macro_mean']:.2f} ± {output['metrics']['macro_se']:.2f}")
        print(f"Success Rate: {output['metrics']['success_rate'] * 100:.1f}%")


def main():
    ap = argparse.ArgumentParser(description="ScienceWorld Pure ICL Baseline Evaluation")

    # Basic
    ap.add_argument("--jar-path", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--step-limit", type=int, default=30)
    ap.add_argument("--output-dir", type=str, default="eval_results")
    ap.add_argument("--model", type=str, default="llama3.2:3b-instruct-q4_K_M")
    ap.add_argument("--ollama-host", type=str, default="http://localhost:11434")
    ap.add_argument("--max-vars-per-task", type=int, default=None)
    ap.add_argument("--num-predict", type=int, default=12)  # 50 -> 12
    ap.add_argument("--request-timeout", type=int, default=120)

    # ICL
    ap.add_argument("--examples-file", type=str, default=None)
    ap.add_argument("--num-examples", type=int, default=2)
    ap.add_argument("--dynamic-examples", action="store_true", default=False)
    ap.add_argument("--use-simple-prompt", action="store_true")

    # RA-ICL
    ap.add_argument("--consistency-n", type=int, default=1,
                    help="votes per step for self-consistency ICL (>=1)")
    ap.add_argument("--icl-retrieval", action="store_true",
                    help="enable retrieval-augmented ICL")
    ap.add_argument("--icl-mmr-lambda", type=float, default=0.3,
                    help="MMR tradeoff for diversity in retrieved examples")

    # Speed
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--num-ctx", type=int, default=2048)

    # Manifest
    ap.add_argument("--manifest-json", type=str, default=_DEFAULT_MANIFEST,
                    help="Path to external manifest (json/jsonl or dict with 'trials')")
    ap.add_argument("--limit-total", type=int, default=_RUN_TRIALS_LIMIT,
                    help="Hard cap on total trials. Applied before sharding.")

    # Pure ICL Baseline
    ap.add_argument("--deterministic", action="store_true",
                    help="Use deterministic decoding for pure ICL baseline (temp=0, top_p=1, top_k=1, no seed jitter).")

    ap.add_argument("--num-examples-list", type=str, default="2,4,8",
                    help="Comma-separated K list, e.g. '2,4,8'")

    args = ap.parse_args()

    # 若未显式提供 examples_file，则默认用你上传的 selector_corpus
    if not args.examples_file:
        args.examples_file = r"C:\Users\ylili\Desktop\ASL\abschlussarbeit\sciworld\eval_results\selector_corpus@20251103_073850.full.json"

    ev = LLMICLEvaluator(
        jar_path=args.jar_path,
        seed=args.seed,
        step_limit=args.step_limit,
        output_dir=args.output_dir,
        model_tag=args.model,
        ollama_host=args.ollama_host,
        max_vars_per_task=args.max_vars_per_task,
        num_predict=args.num_predict,
        request_timeout=args.request_timeout,
        num_ctx=args.num_ctx,
        examples_file=args.examples_file,
        num_examples=args.num_examples,
        dynamic_examples=args.dynamic_examples,
        use_simple_prompt=args.use_simple_prompt,
        consistency_n=args.consistency_n,
        icl_retrieval=args.icl_retrieval,
        icl_mmr_lambda=args.icl_mmr_lambda,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        quiet=args.quiet,
        manifest_json=args.manifest_json,
        limit_total=args.limit_total,
        deterministic=args.deterministic,
    )

    # === Use code-config SHOT_LIST (overrides command-line parameter) ===
    try:
        shot_list = _SHOT_LIST
    except NameError:
        # Fallback to command-line parameter if code-config not available
        if args.num_examples_list:
            shot_list = [int(x) for x in args.num_examples_list.split(",") if x.strip()]
        else:
            shot_list = [args.num_examples]

    # Run multiple K shots in one go (from code-config)
    for k in shot_list:
        ev.num_examples = int(k)
        if hasattr(ev, "example_manager"):
            ev.example_manager.max_examples = int(k)
        out = ev.run_evaluation()
        if isinstance(out, dict) and "error" not in out:
            ev.save_results(out)


if __name__ == "__main__":
    main()