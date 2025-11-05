#!/usr/bin/env python3
"""
ScienceWorld Critical-Step ICL Baseline Evaluation
Clean baseline: only critical-step selection matters; no guardrails/prioritization by default
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
# Critical-step selector file & its m (hard-bound in code, not via CLI)
# Format: selector_{selector_name}_m{value}.json
# Examples:
#   - selector_gemini2.5pro_m10.json
#   - selector_sonnet4.5_m20.json
#   - selector_chatgpt5_m10.json
_CRIT_EXAMPLES_FILE = r"C:\Users\ylili\Desktop\ASL\abschlussarbeit\sciworld\selector_sonnet4.5_m10.json"
_CRIT_M = 0.10  # <- keep consistent with *_m10.json ; change to 0.10/0.20/0.30 when you switch files

# Shots to run in one go (no need to change via terminal)
_SHOT_LIST = [2, 4, 8]

# Baseline vs Exploration
_BASELINE_MODE = True  # True = fair baseline: temp=0, n=1; False = exploration: temp≈0.2, n=3

# Guardrails & retrieval (kept OFF for baseline; flip to True for exploration experiments)
_GUARDRAILS = False
_RETRIEVAL = False
_MMR_LAMBDA = 0.30

# Task manifest and limit
_DEFAULT_MANIFEST = r"C:\Users\ylili\Desktop\ASL\abschlussarbeit\sciworld\agenttraj-l-train-0.sciworld_only.json"
_RUN_TRIALS_LIMIT = 2120


# ===== END EXPERIMENT SWITCH =====


def _extract_selector_name(filepath: str) -> str:
    """
    Extract selector name from example file path.
    Expected format: selector_{name}_m{value}.json
    Examples:
        selector_gemini2.5pro_m10.json -> gemini2.5pro
        selector_sonnet4.5_m20.json -> sonnet4.5
        selector_chatgpt5_m10.json -> chatgpt5
    """
    if not filepath:
        return "unknown"

    filename = os.path.basename(filepath)
    # Remove extension
    name_without_ext = os.path.splitext(filename)[0]

    # Try to match pattern: selector_{name}_m{value}
    import re
    match = re.match(r'selector_(.+?)_m\d+', name_without_ext)
    if match:
        return match.group(1)

    # Fallback: return filename without extension
    return name_without_ext


def _ceil(x):
    """Ceiling function for budget calculation"""
    import math
    return int(math.ceil(x))


class ICLExampleManager:
    """Manage and select ICL examples with critical-step support"""

    def __init__(self, examples_file: str = None, max_examples: int = 3):
        self.max_examples = max_examples
        self.examples = defaultdict(list)
        self.loaded = False
        self.examples_file = examples_file  # Store for reference

    def load_examples(self, filepath: str, critical_only: bool = True, m: float = 0.2):
        """Load critical-step examples with m-budget validation"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            trials = []
            if isinstance(data, dict) and "trials" in data:
                trials = data["trials"]
            elif isinstance(data, list):
                trials = data
            else:
                for _, lst in (data.items() if isinstance(data, dict) else []):
                    if isinstance(lst, list):
                        trials.extend(lst)

            skipped_count = 0
            for tr in trials:
                task_name = tr.get('task_name') or tr.get('task') or tr.get('env_name')
                if not task_name:
                    continue

                T = len(tr.get('action_history') or [])
                crit = tr.get("selected_steps") or tr.get("critical_steps") or []

                if critical_only and not crit:
                    # No critical info -> skip
                    skipped_count += 1
                    continue

                if crit:
                    if T <= 0:
                        continue
                    budget = _ceil(m * T)
                    if len(crit) > budget:
                        # Skip over-budget trials instead of truncating
                        if skipped_count < 5:  # Only print first few warnings
                            print(
                                f"[warn] skip over-budget trial: steps={len(crit)} > ceil(m*T)={budget} ({task_name})")
                        skipped_count += 1
                        continue
                    steps_src = crit
                else:
                    # Fallback only when explicitly allowed (critical_only=False)
                    steps_src = tr.get('action_history') or []

                # Normalize steps
                ex_steps = []
                for s in steps_src:
                    if isinstance(s, dict):
                        act = s.get('action', '')
                        obs = s.get('observation') or s.get('obs_before') or s.get('obs_after') or ''
                    else:
                        act, obs = str(s), ''
                    ex_steps.append({'action': act, 'observation': (obs or '')[:200]})

                if not ex_steps:
                    continue

                self.examples[task_name].append({
                    'task_name': task_name,
                    'goal': tr.get('goal') or tr.get('task_description') or f"Complete {task_name.replace('-', ' ')}",
                    'steps': ex_steps,
                    'T': T,
                    'm': m,
                    'is_critical': bool(crit),
                    'trial_id': f"{task_name}:{tr.get('variation_id', -1)}"
                })

            self.loaded = True
            print(
                f"[critical-load] tasks={len(self.examples)} file={os.path.basename(filepath)} m={m} critical_only={critical_only}")
            if skipped_count > 0:
                print(f"[critical-load] skipped {skipped_count} trials (no critical data or over-budget)")
        except Exception as e:
            print(f"Warning: Could not load examples from {filepath}: {e}")

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


class LLMCriticalStepEvaluator:
    """Critical-Step ICL evaluation for ScienceWorld"""

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
                 dynamic_examples: bool = False,
                 use_simple_prompt: bool = False,

                 # RA-ICL specific
                 consistency_n: int = 1,
                 icl_retrieval: bool = False,
                 icl_mmr_lambda: float = 0.3,

                 # Critical-step specific
                 critical_m: float = 0.2,
                 critical_only: bool = True,
                 guardrails: bool = False,
                 deterministic: bool = False,

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
                 ):

        # === Step 1: Store incoming parameters ===
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
        self.successful_trajectories = []

        # RA-ICL config
        self.consistency_n = consistency_n
        self.icl_retrieval = icl_retrieval
        self.icl_mmr_lambda = icl_mmr_lambda
        self.temperature = 0.4

        # Critical-step config (from parameters)
        self.examples_file = examples_file
        self.critical_m = float(critical_m)
        self.critical_only = bool(critical_only)
        self.guardrails = bool(guardrails)
        self.deterministic = bool(deterministic)

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

        # === Step 2: OVERRIDE by top-of-file code-config ===
        try:
            self.examples_file = _CRIT_EXAMPLES_FILE
            self.critical_m = float(_CRIT_M)
            self.guardrails = bool(_GUARDRAILS)
            # Extract selector name from examples file
            self.selector_name = _extract_selector_name(_CRIT_EXAMPLES_FILE)
            # Baseline vs exploration presets
            if _BASELINE_MODE:
                self.deterministic = True
                self.consistency_n = 1
                self.temperature = 0.0
                self.icl_retrieval = False
            else:
                self.deterministic = False
                self.consistency_n = max(3, self.consistency_n or 3)
                self.temperature = 0.2
                self.icl_retrieval = bool(_RETRIEVAL)
            self.icl_mmr_lambda = float(_MMR_LAMBDA)

            # Manifest config
            if not self.manifest_json:
                self.manifest_json = _DEFAULT_MANIFEST
            if not self.limit_total:
                self.limit_total = _RUN_TRIALS_LIMIT
        except NameError:
            # Fallback if code-config not available
            self.selector_name = _extract_selector_name(self.examples_file) if self.examples_file else "unknown"

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

        # === Step 3: Load examples with critical-step awareness ===
        # Create manager without auto-loading
        self.example_manager = ICLExampleManager(None, self.num_examples)
        if self.examples_file and os.path.exists(self.examples_file):
            self.example_manager.load_examples(
                self.examples_file,
                critical_only=self.critical_only,
                m=self.critical_m
            )

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

    def _filter_leakage(self, items: list, task_name: str, var_id: int):
        """Filter out examples from the same (task_name, variation_id) to prevent leakage"""
        ban = f"{task_name}:{var_id}"
        out = []
        for it in items:
            tid = it.get('trial_id') or ""
            if tid == ban:
                continue
            out.append(it)
        return out

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

    def retrieve_icl_examples(self, goal: str, obs: str, acts_now: list, k: int, task_name: str = "") -> list:
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
                        "action_text": st.get("action", ""),
                        "trial_id": ex.get("trial_id", "")
                    })

        if not pool:
            return []

        # Filter leakage
        pool = self._filter_leakage(pool, task_name, getattr(self, "_cur_var_id", -1))

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

    def _load_manifest_from_json(self, path: str) -> List[Dict]:
        """Load external manifest from json/jsonl"""

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
        if not path or not os.path.exists(path):
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

        # Filter and deduplicate
        try:
            all_task_names = set(self.env.get_task_names())
            manifest = [m for m in manifest if m["task_name"] in all_task_names]
        except Exception:
            pass

        seen = set()
        deduped = []
        for m in manifest:
            key = (m["task_name"], m["variation_id"])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(m)
        return deduped

    def generate_manifest(self) -> List[Dict]:
        """Generate task manifest (fallback)"""
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
        """Build ICL prompt with critical-step examples (clean baseline: no strategy rules)"""
        k = self.num_examples
        exs, source = [], "none"

        if self.icl_retrieval:
            exs = self.retrieve_icl_examples(goal, obs, acts, k=k, task_name=task_name)
            if exs:
                source = "retrieval"

        if not exs and self.example_manager:
            fb = (self.example_manager.get_examples(task_name, k) or [])[:k]
            for ex in fb:
                vas = [a["action"] for a in acts]
                try:
                    look_idx = next(i for i, a in enumerate(vas, 1) if a.lower().startswith("look"))
                except StopIteration:
                    look_idx = 1

                exs.append({
                    "task_name": task_name,
                    "goal": ex.get("goal", ""),
                    "observation": (ex.get("steps", [{}])[0].get("observation", "")),
                    "valid_actions": vas,
                    "chosen_id": look_idx
                })
            if exs:
                source = "example_manager"

        # EXAMPLE blocks (critical steps)
        ex_blocks = []
        for i, ex in enumerate(exs[:k], 1):
            lines = [
                f"EXAMPLE {i} (critical steps)",
                f"Task: {ex.get('task_name', '')}",
                f"Goal: {ex.get('goal', '')}",
                f"Observation: {ex.get('observation', '')}",
                "Valid actions:"
            ]
            vas = ex.get("valid_actions") or [a["action"] for a in acts]
            for j, a in enumerate(vas, 1):
                lines.append(f"[{j}] {a}")
            lines += [
                "Chosen:",
                f"ACTION: {ex.get('chosen_id', 1)}",
                ""
            ]
            ex_blocks.append("\n".join(lines))

        # Current round
        now = [
            f"Task: {task_name}",
            f"Goal: {goal}",
            f"Observation: {obs}",
            "Valid actions:"
        ]
        for j, a in enumerate([x["action"] for x in acts], 1):
            now.append(f"[{j}] {a}")
        now.append("Chosen:\nACTION: ")

        # Record metadata
        self._last_icl_meta = {
            "examples_used": len(exs),
            "source": source,
            "valid_actions_len": len(acts),
        }

        parts = []
        if ex_blocks:
            parts.extend(ex_blocks)
        parts.append("\n".join(now))

        # Note: No extra output constraint to keep consistent with pure ICL baseline
        return "\n".join(parts)

    def _ollama_options(self, temp_override=None) -> Dict:
        """Build Ollama options (deterministic or exploration mode)"""
        if self.deterministic:
            temperature, top_p, top_k, seed = 0.0, 1.0, 1, self.seed
        else:
            temperature = (temp_override if temp_override is not None else self.temperature)
            top_p, top_k, seed = 0.95, 10, self.seed  # No seed jitter for reproducibility

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

            # 1) Prefer format: ACTION: i
            m = re.search(r"action\s*:\s*(\d+)", txt, flags=re.IGNORECASE)
            if m:
                return m.group(1)

            # 2) Fallback: any number
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
            out = self.ollama_generate(prompt, temp_override=self.temperature)
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
        """Get current step's valid actions as text list"""
        # 1) Standard API
        for name in ("getValidActions", "get_valid_actions", "get_valid_actions_text"):
            if hasattr(self.env, name):
                v = getattr(self.env, name)()
                if isinstance(v, list) and (not v or isinstance(v[0], str)):
                    return v

        # 2) Fallback: extract from combinations
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
        """Run single trial"""
        # Record current variation for leakage filtering
        self._cur_var_id = variation_id

        self.env.load(task_name, variation_id, "")
        obs, info = self.env.reset()
        goal = self.env.get_task_description()
        cumulative_reward = 0.0
        steps = 0
        action_history = []

        for t in range(self.step_limit):
            obs_before = obs

            # 1) Get environment's valid actions - keep original order, no truncation
            acts_raw = None
            if hasattr(self.env, "get_valid_action_object_combinations_with_templates"):
                acts_raw = self.env.get_valid_action_object_combinations_with_templates() or []

            # Process actions
            if acts_raw:
                actions_now = []
                for x in acts_raw:
                    if isinstance(x, dict) and isinstance(x.get("action"), str):
                        actions_now.append(x["action"])
            else:
                actions_now = self._get_valid_actions_text() or []

            # Fallback
            if not actions_now:
                actions_now = ["look around"]

            # --- Guardrails optional (OFF for baseline) ---
            if self.guardrails:
                # Apply filtering/prioritization if guardrails enabled
                # (You would implement _filter_and_prioritize_actions if needed)
                pass

            acts = [{"action": a} for a in actions_now]

            if t == 0 and not self.quiet:
                print(f"[DEBUG] valid_actions(total): {len(acts)}, sample: {[a['action'] for a in acts[:5]]}")

            # 2) Build prompt and get LLM response
            prompt = self.build_icl_prompt(obs_before, goal, acts, task_name, t)
            raw = self.generate_llm_response(prompt)
            idx = self.parse_action_id(raw, len(acts))
            if not (isinstance(idx, int) and 0 <= idx < len(acts)):
                idx = 0

            # 3) Execute action
            chosen = acts[idx]
            action_text = chosen.get("action", "look around")
            if t < 2 and not self.quiet:
                print(f"\nStep {t + 1}: LLM chose #{raw} -> action: '{action_text}'")

            # Record action history
            action_history.append({
                "step": t,
                "observation": obs_before,
                "action": action_text,
                "llm_output": raw
            })

            # Execute
            obs, reward, done, info = self.env.step(action_text)
            cumulative_reward += reward
            steps += 1

            if done:
                break

        success = cumulative_reward > 0

        # Save successful trajectory (optional, for dynamic examples)
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
            "goal": goal,
            "steps": steps,
            "cumulative_reward": cumulative_reward,
            "success": success,
            "action_history": action_history,
        }

    def run_evaluation(self) -> Dict:
        """Run full evaluation"""
        if self.llm_available:
            _ = self.generate_llm_response("1")  # Warmup

        # Load manifest
        manifest = self._load_manifest_from_json(self.manifest_json)
        if not manifest:
            manifest = self.generate_manifest()

        # Apply limit
        if self.limit_total and self.limit_total > 0:
            manifest = manifest[:self.limit_total]

        # Sharding
        if self.num_shards > 1:
            manifest = [e for i, e in enumerate(manifest)
                        if (i % self.num_shards) == self.shard_index]

        mode_str = "BASELINE" if _BASELINE_MODE else "EXPLORATION"
        print(f"\n{'=' * 60}")
        print(f"CRITICAL-STEP ICL EVALUATION ({mode_str})")
        print(f"{'=' * 60}")
        print(f"Mode: Critical-Step ICL-{self.num_examples}shot")
        print(f"Selector: {getattr(self, 'selector_name', 'unknown')}")
        print(f"Baseline Mode: {_BASELINE_MODE}")
        print(f"Deterministic: {self.deterministic}")
        print(f"Model: {self.model_tag}")
        print(f"Critical m: {self.critical_m}")
        print(f"Examples file: {os.path.basename(self.examples_file)}")
        print(f"Examples loaded: {len(self.example_manager.examples)} tasks")
        print(f"Guardrails: {self.guardrails}")
        print(f"Retrieval-augmented ICL: {self.icl_retrieval}")
        print(f"Self-consistency voting: n={self.consistency_n}")
        print(f"Temperature: {self.temperature}")
        print(f"Trials: {len(manifest)}")
        print("-" * 60)

        results = []
        CHECKPOINT_EVERY = 200

        for i, entry in enumerate(manifest):
            do_print = not self.quiet or ((i + 1) % 10 == 0) or (i == 0) or ((i + 1) == len(manifest))
            if do_print:
                print(f"[{i + 1}/{len(manifest)}] {entry['task_name']} var{entry['variation_id']}", end=' ')

            tr = self.run_trial(entry["task_name"], entry["variation_id"])
            results.append(tr)

            if do_print:
                status = "✓" if tr['success'] else "✗"
                print(f"R={tr['cumulative_reward']:.2f} {status}")

            # Checkpoint saving
            if (i + 1) % CHECKPOINT_EVERY == 0:
                ts_prog = datetime.now().strftime("%Y%m%d_%H%M%S")
                progress = {
                    "config": {
                        "backend": "ollama",
                        "model": self.model_tag,
                        "mode": f"critical-step-icl-{self.num_examples}shot",
                        "selector": getattr(self, 'selector_name', 'unknown'),
                        "baseline_mode": _BASELINE_MODE,
                        "deterministic": self.deterministic,
                        "critical_m": self.critical_m,
                        "critical_only": self.critical_only,
                        "guardrails": self.guardrails,
                        "seed": self.seed,
                        "step_limit": self.step_limit,
                        "num_examples": self.num_examples,
                        "icl_retrieval": self.icl_retrieval,
                        "consistency_n": self.consistency_n,
                        "temperature": self.temperature,
                    },
                    "metrics": {},
                    "trials": results,
                }
                selector_str = getattr(self, 'selector_name', 'unknown')
                fn_prog = f"{self.output_dir}/llm_critical_{self.num_examples}shot_{selector_str}_m{int(self.critical_m * 100)}_progress_{i + 1}_{ts_prog}.json"
                os.makedirs(self.output_dir, exist_ok=True)
                with open(fn_prog, "w", encoding="utf-8") as f:
                    json.dump(progress, f, indent=2, ensure_ascii=False)

                if not self.quiet:
                    print(f"\n[checkpoint] saved: {fn_prog}")

        # Calculate final metrics
        rewards = [r["cumulative_reward"] for r in results]
        mean_reward = np.mean(rewards) if rewards else 0.0
        se_reward = (np.std(rewards) / np.sqrt(len(rewards))) if len(rewards) > 0 else 0.0

        task_rewards = {}
        for r in results:
            task_rewards.setdefault(r["task_name"], []).append(r["cumulative_reward"])
        task_means = [np.mean(v) for v in task_rewards.values()] if task_rewards else []
        macro_mean = np.mean(task_means) if task_means else 0.0
        macro_se = (np.std(task_means) / np.sqrt(len(task_means))) if len(task_means) > 1 else 0.0

        cfg = f"critical_{self.num_examples}shot_{self.model_tag}_{self.seed}_m{self.critical_m}"
        config_hash = hashlib.sha256(cfg.encode()).hexdigest()[:16]

        return {
            "config": {
                "backend": "ollama",
                "model": self.model_tag,
                "mode": f"critical-step-icl-{self.num_examples}shot",
                "selector": getattr(self, 'selector_name', 'unknown'),
                "baseline_mode": _BASELINE_MODE,
                "deterministic": self.deterministic,
                "critical_m": self.critical_m,
                "critical_only": self.critical_only,
                "guardrails": self.guardrails,
                "examples_file": os.path.basename(self.examples_file),
                "seed": self.seed,
                "step_limit": self.step_limit,
                "max_vars_per_task": self.max_vars_per_task,
                "num_trials": len(manifest),
                "num_examples": self.num_examples,
                "dynamic_examples": self.dynamic_examples,
                "icl_retrieval": self.icl_retrieval,
                "consistency_n": self.consistency_n,
                "icl_mmr_lambda": self.icl_mmr_lambda,
                "temperature": self.temperature,
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
        mode_suffix = "baseline" if _BASELINE_MODE else "exploration"

        # Build filename with selector name
        # Format: llm_critical_{K}shot_{selector}_m{m_value}_{mode}_{timestamp}.json
        selector_str = getattr(self, 'selector_name', 'unknown')
        fn = f"{self.output_dir}/llm_critical_{self.num_examples}shot_{selector_str}_m{int(self.critical_m * 100)}_{mode_suffix}_{ts}.json"

        with open(fn, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to: {fn}")
        print("\n" + "=" * 60)
        print("CRITICAL-STEP ICL EVALUATION COMPLETE")
        print("=" * 60)
        print(f"Mode: Critical-Step ICL-{self.num_examples}shot ({mode_suffix.upper()})")
        print(f"Selector: {selector_str}")
        print(f"Critical m: {self.critical_m}")
        print(f"Deterministic: {output['config']['deterministic']}")
        print(f"Guardrails: {output['config']['guardrails']}")
        print(f"Mean Reward: {output['metrics']['mean_reward']:.2f} ± {output['metrics']['mean_se']:.2f}")
        print(f"Macro Mean: {output['metrics']['macro_mean']:.2f} ± {output['metrics']['macro_se']:.2f}")
        print(f"Success Rate: {output['metrics']['success_rate'] * 100:.1f}%")


def main():
    ap = argparse.ArgumentParser(description="ScienceWorld Critical-Step ICL Baseline Evaluation")

    # Basic
    ap.add_argument("--jar-path", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--step-limit", type=int, default=30)
    ap.add_argument("--output-dir", type=str, default="eval_results")
    ap.add_argument("--model", type=str, default="llama3.2:3b-instruct-q4_K_M")
    ap.add_argument("--ollama-host", type=str, default="http://localhost:11434")
    ap.add_argument("--max-vars-per-task", type=int, default=None)
    ap.add_argument("--num-predict", type=int, default=12)
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

    # Critical-step switches (kept for completeness, but overridden by code config)
    ap.add_argument("--critical-m", type=float, default=None, help="(unused if code-config present)")
    ap.add_argument("--use-full-trajectory", action="store_false", dest="critical_only", default=True,
                    help="Use full trajectory instead of critical steps")
    ap.add_argument("--guardrails", action="store_true", default=False,
                    help="Enable rule-based action filtering / room routing (OFF for clean baseline)")
    ap.add_argument("--deterministic", action="store_true", default=False,
                    help="Deterministic decoding for baseline (temp=0, top_p=1, top_k=1, no seed jitter)")

    # Speed
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--num-ctx", type=int, default=2048)

    # Manifest
    ap.add_argument("--manifest-json", type=str, default=None,
                    help="Path to external manifest (json/jsonl or dict with 'trials')")
    ap.add_argument("--limit-total", type=int, default=None,
                    help="Hard cap on total trials. Applied before sharding.")

    args = ap.parse_args()

    # Build evaluator (args passed but may be overridden by code-config in __init__)
    ev = LLMCriticalStepEvaluator(
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
        critical_m=args.critical_m if args.critical_m else 0.2,
        critical_only=args.critical_only,
        guardrails=args.guardrails,
        deterministic=args.deterministic,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        quiet=args.quiet,
        manifest_json=args.manifest_json,
        limit_total=args.limit_total,
    )

    # Run multiple K shots in one go (from code-config)
    for k in _SHOT_LIST:
        ev.num_examples = int(k)
        if hasattr(ev, "example_manager"):
            ev.example_manager.max_examples = int(k)
        out = ev.run_evaluation()
        if isinstance(out, dict) and "error" not in out:
            ev.save_results(out)


if __name__ == "__main__":
    main()