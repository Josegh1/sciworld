#!/usr/bin/env python3
"""
ScienceWorld LLM Evaluation (Zero-Shot only, speed-optimized; protocol unchanged)

- 仍然是零样本（no ICL），不改论文口径：累计奖励 + 默认 30 步
- 不更换模型，不减少生成 token（默认 num_predict=12）
- 只做“工程级提速”：分片并行、静音日志、Ollama 推理吞吐参数（可选）
  * 分片并行：--num-shards N --shard-index i  （开多个进程并行跑不同切片）
  * 静音日志：--quiet  （仅每 50 条打印一次进度）
  * 吞吐参数（不影响解码策略）：--ollama-num-thread / --ollama-num-batch / --ollama-use-mmap / --ollama-f16-kv
"""

import json
import random
import numpy as np
import argparse
import hashlib
import os
import re
import requests
from typing import List, Dict
from datetime import datetime
from scienceworld import ScienceWorldEnv

SAVE_FULL_TRACE = True  # 导出selector用的全轨迹


class LLMZeroShotEvaluator:
    """Zero-shot LLM evaluation for ScienceWorld via local Ollama"""

    def __init__(self,
                 jar_path: str = None,
                 seed: int = 42,
                 step_limit: int = 30,
                 output_dir: str = "eval_results",
                 model_tag: str = "llama3.2:3b-instruct-q4_K_M",
                 ollama_host: str = "http://localhost:11434",
                 max_vars_per_task: int = None,
                 num_predict: int = 12,
                 request_timeout: int = 120,
                 num_ctx: int = 2048,

                 # Speed knobs (do NOT change decoding semantics)
                 num_shards: int = 1,
                 shard_index: int = 0,
                 quiet: bool = False,
                 ollama_num_thread: int = None,
                 ollama_num_batch: int = None,
                 ollama_use_mmap: bool = None,
                 ollama_f16_kv: bool = None,
                 manifest_json: str = None,
                 limit_total: int = None,
                 ):
        # Core config
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

        # Speed config
        self.num_shards = max(1, int(num_shards))
        self.shard_index = max(0, int(shard_index))
        self.quiet = bool(quiet)
        self.ollama_num_thread = ollama_num_thread
        self.ollama_num_batch = ollama_num_batch
        self.ollama_use_mmap = ollama_use_mmap
        self.ollama_f16_kv = ollama_f16_kv
        self.manifest_json = manifest_json
        self.limit_total = limit_total

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

    # ---------- Ollama ----------
    def test_ollama(self):
        """Ensure Ollama is reachable and model exists (pull if missing)."""
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
                    # 读到完成为止
                    for line in pr.iter_lines(decode_unicode=True):
                        if not line:
                            continue
                        try:
                            evt = json.loads(line)
                        except Exception:
                            continue
                        status = str(evt.get("status", "")).lower()
                        if status:
                            print("  ", status)
                        if evt.get("done") is True or status == "success":
                            break
                # re-check
                r2 = self.session.get(f"{self.ollama_host}/api/tags", timeout=10)
                r2.raise_for_status()
                names = [m.get("name") for m in r2.json().get("models", []) if isinstance(m, dict)]
                if self.model_tag not in names:
                    raise RuntimeError(f"Model {self.model_tag} not listed after pull")
            print(f"✓ Ollama ready: {self.model_tag}")
            self.llm_available = True
        except Exception as e:
            print(f"WARNING: Ollama not ready ({e}). Falling back (ACTION: 1).")
            self.llm_available = False

    # ---------- Manifest ----------
    def _load_manifest_from_json(self, path: str) -> List[Dict]:
        """Load external manifest from json/jsonl to standard [{'task_name':..., 'variation_id':...}, ...]."""

        def _norm_one(rec):
            # 容错字段名
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

        # 兼容 .json / .jsonl / {"trials":[...]}
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

        # 只保留在环境任务列表里的条目，避免越界
        try:
            all_task_names = set(self.env.get_task_names())
            manifest = [m for m in manifest if m["task_name"] in all_task_names]
        except Exception:
            pass

        # 去重（task_name, variation_id）
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
        SELECTED_TASKS = [
            "boil",
            "change-the-state-of-matter-of",
            "chemistry-mix",
            "chemistry-mix-paint-secondary-color",
            "chemistry-mix-paint-tertiary-color",
            "find-animal",
            "find-living-thing",
            "find-non-living-thing",
            "find-plant",
            "freeze",
            "grow-fruit",
            "grow-plant",
            "identify-life-stages-1",
            "identify-life-stages-2",
            "inclined-plane-determine-angle",
            "inclined-plane-friction-named-surfaces",
            "inclined-plane-friction-unnamed-surfaces",
            "lifespan-longest-lived",
            "lifespan-longest-lived-then-shortest-lived",
            "lifespan-shortest-lived",
            "measure-melting-point-known-substance",
            "measure-melting-point-unknown-substance",
            "melt",
            "mendelian-genetics-known-plant",
            "mendelian-genetics-unknown-plant",
            "power-component",
            "power-component-renewable-vs-nonrenewable-energy",
            "test-conductivity",
            "test-conductivity-of-unknown-substances",
            "use-thermometer",
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

    # ---------- Prompt/Decode ----------
    def canonicalize_actions(self, actions: List[Dict]) -> List[Dict]:
        return sorted(actions, key=lambda x: (x.get("action", ""), str(x)))

    def build_prompt(self, obs: str, goal: str, actions: List[Dict]) -> str:
        action_list = "\n".join([f"[{i + 1}] {a['action']}" for i, a in enumerate(actions)])
        return f"""Context:
Goal: {goal}
Observation: {obs}

Valid actions (enumerated):
{action_list}

Output exactly one line: ACTION: <ID>
No explanation. Do not invent actions.

ACTION: """

    def _ollama_options(self) -> Dict:
        """Build options dict; only set speed knobs when provided (to keep defaults stable)."""
        opts = {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
            "num_predict": self.num_predict,
            "seed": self.seed,
            "num_ctx": self.num_ctx,  # <- 用传入值
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

    def generate_llm_response(self, prompt: str) -> str:
        if not self.llm_available:
            return "ACTION: 1"
        try:
            resp = self.session.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.model_tag,
                    "prompt": prompt,
                    "stream": False,
                    "keep_alive": "10m",
                    "options": self._ollama_options(),
                },
                timeout=self.request_timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            txt = (data.get("response") or "ACTION: 1").replace("：", ":")
            first = txt.split("\n")[0].strip()
            if not first.upper().startswith("ACTION:"):
                m = re.search(r"ACTION:\s*(\d+)", txt, re.IGNORECASE)
                if m:
                    first = m.group(0)
                else:
                    m2 = re.search(r"\b(\d+)\b", txt)
                    first = f"ACTION: {m2.group(1)}" if m2 else "ACTION: 1"
            return first
        except requests.exceptions.Timeout:
            print("Ollama request timed out")
            return "ACTION: 1"
        except Exception as e:
            print(f"Ollama generation error: {e}")
            return "ACTION: 1"

    def parse_action_id(self, raw_output: str, num_actions: int) -> int:
        raw_output = (raw_output or "").replace("：", ":")
        m = re.match(r"^ACTION:\s*(\d+)", raw_output, re.IGNORECASE)
        if m:
            i = int(m.group(1))
            if 1 <= i <= num_actions:
                return i - 1
        m = re.search(r"\b(\d+)\b", raw_output)
        if m:
            i = int(m.group(1))
            if 1 <= i <= num_actions:
                return i - 1
        return 0  # 与原逻辑一致：解析失败回第 1 个动作

    # ---------- Episode ----------
    def run_trial(self, task_name: str, variation_id: int) -> Dict:
        self.env.load(task_name, variation_id, "")
        obs, info = self.env.reset()
        goal = self.env.get_task_description()
        cumulative_reward = 0.0
        steps = 0
        action_history = []
        for t in range(self.step_limit):
            # 在每一步执行前，记录"当前观测"
            prev_obs = obs

            valid = self.env.get_valid_action_object_combinations_with_templates()
            if not valid:
                valid = [{"action": "look around"}]
            acts = self.canonicalize_actions(valid)
            prompt = self.build_prompt(obs, goal, acts)
            raw = self.generate_llm_response(prompt)
            idx = self.parse_action_id(raw, len(acts))
            chosen = acts[idx]
            obs, reward, done, info = self.env.step(chosen["action"])
            action_history.append({
                "step": t,
                "obs_before": prev_obs,
                "action": chosen["action"],
                "llm_output": raw,
                "obs_after": obs,
                "reward": reward,
                "done": bool(done)
            })
            cumulative_reward += reward
            steps += 1
            if done:
                break
        return {
            "task_name": task_name,
            "variation_id": variation_id,
            "steps": steps,
            "cumulative_reward": cumulative_reward,
            "success": cumulative_reward > 0,
            "action_history": action_history if SAVE_FULL_TRACE else action_history[:5],
        }

    # ---------- Evaluation ----------
    def run_evaluation(self) -> Dict:
        if self.llm_available:
            # Warmup
            _ = self.generate_llm_response("ACTION: ")

        # 优先使用外部清单
        manifest = []
        if getattr(self, "manifest_json", None):
            manifest = self._load_manifest_from_json(self.manifest_json)

        # 回退到内置生成
        if not manifest:
            manifest = self.generate_manifest()

        if not manifest:
            return {"error": "No tasks selected"}

        # 可选：限制总量（如 2120）
        if getattr(self, "limit_total", None):
            try:
                limit_n = int(self.limit_total)
                if limit_n > 0:
                    manifest = manifest[:limit_n]
            except Exception:
                pass

        # 分片并行：只过滤列表，不改变任何单个 trial 的逻辑
        total_trials = len(manifest)
        if self.num_shards > 1:
            manifest = [e for i, e in enumerate(manifest) if (i % self.num_shards) == self.shard_index]

        print(f"\n{'=' * 60}")
        print("LLM EVALUATION (Ollama)")
        print(f"{'=' * 60}")
        print("Mode: zero-shot")
        print(f"Model: {self.model_tag}")
        print(f"Ollama Host: {self.ollama_host}")
        print(f"Seed: {self.seed}")
        print(f"Step limit: {self.step_limit}")
        print(f"Max vars per task: {self.max_vars_per_task or 'All'}")
        print(f"num_predict: {self.num_predict} | timeout: {self.request_timeout}s")
        if self.num_shards > 1:
            print(
                f"Sharding: shard {self.shard_index}/{self.num_shards}, total_trials={total_trials}, this_shard={len(manifest)}")
        print(f"Trials: {len(manifest)}")
        print("-" * 60)

        results = []
        for i, entry in enumerate(manifest):
            do_print = not self.quiet or ((i + 1) % 50 == 0) or (i == 0) or ((i + 1) == len(manifest))
            if do_print:
                print(f"[{i + 1}/{len(manifest)}] {entry['task_name']} var{entry['variation_id']}", end=' ')
            tr = self.run_trial(entry["task_name"], entry["variation_id"])
            results.append(tr)
            if do_print:
                print(f"R={tr['cumulative_reward']:.2f}")

        rewards = [r["cumulative_reward"] for r in results]
        mean_reward = np.mean(rewards) if rewards else 0.0
        se_reward = (np.std(rewards) / np.sqrt(len(rewards))) if len(rewards) > 0 else 0.0

        task_rewards = {}
        for r in results:
            task_rewards.setdefault(r["task_name"], []).append(r["cumulative_reward"])
        task_means = [np.mean(v) for v in task_rewards.values()] if task_rewards else []
        macro_mean = np.mean(task_means) if task_means else 0.0
        macro_se = (np.std(task_means) / np.sqrt(len(task_means))) if len(task_means) > 1 else 0.0

        cfg = f"ollama_{self.model_tag}_zero-shot_{self.seed}_{self.step_limit}_{self.max_vars_per_task}_{self.num_predict}_{self.request_timeout}_shard{self.shard_index}of{self.num_shards}"
        config_hash = hashlib.sha256(cfg.encode()).hexdigest()[:16]

        return {
            "config": {
                "backend": "ollama",
                "model": self.model_tag,
                "mode": "zero-shot",
                "seed": self.seed,
                "step_limit": self.step_limit,
                "max_vars_per_task": self.max_vars_per_task,
                "num_trials": len(manifest),
                "num_predict": self.num_predict,
                "request_timeout": self.request_timeout,
                "num_shards": self.num_shards,
                "shard_index": self.shard_index,
                "config_hash": config_hash,
            },
            "metrics": {
                "mean_reward": mean_reward,
                "mean_se": se_reward,
                "macro_mean": macro_mean,
                "macro_se": macro_se,
                "num_tasks": len(task_rewards),
                "success_rate": sum(1 for r in results if r["success"]) / len(results) if len(results) > 0 else 0.0,
            },
            "trials": results,
        }

    def save_results(self, output: Dict):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(self.output_dir, exist_ok=True)
        fn = f"{self.output_dir}/llm_zeroshot_{ts}.json"
        with open(fn, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        # --- 额外导出供 critical-step 选择器使用 ---
        sel_fn = f"{self.output_dir}/selector_corpus@{ts}.full.json"
        with open(sel_fn, "w", encoding="utf-8") as f:
            json.dump({"trials": output.get("trials", [])}, f, indent=2, ensure_ascii=False)
        print(f"Selector corpus saved to: {sel_fn}")
        # -----------------------------------------
        print(f"\nResults saved to: {fn}")
        print("\n" + "=" * 60)
        print("EVALUATION COMPLETE")
        print("=" * 60)
        print("Mode: zero-shot")
        print(f"Mean Reward: {output['metrics']['mean_reward']:.2f} ± {output['metrics']['mean_se']:.2f}")
        print(f"Macro Mean: {output['metrics']['macro_mean']:.2f} ± {output['metrics']['macro_se']:.2f}")
        print(f"Success Rate: {output['metrics']['success_rate'] * 100:.1f}%")
        print(f"Config Hash: {output['config']['config_hash']}")


def main():
    ap = argparse.ArgumentParser(description="ScienceWorld Zero-Shot Evaluation via Ollama (speed-optimized)")
    # Basic
    ap.add_argument("--jar-path", type=str, default=None, help="Path to ScienceWorld JAR")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--step-limit", type=int, default=30, help="Max steps per trial")
    ap.add_argument("--output-dir", type=str, default="eval_results", help="Output directory")
    ap.add_argument("--model", type=str, default="llama3.2:3b-instruct-q4_K_M", help="Ollama model tag")
    ap.add_argument("--ollama-host", type=str, default="http://localhost:11434", help="Ollama API host URL")
    ap.add_argument("--max-vars-per-task", type=int, default=None,
                    help="Maximum variations per task (for faster testing)")
    ap.add_argument("--num-predict", type=int, default=12, help="Max new tokens per step (single line).")
    ap.add_argument("--request-timeout", type=int, default=120, help="HTTP timeout (seconds) for Ollama requests.")

    # Speed knobs
    ap.add_argument("--num-shards", type=int, default=1, help="Total number of shards to split the manifest")
    ap.add_argument("--shard-index", type=int, default=0, help="Which shard this process runs (0-based)")
    ap.add_argument("--quiet", action="store_true", help="Less verbose logging (print every 50 trials)")
    ap.add_argument("--ollama-num-thread", type=int, default=None,
                    help="Execution threads for Ollama/llama.cpp (optional)")
    ap.add_argument("--ollama-num-batch", type=int, default=None, help="Batch size for Ollama/llama.cpp (optional)")
    ap.add_argument("--ollama-use-mmap", action="store_true", help="Enable mmap in Ollama (optional)")
    ap.add_argument("--ollama-f16-kv", action="store_true", help="Use fp16 KV cache in Ollama (optional)")
    ap.add_argument("--num-ctx", type=int, default=2048,
                    help="Ollama context window (VRAM-friendly default for 4GB GPUs).")

    # ===== 默认对齐 2120：不带参数也使用外部清单 + 兜底限制 =====
    default_manifest = r"C:\Users\ylili\Desktop\ASL\abschlussarbeit\sciworld\agenttraj-l-train-0.sciworld_only.json"
    ap.add_argument("--manifest-json", type=str, default=default_manifest,
                    help="Path to external manifest (json/jsonl or dict with 'trials'). "
                         "Defaults to agenttraj-l-train-0.sciworld_only.json")
    ap.add_argument("--limit-total", type=int, default=2120,
                    help="Hard cap on total trials (default 2120). Applied before sharding.")

    args = ap.parse_args()

    ev = LLMZeroShotEvaluator(
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
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        quiet=args.quiet,
        ollama_num_thread=args.ollama_num_thread,
        ollama_num_batch=args.ollama_num_batch,
        ollama_use_mmap=args.ollama_use_mmap if args.ollama_use_mmap else None,
        ollama_f16_kv=args.ollama_f16_kv if args.ollama_f16_kv else None,
        manifest_json=args.manifest_json,  # 默认即为 2120 清单路径
        limit_total=args.limit_total,  # 默认即为 2120
    )
    output = ev.run_evaluation()
    if "error" not in output:
        ev.save_results(output)


if __name__ == "__main__":
    main()