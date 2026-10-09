"""vllm_worker.py — generation with vLLM in its own process and Python environment.

experiment.py (gen_engine="vllm") starts this with the vLLM venv's interpreter, so vLLM's pinned
torch never meets the runner kernel's torch. It reads a job JSON:

    {"model": id, "adapter": local dir | null, "lora_rank": int, "max_model_len": int, "mem": float,
     "temps": [..], "n": int, "top_p": float, "seed": int, "chunk": int, "max_num_seqs": int (0 = default),
     "requests": [{"id", "ids": [prompt token ids], "max_tokens", "min_tokens"}, ...],
     "out": path}

and appends one JSON line per request to `out`, a chunk at a time, so the parent can checkpoint:

    {"id", "greedy": text, "samples": [n texts per temperature, in temps order]}

Requests whose id is already in `out` are skipped (resume).
"""

import json
import sys
import time


def main(job_path: str) -> None:
    job = json.load(open(job_path, encoding="utf-8"))
    done = set()
    try:
        with open(job["out"], encoding="utf-8") as f:
            done = {json.loads(line)["id"] for line in f if line.strip()}
    except FileNotFoundError:
        pass
    reqs = [r for r in job["requests"] if r["id"] not in done]
    print(f"[vllm] {len(reqs)} requests to generate ({len(done)} already done)", flush=True)
    if not reqs:
        return

    from vllm import LLM, SamplingParams

    kw = dict(model=job["model"], dtype="bfloat16", max_model_len=int(job["max_model_len"]),
              gpu_memory_utilization=float(job["mem"]), seed=int(job["seed"]), trust_remote_code=True,
              limit_mm_per_prompt={"image": 0, "audio": 0, "video": 0})  # text only: skip vision/audio memory
    if int(job.get("max_num_seqs") or 0) > 0:
        kw["max_num_seqs"] = int(job["max_num_seqs"])
    lora = None
    if job.get("adapter"):
        from vllm.lora.request import LoRARequest

        kw.update(enable_lora=True, max_lora_rank=int(job["lora_rank"]), max_loras=1)
        lora = LoRARequest("adapter", 1, job["adapter"])
    try:
        llm = LLM(**kw)
    except (TypeError, ValueError):  # text-only models reject limit_mm_per_prompt
        kw.pop("limit_mm_per_prompt")
        llm = LLM(**kw)

    n, temps = int(job["n"]), [float(t) for t in job["temps"]]
    chunk = int(job["chunk"])
    t0 = time.time()
    for s in range(0, len(reqs), chunk):
        part = reqs[s : s + chunk]
        prompts, params = [], []
        for r in part:
            p = {"prompt_token_ids": r["ids"]}
            mx, mn = int(r["max_tokens"]), int(r["min_tokens"])
            prompts.append(p)
            params.append(SamplingParams(temperature=0.0, max_tokens=mx, min_tokens=mn))
            for ti, t in enumerate(temps if n > 0 else []):
                prompts.append(p)
                params.append(SamplingParams(n=n, temperature=t, top_p=float(job["top_p"]), max_tokens=mx, min_tokens=mn,
                                             seed=int(job["seed"]) + ti))
        outs = llm.generate(prompts, params, lora_request=lora, use_tqdm=False)
        per = 1 + (len(temps) if n > 0 else 0)
        with open(job["out"], "a", encoding="utf-8") as f:
            for k, r in enumerate(part):
                o = outs[k * per : (k + 1) * per]
                rec = {"id": r["id"], "greedy": o[0].outputs[0].text,
                       "samples": [c.text for x in o[1:] for c in x.outputs]}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[vllm] {s + len(part)}/{len(reqs)} done, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
