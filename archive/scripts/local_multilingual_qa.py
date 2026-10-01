#!/usr/bin/env python3
"""
Interactive multilingual health QA in the same language as the question.

Use this when the full competition pipeline (CUDA + bitsandbytes + pinned torch)
is not available on your machine — for example CPU-only Windows or Python 3.13.

Install (minimal, works on CPU; add bitsandbytes on NVIDIA for 4-bit 8B models):

  python -m pip install torch transformers accelerate sentencepiece tqdm

Optional (NVIDIA GPU, recommended for AfriqueLlama / Aya 8B):

  python -m pip install bitsandbytes

Examples:

  python scripts/local_multilingual_qa.py --lang swa --question "Ni nini dalili za malaria?"
  python scripts/local_multilingual_qa.py --lang amh --question "የጤና ጥያቄ ምላሽ በአማርኛ ይስጡ።" --max-new-tokens 384

For bulk Val/Test generation at competition scale, use a CUDA environment
(GPU Windows, WSL2+CUDA, Kaggle, or Colab) and the repo Makefile / notebooks.
"""
from __future__ import annotations

import argparse
import sys


PROMPTS = {
    "swa": "Swali: {question}\n\nJibu:",
    "lug": "Ekibuuzo: {question}\n\nEky'okuddamu:",
    "aka": "Asɛmmisa: {question}\n\nMmuaeɛ:",
    "amh": "ጥያቄ: {question}\n\nመልስ:",
    "eng": "Question: {question}\n\nAnswer:",
}
ANSWER_MARKERS = {
    "swa": "Jibu:",
    "lug": "Eky'okuddamu:",
    "aka": "Mmuaeɛ:",
    "amh": "መልስ:",
    "eng": "Answer:",
}


def _strip_prompt_artefacts(text: str, lang: str) -> str:
    marker = ANSWER_MARKERS.get(lang, "")
    if marker and marker in text:
        text = text.split(marker, 1)[-1]
    text = text.strip()
    if "\n\n" in text:
        first = text.split("\n\n", 1)[0].strip()
        if len(first) <= 800:
            text = first
    return text


def _build_prompt(question: str, lang: str, tokenizer) -> str:
    content = PROMPTS[lang].format(question=question)
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": content}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return content


def _bnb_available() -> bool:
    try:
        import bitsandbytes  # noqa: F401

        return True
    except Exception:
        return False


def load_model_auto(preferred: str | None, force_cpu: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    has_cuda = torch.cuda.is_available() and not force_cpu
    use_4bit = has_cuda and _bnb_available()

    model_chain: list[str] = []
    if preferred:
        model_chain.append(preferred)
    if has_cuda and use_4bit:
        model_chain.extend(
            [
                "McGill-NLP/AfriqueLlama-8B",
                "CohereForAI/aya-expanse-8b",
                "Qwen/Qwen2.5-7B-Instruct",
            ]
        )
    elif has_cuda:
        model_chain.extend(
            [
                "google/gemma-2-2b-it",
                "Qwen/Qwen2.5-1.5B-Instruct",
            ]
        )
    else:
        model_chain.extend(
            [
                "Qwen/Qwen2.5-1.5B-Instruct",
                "google/gemma-2-2b-it",
            ]
        )

    seen: set[str] = set()
    model_chain = [m for m in model_chain if not (m in seen or seen.add(m))]

    errors: list[str] = []
    dtype_cpu = torch.float32

    for model_id in model_chain:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            if use_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                )
            elif has_cuda:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True,
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=dtype_cpu,
                    device_map="cpu",
                    trust_remote_code=True,
                )
            model.eval()
            return model_id, tokenizer, model, "cuda" if has_cuda else "cpu"
        except Exception as exc:
            errors.append(f"{model_id}: {exc}")

    raise RuntimeError("Could not load any model in the chain:\n" + "\n".join(errors))


def generate_answer(
    tokenizer,
    model,
    device_kind: str,
    question: str,
    lang: str,
    max_new_tokens: int,
    min_new_tokens: int,
) -> str:
    import torch

    prompt = _build_prompt(question, lang, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt")
    dev = next(model.parameters()).device
    inputs = {k: v.to(dev) for k, v in inputs.items()}

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
        )
    new_tokens = out_ids[0][inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return _strip_prompt_artefacts(text, lang)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lang", choices=sorted(PROMPTS.keys()), required=True, help="ISO-style code used in this repo")
    parser.add_argument("--question", required=True, help="Health question in that language")
    parser.add_argument("--model", default=None, help="Override Hugging Face model id (skip auto chain)")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--min-new-tokens", type=int, default=32)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    args = parser.parse_args()

    try:
        model_id, tokenizer, model, device_kind = load_model_auto(args.model, force_cpu=args.cpu)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    print(f"Loaded: {model_id} ({device_kind})")
    answer = generate_answer(
        tokenizer,
        model,
        device_kind,
        args.question,
        args.lang,
        args.max_new_tokens,
        args.min_new_tokens,
    )
    print()
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
