"""Batched generation with configurable decoding.

The core function returns either:
    - a list of single-string outputs (strategy='greedy' | 'beam'), or
    - a list of list-of-string outputs (strategy='beam_rerank'), one per question,
      each containing num_return_sequences candidates.

Per-language min/max length are drawn at runtime from the training-answer
distribution (see configs/decoding/*.yaml ``length_policy``).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class LengthPolicy:
    """Per-language min/max new-token counts derived from training answers."""

    per_language_min: dict[str, int]
    per_language_max: dict[str, int]
    fallback_min: int = 16
    fallback_max: int = 256

    def get(self, language: str) -> tuple[int, int]:
        return (
            self.per_language_min.get(language, self.fallback_min),
            self.per_language_max.get(language, self.fallback_max),
        )


def length_policy_from_training_df(
    df: pd.DataFrame,
    tokenizer,
    low: float = 0.25,
    high: float = 0.95,
    language_col: str = "Language",
    response_col: str = "Response",
) -> LengthPolicy:
    """Build a LengthPolicy from training answers.

    Tokenises each training answer and uses the ``low`` and ``high`` quantiles
    per language. Outputs token counts, not character counts, because
    ``max_new_tokens`` is a token budget.
    """
    per_min: dict[str, int] = {}
    per_max: dict[str, int] = {}
    for lang, group in df.groupby(language_col, sort=True):
        lengths = [
            len(tokenizer(t, add_special_tokens=False)["input_ids"])
            for t in group[response_col].astype(str).tolist()
            if t
        ]
        if not lengths:
            continue
        lengths.sort()
        q_low = int(lengths[int(len(lengths) * low)])
        q_high = int(lengths[min(len(lengths) - 1, int(len(lengths) * high))])
        per_min[lang] = max(4, q_low)
        per_max[lang] = max(q_high, q_low + 4)
    return LengthPolicy(per_language_min=per_min, per_language_max=per_max)


def generate_batch(
    model,
    tokenizer,
    questions: list[str],
    languages: list[str],
    decoding_cfg: dict,
    length_policy: LengthPolicy,
    batch_size: int = 4,
) -> list[str] | list[list[str]]:
    """Run batched generation for a list of questions.

    Args:
        model: HF causal LM (base or adapter-loaded).
        tokenizer: associated tokenizer.
        questions / languages: parallel lists.
        decoding_cfg: one of the configs/decoding/*.yaml dicts.
        length_policy: LengthPolicy instance — per-language min/max.
        batch_size: generation batch size.

    Returns:
        list of strings (if num_return_sequences <= 1), else list of lists.
    """
    import torch

    from afro_health_qa.inference.postprocess import clean_output
    from afro_health_qa.models.prompts import build_prompt, parse_response

    strategy = decoding_cfg.get("strategy", "greedy")
    num_return = int(decoding_cfg.get("num_return_sequences", 1))

    model.eval()
    outputs: list[str] | list[list[str]] = []

    with torch.no_grad():
        for start in range(0, len(questions), batch_size):
            batch_q = questions[start : start + batch_size]
            batch_lang = languages[start : start + batch_size]
            prompts = [build_prompt(q, lang) for q, lang in zip(batch_q, batch_lang)]
            enc = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            ).to(model.device)

            # Use the MAX of per-language max in this batch — we'll trim per-row later.
            batch_max = max(length_policy.get(lang)[1] for lang in batch_lang)
            batch_min = min(length_policy.get(lang)[0] for lang in batch_lang)

            gen_kwargs = dict(
                **enc,
                max_new_tokens=batch_max,
                min_new_tokens=max(batch_min, 1),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=decoding_cfg.get("repetition_penalty", 1.1),
                no_repeat_ngram_size=decoding_cfg.get("no_repeat_ngram_size", 3),
            )

            if strategy == "greedy":
                gen_kwargs.update(dict(do_sample=False, num_beams=1))
            elif strategy == "beam":
                gen_kwargs.update(
                    dict(
                        do_sample=False,
                        num_beams=int(decoding_cfg.get("num_beams", 5)),
                        length_penalty=float(decoding_cfg.get("length_penalty", 1.0)),
                        early_stopping=bool(decoding_cfg.get("early_stopping", True)),
                    )
                )
            elif strategy == "beam_rerank":
                gen_kwargs.update(
                    dict(
                        do_sample=False,
                        num_beams=int(decoding_cfg.get("num_beams", 8)),
                        num_beam_groups=int(decoding_cfg.get("num_beam_groups", 4)),
                        diversity_penalty=float(decoding_cfg.get("diversity_penalty", 0.5)),
                        num_return_sequences=num_return,
                        length_penalty=float(decoding_cfg.get("length_penalty", 1.0)),
                        early_stopping=bool(decoding_cfg.get("early_stopping", True)),
                    )
                )
            else:
                raise ValueError(f"unknown decoding strategy: {strategy}")

            gen_ids = model.generate(**gen_kwargs)
            input_len = enc["input_ids"].shape[1]
            new_tokens = gen_ids[:, input_len:]

            decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

            # Regroup if we got multiple sequences per input.
            for i, lang in enumerate(batch_lang):
                if num_return == 1:
                    cleaned = clean_output(parse_response(decoded[i], lang), lang)
                    outputs.append(cleaned)
                else:
                    group = decoded[i * num_return : (i + 1) * num_return]
                    cleaned = [clean_output(parse_response(d, lang), lang) for d in group]
                    outputs.append(cleaned)

    return outputs
