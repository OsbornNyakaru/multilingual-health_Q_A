"""Open-model LLM-as-judge for local validation.

Zindi's final 20% sub-metric uses an undisclosed judge model. We approximate
with Aya-Expanse-8B (open, multilingual, already loaded in memory during
inference) using a calibrated rubric. The judge MUST be an open model —
calling OpenAI/Anthropic/Google here is a disqualification risk.

Rubric (1-5 Likert, mapped to [0, 1] via (score - 1) / 4):
    5 — factually accurate, complete, language-appropriate, clinically safe.
    4 — minor gap (missing one sub-point) or mild register issue.
    3 — substantively correct but incomplete or awkward.
    2 — partly wrong or partly off-topic.
    1 — wrong, unsafe, or in the wrong language.

The judge is slow; we default to 20-row sampling for eval-during-training and
full-run only for the final validation pass.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_JUDGE_PROMPT_TMPL = (
    "You are a careful medical QA evaluator. Rate the Answer to the Question on "
    "a 1-5 scale for factual accuracy, completeness, language appropriateness, "
    "and clinical safety. Compare against the Reference.\n\n"
    "Language: {language}\n"
    "Question: {question}\n"
    "Reference: {reference}\n"
    "Answer: {answer}\n\n"
    "Respond with ONLY an integer from 1 to 5. No explanation."
)

_INT_RE = re.compile(r"\b([1-5])\b")


@dataclass
class JudgeResult:
    mean_score: float        # in [0, 1]
    n: int
    raw_scores: list[int]    # 1..5 integers


def score_llm_judge(
    questions: list[str],
    predictions: list[str],
    references: list[str],
    languages: list[str],
    judge_model_id: str = "CohereForAI/aya-expanse-8b",
    max_new_tokens: int = 4,
    sample_n: int | None = None,
    seed: int = 42,
) -> JudgeResult:
    """Run the open-model judge over all rows and return averaged score in [0, 1].

    Args:
        questions / predictions / references / languages: parallel lists.
        judge_model_id: HF id of the judge model. Must be an open model.
        max_new_tokens: tiny — we only need a single digit.
        sample_n: if set, randomly sample N rows for speed during training-time eval.
        seed: for sampling determinism.
    """
    if judge_model_id.startswith(("openai/", "anthropic/", "google/gemini")):
        raise ValueError(
            f"judge_model_id '{judge_model_id}' appears to be a closed API. "
            "Only open-source models are allowed per Zindi rules."
        )

    import random

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    n_total = len(predictions)
    if n_total == 0:
        return JudgeResult(0.0, 0, [])

    indices = list(range(n_total))
    if sample_n is not None and sample_n < n_total:
        rng = random.Random(seed)
        indices = rng.sample(indices, sample_n)

    tok = AutoTokenizer.from_pretrained(judge_model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        judge_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    raw_scores: list[int] = []
    with torch.no_grad():
        for i in indices:
            prompt = _JUDGE_PROMPT_TMPL.format(
                language=languages[i],
                question=questions[i],
                reference=references[i],
                answer=predictions[i],
            )
            inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(
                model.device
            )
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
            decoded = tok.decode(out[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
            raw_scores.append(_parse_int_score(decoded))

    mapped = [(s - 1) / 4.0 for s in raw_scores if s is not None]
    if not mapped:
        return JudgeResult(0.0, 0, [])
    return JudgeResult(mean_score=float(sum(mapped) / len(mapped)), n=len(mapped), raw_scores=raw_scores)


def _parse_int_score(decoded: str) -> int:
    """Extract the 1-5 integer; default to 3 if parsing fails (neutral)."""
    m = _INT_RE.search(decoded)
    if not m:
        return 3
    return int(m.group(1))
