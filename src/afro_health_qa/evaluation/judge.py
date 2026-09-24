"""Open-weight LLM-as-judge STAND-IN for the 0.26 leaderboard component.

Zindi's judge model and prompt are undisclosed. This module is a local
approximation, not the grader: use it to rank candidates, never quote it as a
leaderboard number. Calling closed APIs (OpenAI/Anthropic/Gemini) is forbidden
by the rules and refused here.

Design — the backend is injectable so everything runs on CPU without a model:

    judge_fn: Callable[[list[str]], list[str]]   # prompts -> raw completions

* tests / dry runs pass a fake callable;
* ``TransformersJudge`` and ``VLLMJudge`` load a HF model lazily on first call
  (run them on molab, not on a laptop);
* ``judge_fn=None`` means "no judge": the scorer reports the judge as NaN.

Rubric (1-5, mapped to [0, 1] via ``(score - 1) / 4``) — see ``JUDGE_RUBRIC``.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

# Suggested molab judge: strong multilingual (incl. Amharic / Swahili), open
# weights, fits a 96 GB RTX Pro 6000 in bf16, and is NOT one of our generator
# candidates (avoids self-preference bias). See vault/facts/metric-replica.md.
DEFAULT_JUDGE_MODEL_ID = "google/gemma-3-27b-it"

JudgeFn = Callable[[list[str]], list[str]]

JUDGE_RUBRIC = """\
Score each criterion from 1 (worst) to 5 (best):
- accuracy: is the medical content factually correct and consistent with the reference answer? \
Unsafe or contradictory advice scores 1.
- completeness: does it cover the key points of the reference answer for this question?
- language: is it written in the same language as the question ({language}), fluent and in an \
appropriate register? An answer in the wrong language scores 1.
- overall: your holistic 1-5 score for the answer."""

JUDGE_PROMPT_TEMPLATE = """\
You are a strict evaluator of answers to maternal, sexual and reproductive health questions \
asked in African languages.

{rubric}

Language: {language}
Question: {question}
Reference answer: {reference}
Candidate answer: {answer}

Respond ONLY with a JSON object and no other text:
{{"accuracy": <1-5>, "completeness": <1-5>, "language": <1-5>, "overall": <1-5>}}"""

SUBSET_LANGUAGE = {
    "Aka": "Akan (Twi)",
    "Amh": "Amharic",
    "Eng": "English",
    "Lug": "Luganda",
    "Swa": "Swahili",
}

_CLOSED_PREFIXES = ("openai/", "anthropic/", "google/gemini", "gpt-", "claude")
_CRITERIA = ("accuracy", "completeness", "language")
_JSON_RE = re.compile(r"\{.*?\}", flags=re.DOTALL)
_LABELLED_RE = re.compile(r"(?:overall|score|rating)\W{0,5}([1-5](?:\.\d+)?)(?![\d.])", flags=re.IGNORECASE)
_OUT_OF_RE = re.compile(r"\b([1-5](?:\.\d+)?)\s*/\s*5\b")
_BARE_RE = re.compile(r"(?<![\d.])([1-5])(?![\d.])")


def language_name(subset: str) -> str:
    """``Amh_Eth`` -> ``Amharic``; unknown codes pass through unchanged."""
    return SUBSET_LANGUAGE.get(str(subset).split("_")[0], str(subset))


def build_judge_prompt(question: str, reference: str, answer: str, language: str) -> str:
    return JUDGE_PROMPT_TEMPLATE.format(
        rubric=JUDGE_RUBRIC.format(language=language),
        language=language,
        question=question,
        reference=reference,
        answer=answer,
    )


def _valid(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if 1.0 <= v <= 5.0 else None


def parse_judge_score(text: str | None) -> float | None:
    """Extract a 1-5 score from a judge completion; ``None`` if unparseable.

    Order: JSON ``overall`` -> mean of the JSON criteria -> "Score: 4" /
    "overall 4" -> "4/5" -> a single bare digit 1-5. Out-of-range values are
    rejected rather than clipped. Failures are NaN downstream, never a silent 3.
    """
    if not text:
        return None
    for blob in _JSON_RE.findall(text):
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        overall = _valid(obj.get("overall"))
        if overall is not None:
            return overall
        crit = [_valid(obj.get(k)) for k in _CRITERIA]
        if all(c is not None for c in crit):
            return sum(crit) / len(crit)
    for pattern in (_LABELLED_RE, _OUT_OF_RE):
        m = pattern.search(text)
        if m:
            return _valid(m.group(1))
    bare = _BARE_RE.findall(text)
    if len(set(bare)) == 1:  # ambiguous if several different digits appear
        return float(bare[0])
    return None


def normalize_judge_score(score: float | None) -> float:
    """Map 1-5 to [0, 1] via (score - 1) / 4; ``None`` -> NaN."""
    if score is None or (isinstance(score, float) and math.isnan(score)):
        return float("nan")
    return (float(score) - 1.0) / 4.0


@dataclass
class JudgeResult:
    per_row: list[float]  # normalized [0, 1], NaN for unjudged / unparseable rows
    raw: list[str | None] = field(default_factory=list)
    parse_failures: int = 0
    model_id: str | None = None

    @property
    def mean(self) -> float:
        vals = [v for v in self.per_row if not math.isnan(v)]
        return float(sum(vals) / len(vals)) if vals else float("nan")


def run_judge(
    judge_fn: JudgeFn,
    questions: Sequence[str],
    references: Sequence[str],
    answers: Sequence[str],
    subsets: Sequence[str],
    batch_size: int = 32,
    model_id: str | None = None,
) -> JudgeResult:
    """Build prompts, call ``judge_fn`` in batches, parse + normalize each completion."""
    if not (len(questions) == len(references) == len(answers) == len(subsets)):
        raise ValueError("questions, references, answers, subsets must have the same length.")
    prompts = [
        build_judge_prompt(q, r, a, language_name(s))
        for q, r, a, s in zip(questions, references, answers, subsets)
    ]
    raw: list[str | None] = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        out = list(judge_fn(batch))
        if len(out) != len(batch):
            raise RuntimeError(f"judge_fn returned {len(out)} completions for {len(batch)} prompts")
        raw.extend(out)
    parsed = [parse_judge_score(t) for t in raw]
    return JudgeResult(
        per_row=[normalize_judge_score(p) for p in parsed],
        raw=raw,
        parse_failures=sum(p is None for p in parsed),
        model_id=model_id or getattr(judge_fn, "model_id", None),
    )


def _refuse_closed(model_id: str) -> None:
    if model_id.lower().startswith(_CLOSED_PREFIXES):
        raise ValueError(f"judge model '{model_id}' looks like a closed API; only open weights are allowed.")


class TransformersJudge:
    """HF transformers backend. Loads lazily on first call (GPU expected; run on molab)."""

    def __init__(self, model_id: str = DEFAULT_JUDGE_MODEL_ID, max_new_tokens: int = 64, max_input_tokens: int = 3072):
        _refuse_closed(model_id)
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.max_input_tokens = max_input_tokens
        self._tok = None
        self._model = None

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(self.model_id)
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token
        self._tok.padding_side = "left"
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=torch.bfloat16, device_map="auto"
        ).eval()

    def _format(self, prompt: str) -> str:
        if getattr(self._tok, "chat_template", None):
            return self._tok.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
            )
        return prompt

    def __call__(self, prompts: list[str]) -> list[str]:
        import torch

        if self._model is None:
            self._load()
        enc = self._tok(
            [self._format(p) for p in prompts],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_input_tokens,
            add_special_tokens=False,
        ).to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                **enc, max_new_tokens=self.max_new_tokens, do_sample=False, pad_token_id=self._tok.pad_token_id
            )
        return self._tok.batch_decode(out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)


class VLLMJudge:
    """vLLM backend (much faster for full held-out runs). Loads lazily on first call."""

    def __init__(self, model_id: str = DEFAULT_JUDGE_MODEL_ID, max_new_tokens: int = 64, **llm_kwargs):
        _refuse_closed(model_id)
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.llm_kwargs = llm_kwargs
        self._llm = None

    def __call__(self, prompts: list[str]) -> list[str]:
        from vllm import LLM, SamplingParams

        if self._llm is None:
            self._llm = LLM(model=self.model_id, **self.llm_kwargs)
        params = SamplingParams(temperature=0.0, max_tokens=self.max_new_tokens)
        outs = self._llm.chat([[{"role": "user", "content": p}] for p in prompts], params, use_tqdm=False)
        return [o.outputs[0].text for o in outs]


def make_judge(backend: str | None, model_id: str = DEFAULT_JUDGE_MODEL_ID) -> JudgeFn | None:
    """``None``/``"none"`` -> no judge; ``"transformers"`` or ``"vllm"`` -> lazy backend."""
    if backend in (None, "none"):
        return None
    if backend == "transformers":
        return TransformersJudge(model_id)
    if backend == "vllm":
        return VLLMJudge(model_id)
    raise ValueError(f"unknown judge backend {backend!r}")
