from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from rouge_score import rouge_scorer
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

_ROUGE_SCORER = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)
_AFROLM_BUNDLE = None
_JUDGE_BUNDLE = None


class UnicodeWhitespaceTokenizer:
    def tokenize(self, text):
        return re.findall(r"\S+", str(text), flags=re.UNICODE)


@dataclass
class EncoderBundle:
    model_id: str
    tokenizer: object
    model: object
    device: str
    warning: str | None = None


@dataclass
class JudgeBundle:
    model_id: str
    tokenizer: object
    model: object
    device: str
    warning: str | None = None


def compute_rouge(predictions: list[str], references: list[str], langs: list[str] | None = None) -> dict:
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rougeL"],
        use_stemmer=False,
        tokenizer=UnicodeWhitespaceTokenizer(),
    )
    r1_scores: list[float] = []
    rl_scores: list[float] = []
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        r1_scores.append(scores["rouge1"].fmeasure)
        rl_scores.append(scores["rougeL"].fmeasure)

    report = {
        "rouge1": sum(r1_scores) / len(r1_scores),
        "rougeL": sum(rl_scores) / len(rl_scores),
        "rouge1_per_row": r1_scores,
        "rougeL_per_row": rl_scores,
    }
    if langs is not None:
        per_language: dict[str, dict[str, float]] = {}
        for lang in sorted(set(langs)):
            mask = [i for i, value in enumerate(langs) if value == lang]
            per_language[lang] = {
                "rouge1": sum(r1_scores[i] for i in mask) / len(mask),
                "rougeL": sum(rl_scores[i] for i in mask) / len(mask),
            }
        report["per_language"] = per_language
    return report


def mean_pool(token_embeddings, attention_mask):
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * mask_expanded, 1)
    denom = torch.clamp(mask_expanded.sum(1), min=1e-9)
    return summed / denom


def load_afrolm_bundle() -> EncoderBundle:
    global _AFROLM_BUNDLE
    if _AFROLM_BUNDLE is not None:
        return _AFROLM_BUNDLE

    device = "cuda" if torch.cuda.is_available() else "cpu"
    warnings: list[str] = []
    for model_id in ("bonadossou/afrolm_active_learning", "intfloat/multilingual-e5-base"):
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModel.from_pretrained(model_id).to(device).eval()
            warning = None
            if model_id != "bonadossou/afrolm_active_learning":
                warning = (
                    "PROXY METRIC ACTIVE: using multilingual-e5-base instead of AfroLM.\n"
                    "  Local scores will diverge from Zindi's AfroLM BertScore sub-metric.\n"
                    "  Switch to AfroLM before making model selection decisions."
                )
            _AFROLM_BUNDLE = EncoderBundle(
                model_id=model_id,
                tokenizer=tokenizer,
                model=model,
                device=device,
                warning=warning,
            )
            return _AFROLM_BUNDLE
        except Exception as exc:
            warnings.append(f"{model_id}: {exc}")

    raise RuntimeError("failed to load AfroLM or fallback encoder: " + " | ".join(warnings))


def encode_texts(texts: list[str], batch_size: int = 32, bundle: EncoderBundle | None = None) -> torch.Tensor:
    bundle = bundle or load_afrolm_bundle()
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = bundle.tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        enc = {k: v.to(bundle.device) for k, v in enc.items()}
        with torch.no_grad():
            out = bundle.model(**enc)
        embeddings = mean_pool(out.last_hidden_state, enc["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        all_embeddings.append(embeddings.cpu())
    return torch.cat(all_embeddings, dim=0)


def afrolm_bertscore(
    predictions: list[str],
    references: list[str],
    batch_size: int = 32,
    bundle: EncoderBundle | None = None,
) -> dict:
    del batch_size
    try:
        from afro_health_qa.evaluation.afrolm_bertscore import (
            AFROLM_MODEL_ID,
            score_afrolm_bertscore,
            score_afrolm_bertscore_per_example,
        )

        result = score_afrolm_bertscore(predictions, references, use_cache=True)
        per_row = score_afrolm_bertscore_per_example(predictions, references, use_cache=True)
        return {
            "afrolm_bs": result.f1,
            "afrolm_bs_per_row": per_row,
            "model_id": AFROLM_MODEL_ID,
            "warning": None,
        }
    except Exception:
        bundle = bundle or load_afrolm_bundle()
        pred_emb = encode_texts(predictions, batch_size=32, bundle=bundle)
        ref_emb = encode_texts(references, batch_size=32, bundle=bundle)
        cosine_scores = (pred_emb * ref_emb).sum(dim=1)
        scores = cosine_scores.tolist()
        return {
            "afrolm_bs": sum(scores) / len(scores),
            "afrolm_bs_per_row": scores,
            "model_id": bundle.model_id,
            "warning": bundle.warning,
        }


JUDGE_PROMPT = """You are a medical QA evaluator for African-language health information.
Rate the candidate answer against the reference on three criteria:
1. Factual accuracy (is the medical content correct?)
2. Completeness (does it fully address the question?)
3. Language appropriateness (is it in the correct language and register?)

Language: {lang}
Question: {question}
Reference answer: {reference}
Candidate answer: {candidate}

Respond ONLY with a valid JSON object, no other text:
{{"accuracy": <1-5>, "completeness": <1-5>, "language": <1-5>}}"""


def load_judge_bundle() -> JudgeBundle | None:
    global _JUDGE_BUNDLE
    if _JUDGE_BUNDLE is not None:
        return _JUDGE_BUNDLE

    if not torch.cuda.is_available():
        return None

    warnings: list[str] = []
    model_id = "microsoft/Phi-3.5-mini-instruct"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
        ).eval()
        _JUDGE_BUNDLE = JudgeBundle(
            model_id=model_id,
            tokenizer=tokenizer,
            model=model,
            device="cuda",
        )
        return _JUDGE_BUNDLE
    except Exception as exc:
        warnings.append(str(exc))
    return None


def _generate_from_judge_model(prompt: str, bundle: JudgeBundle) -> str:
    enc = bundle.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    enc = {k: v.to(bundle.device) for k, v in enc.items()}
    with torch.no_grad():
        out = bundle.model.generate(
            **enc,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=bundle.tokenizer.eos_token_id,
        )
    new_tokens = out[0][enc["input_ids"].shape[1] :]
    return bundle.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def llm_judge(
    predictions: list[str],
    references: list[str],
    questions: list[str],
    langs: list[str],
    batch_size: int = 4,
) -> dict:
    del batch_size
    bundle = load_judge_bundle()
    if bundle is None:
        return {
            "judge": None,
            "judge_per_row": [None] * len(predictions),
            "model_id": None,
            "warning": (
                "JUDGE DISABLED: microsoft/Phi-3.5-mini-instruct is unavailable on this machine.\n"
                "  Falling back to the 3-metric proxy formula: R1*0.3125 + RL*0.3125 + AfroLM*0.375."
            ),
        }

    scores: list[float] = []
    parse_failures = 0
    for pred, ref, question, lang in zip(predictions, references, questions, langs):
        prompt = JUDGE_PROMPT.format(lang=lang, question=question, reference=ref, candidate=pred)
        raw = _generate_from_judge_model(prompt, bundle)
        try:
            start = raw.find("{")
            end = raw.rfind("}")
            parsed = json.loads(raw[start : end + 1] if start != -1 and end != -1 else raw.strip())
            score = (parsed["accuracy"] + parsed["completeness"] + parsed["language"]) / 15.0
        except Exception:
            score = 0.5
            parse_failures += 1
        scores.append(score)

    warning = None
    if parse_failures:
        warning = f"Judge JSON parse failures: {parse_failures} rows fell back to neutral 0.5."
    return {
        "judge": sum(scores) / len(scores),
        "judge_per_row": scores,
        "model_id": bundle.model_id,
        "warning": warning,
    }


def combined_score(r1: float, rl: float, afrolm_bs: float, judge: float) -> float:
    return 0.25 * r1 + 0.25 * rl + 0.30 * afrolm_bs + 0.20 * judge


def combined_score_proxy(r1: float, rl: float, afrolm_bs: float) -> float:
    return 0.3125 * r1 + 0.3125 * rl + 0.375 * afrolm_bs


def full_eval_report(
    predictions: list[str],
    val_df,
    *,
    answer_col: str,
    question_col: str,
    lang_col: str,
    lang_code_map: dict[str, str],
) -> dict:
    references = val_df[answer_col].astype(str).tolist()
    questions = val_df[question_col].astype(str).tolist()
    langs = [lang_code_map[str(l)] for l in val_df[lang_col].astype(str).tolist()]

    rouge = compute_rouge(predictions, references, langs=langs)
    afrolm = afrolm_bertscore(predictions, references)
    judge = llm_judge(predictions, references, questions, langs)

    warnings = [w for w in (afrolm.get("warning"), judge.get("warning")) if w]
    if judge["judge"] is None:
        combined = combined_score_proxy(rouge["rouge1"], rouge["rougeL"], afrolm["afrolm_bs"])
        scoring_mode = "proxy_3_metric"
    else:
        combined = combined_score(rouge["rouge1"], rouge["rougeL"], afrolm["afrolm_bs"], judge["judge"])
        scoring_mode = "full_4_metric"

    per_lang: dict[str, dict[str, float | None]] = {}
    combined_per_row: list[float] = []
    for i, lang in enumerate(langs):
        judge_row = judge["judge_per_row"][i]
        if judge_row is None:
            combined_per_row.append(
                combined_score_proxy(
                    rouge["rouge1_per_row"][i],
                    rouge["rougeL_per_row"][i],
                    afrolm["afrolm_bs_per_row"][i],
                )
            )
        else:
            combined_per_row.append(
                combined_score(
                    rouge["rouge1_per_row"][i],
                    rouge["rougeL_per_row"][i],
                    afrolm["afrolm_bs_per_row"][i],
                    judge_row,
                )
            )

    for lang in sorted(set(langs)):
        mask = [i for i, value in enumerate(langs) if value == lang]
        r1_lang = sum(rouge["rouge1_per_row"][i] for i in mask) / len(mask)
        rl_lang = sum(rouge["rougeL_per_row"][i] for i in mask) / len(mask)
        bs_lang = sum(afrolm["afrolm_bs_per_row"][i] for i in mask) / len(mask)
        judge_values = [judge["judge_per_row"][i] for i in mask if judge["judge_per_row"][i] is not None]
        judge_lang = sum(judge_values) / len(judge_values) if judge_values else None
        combined_lang = (
            combined_score_proxy(r1_lang, rl_lang, bs_lang)
            if judge_lang is None
            else combined_score(r1_lang, rl_lang, bs_lang, judge_lang)
        )
        per_lang[lang] = {
            "combined": combined_lang,
            "rouge1": r1_lang,
            "rougeL": rl_lang,
            "afrolm_bs": bs_lang,
            "judge": judge_lang,
        }

    return {
        "combined": combined,
        "r1": rouge["rouge1"],
        "rl": rouge["rougeL"],
        "afrolm_bs": afrolm["afrolm_bs"],
        "judge": judge["judge"],
        "per_language": per_lang,
        "warnings": warnings,
        "scoring_mode": scoring_mode,
        "rouge1_per_row": rouge["rouge1_per_row"],
        "rougeL_per_row": rouge["rougeL_per_row"],
        "afrolm_bs_per_row": afrolm["afrolm_bs_per_row"],
        "judge_per_row": judge["judge_per_row"],
        "combined_per_row": combined_per_row,
    }


def calibration_checks(
    val_df,
    *,
    answer_col: str,
    question_col: str,
    lang_col: str,
    lang_code_map: dict[str, str],
    seed: int = 42,
) -> dict:
    sample = val_df.sample(n=min(10, len(val_df)), random_state=seed).reset_index(drop=True)
    perfect_predictions = sample[answer_col].astype(str).tolist()
    perfect_report = full_eval_report(
        perfect_predictions,
        sample,
        answer_col=answer_col,
        question_col=question_col,
        lang_col=lang_col,
        lang_code_map=lang_code_map,
    )

    random.seed(seed)
    shuffled_predictions: list[str] = []
    lang_values = [lang_code_map[str(v)] for v in sample[lang_col].astype(str).tolist()]
    full_langs = [lang_code_map[str(v)] for v in val_df[lang_col].astype(str).tolist()]
    full_refs = val_df[answer_col].astype(str).tolist()
    for lang in lang_values:
        candidates = [ref for ref, candidate_lang in zip(full_refs, full_langs) if candidate_lang != lang]
        if not candidates:
            raise RuntimeError("could not build wrong-language calibration set.")
        shuffled_predictions.append(random.choice(candidates))
    shuffled_report = full_eval_report(
        shuffled_predictions,
        sample,
        answer_col=answer_col,
        question_col=question_col,
        lang_col=lang_col,
        lang_code_map=lang_code_map,
    )

    return {
        "check_a": perfect_report["combined"],
        "check_b": shuffled_report["combined"],
        "check_a_pass": perfect_report["combined"] >= 0.95,
        "check_b_pass": shuffled_report["combined"] <= 0.20,
        "reports": {"perfect": perfect_report, "wrong_language": shuffled_report},
    }
