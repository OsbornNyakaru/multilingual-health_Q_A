from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from afro_health_qa.data.runtime import (
    ColumnMapping,
    build_submission_schema,
    canonical_prompt_language,
    contains_geez,
    detect_column_mapping,
    infer_lang_code_map,
    read_csv,
    set_global_seed,
)
from afro_health_qa.submission.format import SubmissionMetadata, build_submission_df, write_submission_csv
from afro_health_qa.submission.validate import SubmissionValidationError, validate_submission_csv

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
SUBMISSIONS_DIR = Path("submissions")
TARGET_LEADERBOARD = 0.768095
EXPERIMENT_ID = "exp001"
HYPOTHESIS = (
    "Zero-shot Aya-Expanse-8B lang-native prompts length-matched beam5 beats 0.768095"
)
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
ALLOW_HELD_OUT = False


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _ensure_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)


def _print_step(title: str) -> None:
    print()
    print(title)


def _language_distribution(df: pd.DataFrame, lang_col: str) -> pd.DataFrame:
    counts = df[lang_col].astype(str).value_counts(dropna=False).sort_index()
    pct = counts / len(df) * 100.0
    return pd.DataFrame({"count": counts, "pct": pct.round(2)})


def _length_stats(df: pd.DataFrame, lang_col: str, answer_col: str, lang_code_map: dict[str, str]) -> pd.DataFrame:
    rows = []
    for raw_lang, group in df.groupby(lang_col, sort=True):
        canonical = lang_code_map[str(raw_lang)]
        token_counts = group[answer_col].astype(str).map(lambda x: len(x.split()))
        rows.append(
            {
                "raw_lang": raw_lang,
                "lang": canonical,
                "n": int(len(group)),
                "mean": round(float(token_counts.mean()), 2),
                "median": round(float(token_counts.median()), 2),
                "p10": int(np.percentile(token_counts, 10)),
                "p90": int(np.percentile(token_counts, 90)),
            }
        )
    return pd.DataFrame(rows).sort_values(["lang", "raw_lang"]).reset_index(drop=True)


def _normalise_for_runtime(df: pd.DataFrame, mapping: ColumnMapping) -> pd.DataFrame:
    return df.rename(
        columns={
            mapping.id_col: "ID",
            mapping.question_col: "Question",
            mapping.lang_col: "Language",
            mapping.answer_col: "Response",
        }
    )


def _df_block(df: pd.DataFrame, *, index: bool = True) -> str:
    return "```\n" + df.to_string(index=index) + "\n```"


def _strip_prompt_artefacts(text: str, lang: str) -> str:
    marker = ANSWER_MARKERS.get(lang, "")
    if marker and marker in text:
        text = text.split(marker, 1)[-1]
    text = text.strip()
    if "\n\n" in text:
        text = text.split("\n\n", 1)[0].strip()
    return text


def _build_prompt(question: str, lang: str, tokenizer) -> str:
    content = PROMPTS[lang].format(question=question)
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": content}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return content


def _load_fasttext_detector():
    try:
        import fasttext

        model_path = Path("data/external/lid.176.bin")
        if not model_path.exists():
            raise FileNotFoundError(
                "missing data/external/lid.176.bin. Download it before running language-mismatch checks."
            )
        return ("fasttext", fasttext.load_model(str(model_path)))
    except Exception:
        from fast_langdetect import detect

        return ("fast_langdetect", detect)


def _detect_lang(detector, text: str, subset_hint: str) -> tuple[str, float]:
    backend, detector_obj = detector
    subset_hint = subset_hint.lower()
    if subset_hint.startswith("eng_"):
        expected_ft = "en"
    elif subset_hint.startswith("swa_"):
        expected_ft = "sw"
    elif subset_hint.startswith("lug_"):
        expected_ft = "lg"
    elif subset_hint.startswith("aka_"):
        expected_ft = "ak"
    elif subset_hint.startswith("amh_"):
        expected_ft = "am"
    else:
        expected_ft = subset_hint
    clean_text = text.replace("\n", " ") or " "
    if backend == "fasttext":
        labels, scores = detector_obj.predict(clean_text, k=1)
        detected = labels[0].replace("__label__", "")
        return detected, float(scores[0]), expected_ft
    result = detector_obj(text=clean_text)
    return str(result["lang"]), float(result["score"]), expected_ft


def _check_lang_mismatch(predictions, expected_subsets, row_ids):
    detector = _load_fasttext_detector()
    mismatches = []
    per_language_counts: dict[str, dict[str, int]] = {}
    for pred, expected, row_id in zip(predictions, expected_subsets, row_ids):
        detected, conf, expected_ft = _detect_lang(detector, str(pred), str(expected))
        bucket = per_language_counts.setdefault(str(expected), {"total": 0, "mismatch": 0})
        bucket["total"] += 1
        if detected != expected_ft and conf > 0.6:
            bucket["mismatch"] += 1
            mismatches.append(
                {
                    "id": row_id,
                    "expected": expected,
                    "detected": detected,
                    "confidence": conf,
                    "preview": str(pred)[:80],
                }
            )
    return mismatches, per_language_counts


def _load_generation_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    attempted = []
    model_chain = [
        "CohereForAI/aya-expanse-8b",
        "CohereForAI/aya-23-8b",
        "google/gemma-2-9b-it",
        "google/gemma-2-2b-it",
    ]
    has_cuda = torch.cuda.is_available()
    if not has_cuda:
        raise RuntimeError(
            "No CUDA device is available on this machine. Step 5 requires GPU-backed generation for this "
            "experiment; the 8B/9B models are impractical on CPU, and the 2B fallback is not realistic for "
            "full Val/Test generation at beam=5 over 9,304 rows."
        )
    for model_id in model_chain:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            if has_cuda:
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
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map="cpu",
                    trust_remote_code=True,
                )
            model.eval()
            return model_id, tokenizer, model
        except Exception as exc:
            attempted.append(f"{model_id}: {exc}")
    raise RuntimeError("failed to load any generation model: " + " | ".join(attempted))


def _generate_answers(df: pd.DataFrame, split_name: str, tokenizer, model, length_bounds, lang_col: str, question_col: str, id_col: str, lang_code_map: dict[str, str]) -> list[str]:
    import torch
    from tqdm import tqdm

    predictions: list[str] = []
    checkpoint_path = PROCESSED_DIR / f"predictions_{split_name}_partial.csv"
    if checkpoint_path.exists():
        partial = pd.read_csv(checkpoint_path, dtype=str).fillna("")
        predictions = partial["prediction"].tolist()
        print(f"Resuming from checkpoint at row {len(predictions)}")

    start_idx = len(predictions)
    model_device = getattr(model, "device", None)
    if model_device is None or str(model_device) == "meta":
        model_device = "cuda" if torch.cuda.is_available() else "cpu"

    for i in tqdm(range(start_idx, len(df)), desc=f"Generating [{split_name}]"):
        row = df.iloc[i]
        lang = canonical_prompt_language(lang_code_map[str(row[lang_col])])
        question = str(row[question_col])
        prompt = _build_prompt(question, lang, tokenizer)
        bounds = length_bounds[lang]
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                num_beams=5,
                length_penalty=1.0,
                no_repeat_ngram_size=3,
                early_stopping=True,
                min_new_tokens=bounds["min_new_tokens"],
                max_new_tokens=bounds["max_new_tokens"],
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        answer = _strip_prompt_artefacts(answer, lang)
        predictions.append(answer)

        if (i + 1) % 50 == 0:
            pd.DataFrame(
                {
                    "id": df[id_col].astype(str).tolist()[: len(predictions)],
                    "prediction": predictions,
                }
            ).to_csv(checkpoint_path, index=False, encoding="utf-8")
    return predictions


def step1_audit():
    _print_step("STEP 1 - INGEST AND AUDIT ALL FOUR FILES")
    train = read_csv(RAW_DIR / "Train.csv")
    val = read_csv(RAW_DIR / "Val.csv")
    test = read_csv(RAW_DIR / "Test.csv")
    sample = read_csv(RAW_DIR / "SampleSubmission.csv")
    print(f"Train.csv     : {len(train)} rows  {len(train.columns)} cols  |  columns: {train.columns.tolist()}")
    print(f"Val.csv       : {len(val)} rows  {len(val.columns)} cols  |  columns: {val.columns.tolist()}")
    print(f"Test.csv      : {len(test)} rows  {len(test.columns)} cols  |  columns: {test.columns.tolist()}")
    print(
        f"SampleSubmission.csv : {len(sample)} rows  {len(sample.columns)} cols  |  columns: {sample.columns.tolist()}"
    )

    mapping = detect_column_mapping(train, val, test)
    print()
    print("1a. Column name normalisation")
    print(asdict(mapping))

    combined_lang_values = (
        train[mapping.lang_col].astype(str).tolist()
        + val[mapping.lang_col].astype(str).tolist()
        + test[mapping.lang_col].astype(str).tolist()
    )
    lang_code_map = infer_lang_code_map(combined_lang_values)

    print()
    print("1b. Language distribution")
    train_dist = _language_distribution(train, mapping.lang_col)
    val_dist = _language_distribution(val, mapping.lang_col)
    print("Train distribution:")
    print(train_dist.to_string())
    print("Val distribution:")
    print(val_dist.to_string())
    gaps = []
    for raw_lang in sorted(set(train_dist.index) | set(val_dist.index)):
        train_pct = float(train_dist.loc[raw_lang, "pct"]) if raw_lang in train_dist.index else 0.0
        val_pct = float(val_dist.loc[raw_lang, "pct"]) if raw_lang in val_dist.index else 0.0
        gap = round(abs(train_pct - val_pct), 2)
        if train_pct < 10 or val_pct < 10:
            gaps.append(f"[FLAG] {raw_lang} below 10% in {'train' if train_pct < 10 else 'val'}")
        if gap > 15:
            gaps.append(f"[FLAG] {raw_lang} distribution gap is {gap}pp between Train and Val")
    if gaps:
        print("\n".join(gaps))
    else:
        print("No distribution flags.")

    print()
    print("1c. Answer length distribution (Train only)")
    train_len = _length_stats(train, mapping.lang_col, mapping.answer_col, lang_code_map)
    print(train_len.to_string(index=False))

    print()
    print("1d. Val answer length distribution")
    val_len = _length_stats(val, mapping.lang_col, mapping.answer_col, lang_code_map)
    print(val_len.to_string(index=False))
    p90_notes = []
    for raw_lang in sorted(set(train_len["raw_lang"]) & set(val_len["raw_lang"])):
        train_p90 = int(train_len.loc[train_len["raw_lang"] == raw_lang, "p90"].iloc[0])
        val_p90 = int(val_len.loc[val_len["raw_lang"] == raw_lang, "p90"].iloc[0])
        delta = val_p90 - train_p90
        if abs(delta) > 0:
            p90_notes.append(f"{raw_lang}: Val p90 - Train p90 = {delta}")
    print("\n".join(p90_notes) if p90_notes else "No p90 differences.")

    print()
    print("1e. Missing / null values")
    null_report = {}
    cleaned = {}
    for name, df in [("train", train), ("val", val), ("test", test), ("sample", sample)]:
        nulls = df.isna().sum().to_dict()
        null_report[name] = nulls
        print(f"{name}: {nulls}")
        cleaned[name] = df.copy()
    dropped_counts = {"train": 0, "val": 0}
    for name in ("train", "val"):
        df = cleaned[name]
        if df[mapping.answer_col].isna().any():
            before = len(df)
            df = df.dropna(subset=[mapping.answer_col]).reset_index(drop=True)
            dropped_counts[name] = before - len(df)
            cleaned[name] = df
    if cleaned["test"][mapping.question_col].isna().any():
        raise RuntimeError("critical: Test question column contains null values.")
    print(f"Dropped rows with null answers: {dropped_counts}")

    print()
    print("1f. Amharic script check")
    suspect_frames = []
    for name in ("train", "val", "test"):
        df = cleaned[name]
        mask = df[mapping.lang_col].astype(str).map(lambda x: lang_code_map[str(x)] == "amh")
        suspect = df.loc[mask & ~df[mapping.question_col].astype(str).map(contains_geez)].copy()
        if len(suspect):
            suspect["source_file"] = name
            suspect_frames.append(suspect)
        cleaned[name] = df.loc[~(mask & ~df[mapping.question_col].astype(str).map(contains_geez))].reset_index(drop=True)
        print(f"{name}: {len(suspect)} suspect Amharic rows quarantined")
    suspect_path = PROCESSED_DIR / "suspect_amharic.csv"
    if suspect_frames:
        pd.concat(suspect_frames, ignore_index=True).to_csv(suspect_path, index=False, encoding="utf-8")
    else:
        pd.DataFrame(columns=list(train.columns) + ["source_file"]).to_csv(
            suspect_path, index=False, encoding="utf-8"
        )
    print(f"Saved quarantine file: {suspect_path}")

    print()
    print("1g. Submission format fingerprint")
    submission_schema = build_submission_schema(sample, cleaned["test"], mapping.id_col)
    print(f"Exact columns: {list(submission_schema.columns)}")
    print(f"Dtypes: {submission_schema.dtypes}")
    print(f"Rows: {submission_schema.n_rows}")
    print(f"ID values match Test.csv exactly: {submission_schema.ids_match_test_exactly}")

    print()
    print("1h. Train/Val/Test overlap check")
    train_q = cleaned["train"][mapping.question_col].astype(str)
    val_q = cleaned["val"][mapping.question_col].astype(str)
    test_q = cleaned["test"][mapping.question_col].astype(str)
    overlap_train_val = sorted(set(train_q) & set(val_q))
    overlap_train_test = sorted(set(train_q) & set(test_q))
    overlap_val_test = sorted(set(val_q) & set(test_q))
    print(f"Train ∩ Val duplicate questions: {len(overlap_train_val)}")
    print(f"Train ∩ Test duplicate questions: {len(overlap_train_test)}")
    print(f"Val ∩ Test duplicate questions: {len(overlap_val_test)}")

    print()
    print("1i. Language code inventory")
    inventory = {
        "train": sorted(cleaned["train"][mapping.lang_col].astype(str).unique().tolist()),
        "val": sorted(cleaned["val"][mapping.lang_col].astype(str).unique().tolist()),
        "test": sorted(cleaned["test"][mapping.lang_col].astype(str).unique().tolist()),
    }
    print(json.dumps(inventory, indent=2, ensure_ascii=False))
    print("LANG_CODE_MAP =", json.dumps(lang_code_map, indent=2, ensure_ascii=False))

    audit_report = []
    audit_report.append("# Data Audit Report")
    audit_report.append("")
    audit_report.append("## File shapes")
    for label, df in [("Train.csv", train), ("Val.csv", val), ("Test.csv", test), ("SampleSubmission.csv", sample)]:
        audit_report.append(f"- {label}: {len(df)} rows, {len(df.columns)} cols, columns={df.columns.tolist()}")
    audit_report.append("")
    audit_report.append("## Column mapping")
    audit_report.append(f"- {asdict(mapping)}")
    audit_report.append("")
    audit_report.append("## Language distributions")
    audit_report.append("### Train")
    audit_report.append(_df_block(train_dist))
    audit_report.append("")
    audit_report.append("### Val")
    audit_report.append(_df_block(val_dist))
    audit_report.append("")
    audit_report.append("## Train answer lengths")
    audit_report.append(_df_block(train_len, index=False))
    audit_report.append("")
    audit_report.append("## Val answer lengths")
    audit_report.append(_df_block(val_len, index=False))
    audit_report.append("")
    audit_report.append("## Null report")
    audit_report.append("```json")
    audit_report.append(json.dumps(null_report, indent=2))
    audit_report.append("```")
    audit_report.append("")
    audit_report.append("## Submission schema")
    audit_report.append("```json")
    audit_report.append(json.dumps(asdict(submission_schema), indent=2))
    audit_report.append("```")
    audit_report.append("")
    audit_report.append("## Overlap counts")
    audit_report.append(f"- Train ∩ Val: {len(overlap_train_val)}")
    audit_report.append(f"- Train ∩ Test: {len(overlap_train_test)}")
    audit_report.append(f"- Val ∩ Test: {len(overlap_val_test)}")
    audit_report.append("")
    audit_report.append("## Language code map")
    audit_report.append("```json")
    audit_report.append(json.dumps(lang_code_map, indent=2, ensure_ascii=False))
    audit_report.append("```")
    (PROCESSED_DIR / "data_audit_report.md").write_text("\n".join(audit_report), encoding="utf-8")
    print()
    print("Saved full audit: data/processed/data_audit_report.md")
    print(
        "Summary: the live Zindi files use input/output/subset rather than the scaffold's Question/Response/Language names. "
        "The submission schema has four columns, not five, and the data includes English subsets alongside Akan, Amharic, Luganda, and Swahili. "
        "All downstream steps will use the detected column mapping and the sample submission contract rather than scaffold-era assumptions."
    )

    return {
        "train": cleaned["train"],
        "val": cleaned["val"],
        "test": cleaned["test"],
        "sample": sample,
        "mapping": mapping,
        "lang_code_map": lang_code_map,
        "submission_schema": submission_schema,
        "train_len": train_len,
        "val_len": val_len,
        "p90_notes": p90_notes,
    }


def step2_length_bounds(state):
    _print_step("STEP 2 - BUILD THE PER-LANGUAGE LENGTH BOUNDS TABLE")
    train_len = state["train_len"]
    val_len = state["val_len"]
    final_bounds = {}
    comparison_rows = []
    notes = []
    for lang in sorted(train_len["lang"].unique().tolist()):
        train_group = train_len[train_len["lang"] == lang]
        val_group = val_len[val_len["lang"] == lang]
        train_p10 = int(train_group["p10"].mean())
        train_p90 = int(train_group["p90"].mean())
        val_p10 = int(val_group["p10"].mean()) if len(val_group) else train_p10
        val_p90 = int(val_group["p90"].mean()) if len(val_group) else train_p90
        train_bound = {"min_new_tokens": max(1, int(train_p10 * 0.8)), "max_new_tokens": int(train_p90) + 30}
        val_bound = {"min_new_tokens": max(1, int(val_p10 * 0.8)), "max_new_tokens": int(val_p90) + 30}
        if abs(val_p90 - train_p90) > 20:
            avg_p10 = int(round((train_p10 + val_p10) / 2))
            avg_p90 = int(round((train_p90 + val_p90) / 2))
            chosen = {"min_new_tokens": max(1, int(avg_p10 * 0.8)), "max_new_tokens": int(avg_p90) + 30}
            notes.append(
                f"{lang}: Train/Val p90 diverged by {abs(val_p90 - train_p90)} tokens, using average-derived bounds."
            )
        else:
            chosen = train_bound
            if abs(val_p90 - train_p90) > 20:
                notes.append(f"{lang}: Val p90 differs from Train by {abs(val_p90 - train_p90)}.")
        final_bounds[lang] = chosen
        comparison_rows.append(
            {
                "lang": lang,
                "train_min": train_bound["min_new_tokens"],
                "train_max": train_bound["max_new_tokens"],
                "val_min": val_bound["min_new_tokens"],
                "val_max": val_bound["max_new_tokens"],
                "final_min": chosen["min_new_tokens"],
                "final_max": chosen["max_new_tokens"],
            }
        )
    comparison_df = pd.DataFrame(comparison_rows)
    print(comparison_df.to_string(index=False))
    if notes:
        print("\n".join(notes))
    else:
        print("No Train-vs-Val bound adjustments were required.")

    content = [
        '"""Auto-generated length bounds for exp001."""',
        "",
        f"LENGTH_BOUNDS = {json.dumps(final_bounds, indent=4, ensure_ascii=False)}",
        "",
    ]
    Path("src/afro_health_qa/data/length_bounds.py").write_text("\n".join(content), encoding="utf-8")
    print("Final LENGTH_BOUNDS =", json.dumps(final_bounds, indent=2, ensure_ascii=False))
    state["length_bounds"] = final_bounds
    state["length_bound_notes"] = notes
    return state


def step3_held_out(state):
    _print_step("STEP 3 - CARVE A HELD-OUT SLICE FROM TRAIN (5% ONLY)")
    train = state["train"].copy()
    mapping = state["mapping"]
    train_core, held_out = train_test_split(
        train,
        test_size=0.05,
        stratify=train[mapping.lang_col],
        random_state=42,
    )
    train_core = train_core.reset_index(drop=True)
    held_out = held_out.reset_index(drop=True)
    train_core.to_csv(PROCESSED_DIR / "train_core.csv", index=False, encoding="utf-8")
    held_out.to_csv(PROCESSED_DIR / "held_out.csv", index=False, encoding="utf-8")
    print("train_core per-language counts:")
    print(_language_distribution(train_core, mapping.lang_col).to_string())
    print("held_out per-language counts:")
    print(_language_distribution(held_out, mapping.lang_col).to_string())
    print()
    print("held_out.csv is now SEALED.")
    print("Val.csv is our local leaderboard proxy.")
    print("train_core.csv is everything available for training.")
    print("Do NOT load held_out.csv again until final week.")
    state["train_core"] = train_core
    state["held_out"] = held_out
    return state


def step4_calibration(state):
    _print_step("STEP 4 - BUILD AND CALIBRATE THE LOCAL EVALUATION PIPELINE")
    from afro_health_qa.evaluation.local_eval import calibration_checks

    mapping = state["mapping"]
    checks = calibration_checks(
        state["val"],
        answer_col=mapping.answer_col,
        question_col=mapping.question_col,
        lang_col=mapping.lang_col,
        lang_code_map=state["lang_code_map"],
    )
    print("CALIBRATION CHECK")
    perfect = checks["reports"]["perfect"]
    wrong = checks["reports"]["wrong_language"]
    perfect_judge = "SKIPPED" if perfect["judge"] is None else f"{perfect['judge']:.4f}"
    wrong_judge = "SKIPPED" if wrong["judge"] is None else f"{wrong['judge']:.4f}"
    print(
        "  Perfect breakdown: "
        f"R1={perfect['r1']:.4f} RL={perfect['rl']:.4f} AfroLM={perfect['afrolm_bs']:.4f} "
        f"Judge={perfect_judge} "
        f"mode={perfect['scoring_mode']}"
    )
    print(
        "  Wrong-lang breakdown: "
        f"R1={wrong['r1']:.4f} RL={wrong['rl']:.4f} AfroLM={wrong['afrolm_bs']:.4f} "
        f"Judge={wrong_judge} "
        f"mode={wrong['scoring_mode']}"
    )
    for warning in perfect["warnings"]:
        print("  Perfect warning:", warning)
    for warning in wrong["warnings"]:
        print("  Wrong-lang warning:", warning)
    print(
        f"Check A (prediction = reference) : {checks['check_a']:.4f}  "
        f"[{'PASS >= 0.95' if checks['check_a_pass'] else 'FAIL'}]"
    )
    print(
        f"Check B (wrong language shuffle)  : {checks['check_b']:.4f}  "
        f"[{'PASS <= 0.20' if checks['check_b_pass'] else 'FAIL'}]"
    )
    if not (checks["check_a_pass"] and checks["check_b_pass"]):
        raise RuntimeError("Calibration failed. Do not proceed to Step 5.")
    state["calibration"] = checks
    return state


def step5_generate_and_score(state):
    _print_step("STEP 5 - GENERATE PREDICTIONS: AYA-EXPANSE-8B ZERO-SHOT")
    from afro_health_qa.evaluation.local_eval import full_eval_report

    mapping = state["mapping"]
    model_id, tokenizer, model = _load_generation_model()
    print(f"Loaded generation model: {model_id}")

    val_predictions = _generate_answers(
        state["val"],
        "val",
        tokenizer,
        model,
        state["length_bounds"],
        mapping.lang_col,
        mapping.question_col,
        mapping.id_col,
        state["lang_code_map"],
    )
    pd.DataFrame(
        {mapping.id_col: state["val"][mapping.id_col].astype(str), "prediction": val_predictions}
    ).to_csv(PROCESSED_DIR / "val_predictions_raw.csv", index=False, encoding="utf-8")

    val_mismatches, val_mismatch_counts = _check_lang_mismatch(
        val_predictions,
        state["val"][mapping.lang_col].astype(str).tolist(),
        state["val"][mapping.id_col].astype(str).tolist(),
    )
    val_report = full_eval_report(
        val_predictions,
        state["val"],
        answer_col=mapping.answer_col,
        question_col=mapping.question_col,
        lang_col=mapping.lang_col,
        lang_code_map=state["lang_code_map"],
    )
    for warning in val_report["warnings"]:
        print(warning)

    print("LOCAL VAL SCORE (Zindi's official val set - best LB proxy)")
    print(f"Combined score : {val_report['combined']:.6f}")
    print(f"    ROUGE-1      : {val_report['r1']:.6f}  (weight 0.25)")
    print(f"    ROUGE-L      : {val_report['rl']:.6f}  (weight 0.25)")
    print(f"    AfroLM-BS    : {val_report['afrolm_bs']:.6f}  (weight 0.30)")
    judge_display = "SKIPPED (proxy active)" if val_report["judge"] is None else f"{val_report['judge']:.6f}"
    print(f"    LLM-Judge    : {judge_display}  (weight 0.20)")
    print()
    print("  Per-language combined:")
    for lang in sorted(val_report["per_language"]):
        print(f"    {lang} : {val_report['per_language'][lang]['combined']:.6f}")
    print()
    print(f"  Lang mismatch   : {len(val_mismatches)} / {len(state['val'])} total val rows")
    print(f"  Leaderboard #1  : {TARGET_LEADERBOARD}")
    delta = val_report["combined"] - TARGET_LEADERBOARD
    if abs(delta) <= 0.02:
        status = "WITHIN 0.02"
    elif delta > 0:
        status = "ABOVE"
    else:
        status = "BELOW"
    print(f"  Our val score   : {val_report['combined']:.6f}    [{status}]")
    print("  Mismatch counts by subset:")
    for subset, counts in sorted(val_mismatch_counts.items()):
        rate = counts["mismatch"] / counts["total"] if counts["total"] else 0.0
        print(f"    {subset}: {counts['mismatch']} / {counts['total']} ({rate:.2%})")

    for subset, counts in val_mismatch_counts.items():
        rate = counts["mismatch"] / counts["total"] if counts["total"] else 0.0
        if rate > 0.20:
            raise RuntimeError(f"STOP: language mismatch rate > 20% for {subset}")
    if val_report["combined"] < 0.50:
        raise RuntimeError("STOP: combined val score is below 0.50")
    if val_report["combined"] >= 0.75:
        print("Note: local Val score is already above the current public leaderboard leader.")

    test_predictions = _generate_answers(
        state["test"],
        "test",
        tokenizer,
        model,
        state["length_bounds"],
        mapping.lang_col,
        mapping.question_col,
        mapping.id_col,
        state["lang_code_map"],
    )
    pd.DataFrame(
        {mapping.id_col: state["test"][mapping.id_col].astype(str), "prediction": test_predictions}
    ).to_csv(PROCESSED_DIR / "test_predictions_raw.csv", index=False, encoding="utf-8")
    test_mismatches, test_mismatch_counts = _check_lang_mismatch(
        test_predictions,
        state["test"][mapping.lang_col].astype(str).tolist(),
        state["test"][mapping.id_col].astype(str).tolist(),
    )
    state["generation_model_id"] = model_id
    state["val_predictions"] = val_predictions
    state["test_predictions"] = test_predictions
    state["val_report"] = val_report
    state["val_mismatches"] = val_mismatches
    state["test_mismatches"] = test_mismatches
    state["val_mismatch_counts"] = val_mismatch_counts
    state["test_mismatch_counts"] = test_mismatch_counts
    return state


def step6_submission(state):
    _print_step("STEP 6 - BUILD, VALIDATE, AND SAVE THE SUBMISSION FILE")
    sample = state["sample"]
    test = state["test"]
    mapping = state["mapping"]
    print("SampleSubmission columns:", sample.columns.tolist())
    print("SampleSubmission dtypes:\n", sample.dtypes.to_string())
    print("SampleSubmission shape:", sample.shape)
    print("First 2 rows:\n", sample.head(2).to_string(index=False))

    submission_columns = tuple(sample.columns.tolist())
    submission = build_submission_df(
        test[mapping.id_col].astype(str).tolist(),
        state["test_predictions"],
        submission_columns=submission_columns,
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    submission_path = SUBMISSIONS_DIR / f"{timestamp}_exp001_aya_zeroshot.csv"

    checks = []
    checks.append(
        ("Column count matches SampleSubmission.csv exactly", submission.shape[1] == sample.shape[1], None)
    )
    checks.append(
        ("Column names match SampleSubmission.csv exactly", submission.columns.tolist() == sample.columns.tolist(), None)
    )
    checks.append(("Column order matches SampleSubmission.csv exactly", submission.columns.tolist() == sample.columns.tolist(), None))
    checks.append(("No extra columns, no pandas index column", "Unnamed: 0" not in submission.columns, None))
    sample_n = min(100, len(submission))
    identical_targets = all(
        submission.iloc[:sample_n, i].equals(submission.iloc[:sample_n, 1])
        for i in range(2, submission.shape[1])
    )
    checks.append(("All Target columns are identical per row (sample 100 rows)", identical_targets, None))
    checks.append((f"Row count = Test.csv row count exactly ({len(test)} rows)", len(submission) == len(test), None))
    checks.append(("ID values match Test.csv IDs exactly", set(submission.iloc[:, 0].astype(str)) == set(test[mapping.id_col].astype(str)), None))
    checks.append(("ID order matches Test.csv order exactly", submission.iloc[:, 0].astype(str).tolist() == test[mapping.id_col].astype(str).tolist(), None))
    checks.append(("No null / NaN in any column", not submission.isna().any().any(), None))
    target_cols = submission.columns[1:]
    checks.append(("No empty string (\"\") in any Target column", not (submission[target_cols] == "").any().any(), None))
    checks.append(
        (
            "No whitespace-only string in any Target column",
            not submission[target_cols].applymap(lambda x: str(x).strip() == "").any().any(),
            None,
        )
    )
    write_submission_csv(
        submission,
        submission_path,
        submission_columns=submission_columns,
    )
    with submission_path.open("rb") as handle:
        raw_head = handle.read(4)
    checks.append(("File encoding = UTF-8, no BOM", raw_head[:3] != b"\xef\xbb\xbf" and raw_head[:2] != b"\xff\xfe", None))
    max_chars = int(submission[target_cols[0]].astype(str).str.len().max())
    min_chars = int(submission[target_cols[0]].astype(str).str.len().min())
    checks.append(("No Target value longer than 1000 characters", max_chars <= 1000, None))
    checks.append(("No Target value shorter than 5 characters", min_chars >= 5, None))

    print("SUBMISSION VALIDATION CHECKLIST")
    all_passed = True
    for label, passed, _ in checks:
        if passed:
            print(f"[PASS] {label}")
        else:
            all_passed = False
            print(f"[FAIL] {label}")

    try:
        validate_submission_csv(
            submission_path,
            test_csv=RAW_DIR / "Test.csv",
            sample_csv=RAW_DIR / "SampleSubmission.csv",
        )
    except SubmissionValidationError as exc:
        all_passed = False
        print(f"[FAIL] Validator raised: {exc}")

    print(f"RESULT: [{'ALL PASSED - READY TO SUBMIT' if all_passed else 'FAILED - see above'}]")
    if not all_passed:
        raise RuntimeError("Submission validation failed.")

    print(f"Saved: {submission_path}")
    print("Encoding check: PASS")

    metadata = {
        "experiment_id": EXPERIMENT_ID,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hypothesis": HYPOTHESIS,
        "model_id": state["generation_model_id"],
        "decoding": "beam5, length_penalty=1.0, no_repeat_ngram_size=3",
        "augmentation": "none",
        "data_files": {
            "train": "data/raw/Train.csv",
            "val": "data/raw/Val.csv",
            "test": "data/raw/Test.csv",
            "sample_submission": "data/raw/SampleSubmission.csv",
        },
        "local_val_scores": {
            "combined": state["val_report"]["combined"],
            "rouge1": state["val_report"]["r1"],
            "rougeL": state["val_report"]["rl"],
            "afrolm_bs": state["val_report"]["afrolm_bs"],
            "judge": state["val_report"]["judge"],
            "per_language": {k: v["combined"] for k, v in state["val_report"]["per_language"].items()},
        },
        "lang_mismatch_count": {"val": len(state["val_mismatches"]), "test": len(state["test_mismatches"])},
        "val_rows_used": len(state["val"]),
        "test_rows_used": len(state["test"]),
        "submission_file": str(submission_path).replace("\\", "/"),
        "submission_row_count": len(submission),
        "submission_validation": "ALL_PASSED",
        "leaderboard_target": TARGET_LEADERBOARD,
        "public_lb_score": None,
        "notes": "First submission. Update public_lb_score after upload.",
    }
    sidecar_path = submission_path.with_suffix(submission_path.suffix + ".json")
    sidecar_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    state["submission_path"] = submission_path
    state["submission_metadata"] = metadata
    return state


def step7_log(state):
    _print_step("STEP 7 - EXPERIMENT LOG ENTRY")
    log_path = Path("experiments/LOG.md")
    line = (
        f"| {EXPERIMENT_ID} | {datetime.now().date().isoformat()} | "
        f"Zero-shot Aya beam5 lang-native prompts, beat 0.768095 | {state['generation_model_id']} | "
        f"none | beam5 lp=1.0 nrn=3 len-matched | {state['val_report']['r1']:.6f} | "
        f"{state['val_report']['rl']:.6f} | {state['val_report']['afrolm_bs']:.6f} | "
        f"{state['val_report']['judge'] if state['val_report']['judge'] is not None else 'SKIPPED'} | "
        f"{state['val_report']['combined']:.6f} | PENDING | "
        f"{str(state['submission_path']).replace('\\', '/')} | First run. Val.csv used as LB proxy. |"
    )
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\n" + line)
    print(line)
    return state


def step8_diagnostics(state):
    _print_step("STEP 8 - POST-RUN DIAGNOSTICS AND NEXT-STEP RECOMMENDATION")
    print("WEAKEST SUB-METRIC PER LANGUAGE")
    print("lang    ROUGE-1    ROUGE-L    AfroLM-BS    Judge    WEAKEST")
    for lang in sorted(state["val_report"]["per_language"]):
        metrics = state["val_report"]["per_language"][lang]
        metric_values = {
            "ROUGE-1": metrics["rouge1"],
            "ROUGE-L": metrics["rougeL"],
            "AfroLM-BS": metrics["afrolm_bs"],
            "Judge": metrics["judge"] if metrics["judge"] is not None else 1.0,
        }
        weakest = min(metric_values, key=metric_values.get)
        judge_val = "SKIP" if metrics["judge"] is None else f"{metrics['judge']:.2f}"
        print(
            f"{lang:<6} {metrics['rouge1']:.2f}       {metrics['rougeL']:.2f}        "
            f"{metrics['afrolm_bs']:.2f}       {judge_val}     {weakest}"
        )

    mapping = state["mapping"]
    val_df = state["val"].copy().reset_index(drop=True)
    val_df["prediction"] = state["val_predictions"]
    val_df["row_score"] = state["val_report"]["combined_per_row"]
    val_df["r1"] = state["val_report"]["rouge1_per_row"]
    val_df["rl"] = state["val_report"]["rougeL_per_row"]
    val_df["bs"] = state["val_report"]["afrolm_bs_per_row"]
    lang_code_map = state["lang_code_map"]
    val_df["canonical_lang"] = val_df[mapping.lang_col].astype(str).map(lambda x: lang_code_map[str(x)])

    for lang in sorted(val_df["canonical_lang"].unique()):
        examples = val_df[val_df["canonical_lang"] == lang].head(3)
        for _, row in examples.iterrows():
            print(f"[{lang}  row score: {row['row_score']:.2f}]")
            print(f"Q : {row[mapping.question_col]}")
            print(f"REF : {row[mapping.answer_col]}")
            print(f"GEN : {row['prediction']}")
            print(f"Gap : R1={row['r1']:.2f} RL={row['rl']:.2f} BS={row['bs']:.2f}")

    per_lang_scores = [v["combined"] for v in state["val_report"]["per_language"].values()]
    min_lang = min(per_lang_scores)
    max_lang = max(per_lang_scores)
    r1 = state["val_report"]["r1"]
    rl = state["val_report"]["rl"]
    bs = state["val_report"]["afrolm_bs"]
    judge = state["val_report"]["judge"] if state["val_report"]["judge"] is not None else 1.0

    if r1 < 0.60 and rl < 0.60:
        exp002 = "QLoRA fine-tune on train_core.csv, 3 epochs, r=32, alpha=64, all attention + MLP modules, Val.csv eval."
        hypothesis = "Fine-tuning Aya on train_core.csv will materially improve lexical overlap on Val.csv over zero-shot beam5."
    elif bs < 0.55:
        exp002 = "Add 2 few-shot examples per language to the prompt from train_core.csv and re-run generation."
        hypothesis = "Adding two short clean in-language exemplars per language will improve semantic similarity without fine-tuning."
    elif state["val_report"]["judge"] is not None and judge < 0.50:
        exp002 = (
            'Add to the prompt: "Provide a complete, factually accurate answer in {language}. '
            'Do not invent drug names, dosages, or procedures."'
        )
        hypothesis = "A stricter completeness-and-safety instruction will raise the judge score on Val.csv."
    elif max_lang - min_lang > 0.12:
        exp002 = "Augment the weakest language with the language-specific external corpus before the next run."
        hypothesis = "Targeted augmentation of the weakest language will close the per-language combined-score gap."
    elif state["val_report"]["combined"] >= 0.78:
        exp002 = "Submit immediately, then run fine-tuning as exp002 to push beyond the current proxy score."
        hypothesis = "A fine-tuned model on train_core.csv can improve an already leaderboard-leading zero-shot configuration."
    elif state["val_report"]["combined"] >= 0.75:
        exp002 = "Try length_penalty=0.8, num_beams=8, and candidate reranking by AfroLM similarity."
        hypothesis = "A broader beam with lighter length penalty and AfroLM reranking will outperform plain beam5 on Val.csv."
    else:
        exp002 = "QLoRA fine-tune on train_core.csv, 3 epochs, r=32, alpha=64, all attention + MLP modules, Val.csv eval."
        hypothesis = "Supervised fine-tuning on train_core.csv will outperform the current zero-shot baseline on Val.csv."

    print()
    print("exp002 recommendation:", exp002)
    print("exp002 hypothesis:", hypothesis)
    print()
    print(f"SUBMISSION READY TO UPLOAD: [{'YES' if state.get('submission_path') else 'NO'}]")
    print(f"File   : {str(state['submission_path']).replace('\\', '/')}")
    print(f"Rows   : {len(state['test'])}  (must match Test.csv: {len(state['test'])})")
    print("Checks : ALL 14 PASSED")
    print()
    print(f"Val score (our LB proxy) : {state['val_report']['combined']:.6f}")
    print(f"Leaderboard target       : {TARGET_LEADERBOARD}")
    status = "ABOVE TARGET" if state["val_report"]["combined"] > TARGET_LEADERBOARD else "BELOW TARGET - exp002 plan above"
    print(f"Status                   : {status}")
    print()
    print("NEXT STEPS:")
    print("1. Upload the file above to Zindi:")
    print("   https://zindi.africa/competitions/multilingual-health-question-answering-in-low-resource-african-languages-challenge")
    print("   My Submissions -> Upload -> select the file")
    print("2. While the score processes, read the 3 sample predictions per language (Step 8b).")
    print("3. When the public score comes back, paste it here.")
    print("   I'll update the LOG and tell you precisely what exp002 should be.")
    print()
    print("Submissions used : 1 / 50")
    print("Budget remaining : 49")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stop-after-step", type=int, default=8)
    args = parser.parse_args()

    _ensure_dirs()
    set_global_seed(42)
    state = step1_audit()
    if args.stop_after_step <= 1:
        return 0
    state = step2_length_bounds(state)
    if args.stop_after_step <= 2:
        return 0
    state = step3_held_out(state)
    if args.stop_after_step <= 3:
        return 0
    state = step4_calibration(state)
    if args.stop_after_step <= 4:
        return 0
    state = step5_generate_and_score(state)
    if args.stop_after_step <= 5:
        return 0
    state = step6_submission(state)
    if args.stop_after_step <= 6:
        return 0
    state = step7_log(state)
    if args.stop_after_step <= 7:
        return 0
    step8_diagnostics(state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
