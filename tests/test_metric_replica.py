"""CPU-only, fast tests for the local leaderboard-metric replica."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from afro_health_qa.evaluation.combined import W_JUDGE, W_R1, W_RL, combined_score, rouge_only_score
from afro_health_qa.evaluation.judge import (
    build_judge_prompt,
    normalize_judge_score,
    parse_judge_score,
    run_judge,
)
from afro_health_qa.evaluation.rouge import (
    TOKENIZER_CHOICES,
    UnicodeWordTokenizer,
    WhitespaceTokenizer,
    score_rouge,
)
from afro_health_qa.evaluation.scorer import load_predictions, score_rows, subset_from_id, summarize

AMH_REF = "የእርግዝና ምርመራ በሆስፒታል ማድረግ ይመከራል። እናቶች፣ ክትትል ያስፈልጋቸዋል።"
AMH_PRED = "የእርግዝና ምርመራ ማድረግ ይመከራል።"


def test_weights_sum_to_one_and_match_leaderboard():
    assert (W_R1, W_RL, W_JUDGE) == (0.37, 0.37, 0.26)
    assert math.isclose(W_R1 + W_RL + W_JUDGE, 1.0)
    assert math.isclose(combined_score(1.0, 1.0, judge=1.0), 1.0)
    assert math.isnan(combined_score(0.5, 0.5))  # no judge -> NaN, never silently 0
    assert math.isclose(rouge_only_score(1.0, 1.0), 0.74)


@pytest.mark.parametrize("tokenizer", TOKENIZER_CHOICES)
def test_identical_text_scores_one(tokenizer):
    text = "Folic acid before and during early pregnancy prevents neural tube defects."
    r = score_rouge([text], [text], tokenizer=tokenizer)
    assert r.rouge1 == pytest.approx(1.0) and r.rouge_l == pytest.approx(1.0)


@pytest.mark.parametrize("tokenizer", ["whitespace", "unicode"])
def test_identical_amharic_scores_one(tokenizer):
    r = score_rouge([AMH_REF], [AMH_REF], tokenizer=tokenizer)
    assert r.rouge1 == pytest.approx(1.0)


def test_default_tokenizer_erases_amharic_but_ours_do_not():
    from rouge_score.tokenizers import DefaultTokenizer

    assert DefaultTokenizer(use_stemmer=False).tokenize(AMH_REF) == []
    assert score_rouge([AMH_PRED], [AMH_REF], tokenizer="default").rouge1 == 0.0

    ws = WhitespaceTokenizer().tokenize(AMH_REF)
    uw = UnicodeWordTokenizer().tokenize(AMH_REF)
    assert len(ws) == 8 and "ይመከራል።" in ws  # starter keeps Ethiopic punctuation attached
    assert len(uw) == 8 and "ይመከራል" in uw and not any("።" in t or "፣" in t for t in uw)
    assert score_rouge([AMH_PRED], [AMH_REF], tokenizer="whitespace").rouge1 == pytest.approx(2 / 3)
    assert score_rouge([AMH_PRED], [AMH_REF], tokenizer="unicode").rouge1 == pytest.approx(2 / 3)


def test_whitespace_is_case_sensitive_like_starter():
    assert score_rouge(["Pregnancy test"], ["pregnancy test"], tokenizer="whitespace").rouge1 == pytest.approx(0.5)
    assert score_rouge(["Pregnancy test"], ["pregnancy test"], tokenizer="unicode").rouge1 == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"accuracy": 5, "completeness": 4, "language": 5, "overall": 4}', 4.0),
        ('Sure! {"accuracy": 3, "completeness": 3, "language": 5}', 11 / 3),
        ("Score: 2", 2.0),
        ("**Overall**: 5", 5.0),
        ("I'd rate it 3/5.", 3.0),
        ("4", 4.0),
        ('{"overall": 7}', None),
        ("no idea", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_judge_score(text, expected):
    got = parse_judge_score(text)
    assert got == (pytest.approx(expected) if expected is not None else None)


def test_normalize_judge_score():
    assert normalize_judge_score(1) == 0.0
    assert normalize_judge_score(5) == 1.0
    assert normalize_judge_score(3) == 0.5
    assert math.isnan(normalize_judge_score(None))


def test_prompt_contains_rubric_and_language():
    p = build_judge_prompt("Q?", "Ref.", "Ans.", "Amharic")
    for needle in ("accuracy", "completeness", "language", "Amharic", "Ref.", "Ans.", "1-5"):
        assert needle in p


def _frame():
    return pd.DataFrame(
        {
            "ID": ["ID_TR_Eng_Uga_1", "ID_TR_Eng_Uga_2", "ID_TR_Amh_Eth_3"],
            "question": ["q1", "q2", "q3"],
            "reference": ["a b c d", "a b", AMH_REF],
            "subset": ["Eng_Uga", "Eng_Uga", "Amh_Eth"],
            "pred_r1": ["a b c d", "x y", AMH_REF],
            "pred_rl": ["a b c d", "x y", AMH_REF],
            "pred_llm": ["a b c d", "x y", AMH_REF],
        }
    )


def test_per_subset_aggregation_without_judge():
    table = summarize(score_rows(_frame()))
    assert list(table.index) == ["Amh_Eth", "Eng_Uga", "ALL"]
    assert table.loc["Eng_Uga", "n"] == 2
    assert table.loc["Eng_Uga", "rouge1"] == pytest.approx(0.5)
    assert table.loc["Amh_Eth", "rouge1"] == pytest.approx(1.0)
    assert table.loc["ALL", "rouge1"] == pytest.approx(2 / 3)  # row-weighted, not mean of subsets
    assert table.loc["ALL", "rouge_only"] == pytest.approx(0.74 * 2 / 3)
    assert table["judge"].isna().all() and table["total"].isna().all()


def test_fake_judge_fills_judge_and_total():
    calls = []

    def fake_judge(prompts):
        calls.append(len(prompts))
        return ['{"overall": 5}' if "a b c d" in p.split("Candidate answer:")[1] else "Score: 1" for p in prompts]

    rows = score_rows(_frame(), judge_fn=fake_judge)
    table = summarize(rows)
    assert sum(calls) == 3
    assert list(rows["judge"]) == [1.0, 0.0, 0.0]
    assert table.loc["Eng_Uga", "judge"] == pytest.approx(0.5)
    assert table.loc["ALL", "total"] == pytest.approx(0.74 * 2 / 3 + 0.26 * (1 / 3))


def test_run_judge_counts_parse_failures_as_nan():
    res = run_judge(lambda ps: ["garbage"] * len(ps), ["q"], ["r"], ["a"], ["Swa_Ken"])
    assert res.parse_failures == 1 and math.isnan(res.mean)


def test_load_predictions_submission_and_single_column(tmp_path):
    sub = tmp_path / "sub.csv"
    pd.DataFrame({"ID": ["x"], "TargetRLF1": ["rl"], "TargetR1F1": ["r1"], "TargetLLM": ["llm"]}).to_csv(sub, index=False)
    p = load_predictions(sub)
    assert (p.loc[0, "pred_r1"], p.loc[0, "pred_rl"], p.loc[0, "pred_llm"]) == ("r1", "rl", "llm")

    single = tmp_path / "preds.csv"
    pd.DataFrame({"ID": ["x"], "output": ["ref"], "prediction": ["ans"]}).to_csv(single, index=False)
    p = load_predictions(single)
    assert p.loc[0, "pred_r1"] == "ans"


def test_subset_from_id():
    assert subset_from_id("ID_TS_Amh_Eth_A3B1799D") == "Amh_Eth"
