"""CPU tests for scripts/ltr.py feature building and picking."""

import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ltr as L  # noqa: E402

POOL = pd.DataFrame({"ID": ["t1", "t2", "t3", "t4"], "subset": ["Lug_Uga"] * 4,
                     "input": ["siriimu okusiigibwa", "siriimu okufumbiriganwa", "omusujja", "siriimu okusiigibwa omwana"],
                     "output": ["okusiigibwa kuba nga", "okufumbiriganwa kisoboka", "omusujja gw'ensiri", "okusiigibwa kuba nga"]})
Q = pd.DataFrame({"ID": ["e1"], "subset": ["Lug_Uga"], "input": ["asobola okusiigibwa siriimu?"], "output": ["okusiigibwa kuba nga"]})
PICK = pd.DataFrame({"ID": ["e1"], "cand_ids": ["t2|t1|t3"], "ret_ids": ["t1|t2|t3"]})


def test_build_features_one_row_per_distinct_answer_with_labels():
    f = L.build_features(Q, POOL, [PICK], None, ["Lug_Uga"])
    assert sorted(f["answer"]) == sorted({"okusiigibwa kuba nga", "okufumbiriganwa kisoboka", "omusujja gw'ensiri"})
    row = f.set_index("answer").loc["okusiigibwa kuba nga"]
    assert row["label"] == 1.0 and row["logrank_0"] < 0 and row["lfreq"] > 0  # answer shared by 2 pool rows
    assert f["gen_overlap"].isna().all() and set(L.feature_cols(f)) <= set(f.columns)


def test_pick_takes_the_highest_scored_candidate():
    class M:
        def predict(self, X):
            return X["q_ans"].to_numpy()

    f = L.build_features(Q, POOL, [PICK], None, ["Lug_Uga"])
    assert L.pick(f, M())["pred"].iloc[0] == "okusiigibwa kuba nga"


def test_v2_features_echo_agreement_and_relative(monkeypatch):
    monkeypatch.setattr(L, "FEATS", "v2")
    f = L.build_features(Q, POOL, [PICK], None, ["Lug_Uga"]).set_index("answer")
    assert set(L.feature_cols(f.reset_index())) <= set(f.columns)
    shared = f.loc["okusiigibwa kuba nga"]
    assert abs(shared["l_echo"] - math.log(2)) < 1e-9  # its 2 pool rows, no other near-duplicate
    assert shared["cand_echo"] == 0 and 0 <= shared["cand_mean"] <= shared["cand_max"] <= 1
    assert abs(f["q_ans_z"].mean()) < 1e-6 and (f["q_ans_gap"] <= 0).all() and (f["q_ans_gap"] == 0).any()


def test_preds_for_column_suffix_uses_that_ranking_as_candidates(monkeypatch, tmp_path):
    (tmp_path / "val_preds.csv").write_text("ID,cand_ids,tr_ids\ne1,t2|t1,t3|t1\n")
    monkeypatch.setattr(L.C, "run_dir", lambda e: tmp_path)
    assert L.preds_for("EXP-088", "val")["cand_ids"].iloc[0] == "t2|t1"
    assert L.preds_for("EXP-088:tr_ids", "val")["cand_ids"].iloc[0] == "t3|t1"
