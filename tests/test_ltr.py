"""CPU tests for scripts/ltr.py feature building and picking."""

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
