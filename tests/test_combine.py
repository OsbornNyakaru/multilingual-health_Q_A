"""CPU tests for scripts/combine.py."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import combine as C  # noqa: E402

ANS = {"t1": "use a bed net", "t2": "see a doctor", "t3": "drink water"}


def test_agree_pick_uses_generator_only_inside_top_k():
    assert C.agree_pick("see a doctor", "t2|t1|t3", "use a bed net", ANS, 2) == ("use a bed net", "gen")
    assert C.agree_pick("see a doctor", "t2|t3|t1", "use a bed net", ANS, 2) == ("see a doctor", "sel")  # t1 is 3rd
    assert C.agree_pick("see a doctor", "t2|t1", "a brand new answer", ANS, 5) == ("see a doctor", "sel")  # not stored
    assert C.agree_pick("see a doctor", "t2|t1", "use a bed net", ANS, 0) == ("see a doctor", "sel")    # k=0 = off
    assert C.agree_pick("see a doctor", "t2|t1", "", ANS, 5) == ("see a doctor", "sel")                  # empty gen


def test_combine_set_applies_k_per_subset():
    sel = pd.DataFrame({"ID": ["a", "b"], "subset": ["Eng_Uga", "Lug_Uga"], "pred": ["see a doctor"] * 2, "cand_ids": ["t2|t1"] * 2})
    gen = pd.DataFrame({"ID": ["a", "b"], "pred": ["use a bed net"] * 2})
    out = C.combine_set(sel, gen, ANS, {"Eng_Uga": 2, "Lug_Uga": 0})
    assert out.set_index("ID")["pred"].to_dict() == {"a": "use a bed net", "b": "see a doctor"}
    assert out.set_index("ID")["source"].to_dict() == {"a": "gen", "b": "sel"}


def test_rescore_prefers_the_answer_that_matches_the_question():
    pool = pd.DataFrame({"ID": ["t1", "t2", "t3"], "subset": ["Lug_Uga"] * 3,
                         "input": ["siriimu ekika", "siriimu omukazi", "omusujja"],
                         "output": ["okusiigibwa siriimu kuba nga", "okufumbiriganwa n'oyo alina siriimu", "omusujja gw'ensiri"]})
    q = pd.DataFrame({"ID": ["e1"], "input": ["asobola okusiigibwa siriimu?"], "output": ["okusiigibwa siriimu kuba nga"]})
    base = pd.DataFrame({"ID": ["e1"], "subset": ["Lug_Uga"], "pred": ["okufumbiriganwa n'oyo alina siriimu"],
                         "cand_ids": ["t2|t1|t3"], "ret_ids": ["t2|t1|t3"]})
    off = C.rescore_set(base, q, pool, {"Lug_Uga": (0.0, 0.0)})
    assert off["pred"].iloc[0] == "okufumbiriganwa n'oyo alina siriimu"  # weights off: selector top-1 kept
    on = C.rescore_set(base, q, pool, {"Lug_Uga": (0.0, 8.0)})
    assert on["pred"].iloc[0] == "okusiigibwa siriimu kuba nga"          # answer overlap overrides rank 2 vs 1


def test_pool_pick_weights_the_base_answer():
    base = "take the full course of antibiotics"
    cands = ["drink clean water and rest", "drink clean water and rest well", "rest and drink clean water"]
    m = C.overlap_matrix([base] + cands)
    assert C.pool_pick(m, float("inf")) == 0           # inf keeps the base
    assert C.pool_pick(m, 0.0) in (1, 2, 3)            # no weight: the generated consensus wins
    assert C.pool_pick(C.overlap_matrix([base]), 1.0) == 0  # no candidates


def test_pool_apply_falls_back_to_greedy_without_cands():
    import pandas as pd

    base = pd.DataFrame({"ID": ["a"], "subset": ["Eng_Uga"], "pred": ["x y z"]})
    gen = pd.DataFrame({"ID": ["a"], "pred": ["x y z w"]})
    out = C.pool_apply(C.pool_rows(base, gen), {"Eng_Uga": 0.0})
    assert out["pred"].iloc[0] in ("x y z", "x y z w") and len(out) == 1


def test_twin_apply_routes_confident_twins_per_subset_and_falls_back_without_a_translation():
    base = pd.DataFrame({"ID": ["a", "b", "c", "d"], "subset": ["Aka_Gha", "Aka_Gha", "Amh_Eth", "Amh_Eth"],
                         "pred": ["gen a", "gen b", "gen c", "gen d"]})
    twin = pd.DataFrame({"ID": ["a", "b", "c", "d"], "twin_sim": [0.9, 0.8, 0.7, 0.9],
                         "pred": ["nllb a", "nllb b", "nllb c", "nllb d"], "llm_answer": ["llm a", "llm b", "llm c", None]})
    out = C.twin_apply(base, twin, {"Aka_Gha": ("nllb", 0.85), "Amh_Eth": ("llm", 0.6)})
    assert out["pred"].tolist() == ["nllb a", "gen b", "llm c", "gen d"]
    assert out["source"].tolist() == ["twin_nllb", "base", "twin_llm", "base"]
    assert C.twin_apply(base, twin, {})["pred"].tolist() == base["pred"].tolist()  # no rule: never route
