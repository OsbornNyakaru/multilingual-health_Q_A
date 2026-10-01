"""CPU tests for autoresearch_nlp/experiment.py (retrieval, vote, router routing, config checks)."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "autoresearch_nlp"))
import experiment as E  # noqa: E402

pytest.importorskip("sklearn")


class Ctx:
    def __init__(self):
        self.cache, self.resume, self.saved = {}, {}, []

    def save(self, answers):
        self.saved.append(dict(answers))

    def should_stop(self):
        return False

    def log(self, msg, progress=None):
        pass

    def get_embedder(self, name):
        raise AssertionError("tfidf-char must not load an embedder")

    def get_model(self, *a):
        raise AssertionError("retrieval must not load a model")


POOL = pd.DataFrame(
    {
        "ID": [f"p{i}" for i in range(6)],
        "input": [
            "how do i prevent malaria during pregnancy",
            "how to prevent malaria in pregnancy",
            "what causes malaria in pregnant women",
            "symptoms of hiv infection in adults",
            "jinsi ya kuzuia malaria wakati wa ujauzito",
            "dalili za ukimwi kwa watu wazima",
        ],
        "output": ["use a bed net", "use a bed net", "mosquito bites", "fever and rash", "tumia chandarua", "homa"],
        "subset": ["Eng_Uga", "Eng_Uga", "Eng_Uga", "Eng_Uga", "Swa_Ken", "Swa_Ken"],
    }
)
EVAL = pd.DataFrame(
    {
        "ID": ["q1", "q2", "q3"],
        "input": ["how can i prevent malaria while pregnant", "hiv infection symptoms in adults", "kuzuia malaria ujauzito"],
        "subset": ["Eng_Uga", "Eng_Uga", "Swa_Ken"],
    }
)


def test_retrieval_top1_stays_within_subset():
    answers, meta = E.run({"mode": "retrieval", "embedder": "tfidf-char"}, EVAL, POOL, Ctx())
    assert answers["q1"] == "use a bed net"
    assert answers["q2"] == "fever and rash"
    assert answers["q3"] == "tumia chandarua"  # never an English answer for a Swahili question
    assert set(meta) == {"q1", "q2", "q3"} and 0 < meta["q1"]["sim"] <= 1


def test_vote_prefers_repeated_answer():
    answers, _ = E.run({"mode": "retrieval", "embedder": "tfidf-char", "select": "vote", "vote_k": 3, "vote_power": 1.0}, EVAL, POOL, Ctx())
    assert answers["q1"] == "use a bed net"


def test_router_copies_above_threshold_and_generates_below(monkeypatch):
    calls = {}

    def fake_generate(cfg, rows, examples_for, ctx, answers, pool_df):
        calls["ids"] = sorted(rows["ID"])
        if rows.empty:
            return
        calls["examples"] = examples_for(rows["ID"].iloc[0], rows["subset"].iloc[0])
        for i in rows["ID"]:
            answers[i] = "GEN"

    monkeypatch.setattr(E, "generate", fake_generate)
    cfg = {"mode": "router", "embedder": "tfidf-char", "router_threshold": 1.01, "router_generate_mode": "rag_few_shot", "few_shot_k": 2}
    answers, meta = E.run(cfg, EVAL, POOL, Ctx())
    assert all(m["route"] == "generate" for m in meta.values())
    assert calls["ids"] == ["q1", "q2", "q3"] and len(calls["examples"]) == 2
    answers, meta = E.run({**cfg, "router_threshold": 0.0}, EVAL, POOL, Ctx())
    assert all(m["route"] == "copy" for m in meta.values()) and "GEN" not in answers.values()


def test_resume_answers_are_kept(monkeypatch):
    monkeypatch.setattr(E, "generate", lambda cfg, rows, ex, ctx, answers, pool: answers.update({i: "NEW" for i in rows["ID"] if i not in answers}))
    ctx = Ctx()
    ctx.resume = {"q1": "OLD"}
    answers, _ = E.run({"mode": "zero_shot"}, EVAL, POOL, ctx)
    assert answers["q1"] == "OLD" and answers["q2"] == "NEW"


def test_unknown_config_key_rejected():
    with pytest.raises(ValueError, match="unknown config keys"):
        E.run({"mode": "retrieval", "embeder": "x"}, EVAL, POOL, Ctx())


def test_postprocess_strips_marker_and_trailing_paragraphs():
    assert E.postprocess("Answer: use a net\n\nMore text", "Eng_Uga") == "use a net"
    assert E.postprocess("Jibu: tumia chandarua", "Swa_Ken") == "tumia chandarua"
