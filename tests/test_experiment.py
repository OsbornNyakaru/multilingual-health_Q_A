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


class FakeDense:
    """Bag-of-words 'embedder' over a tiny vocabulary, L2-normalised."""

    VOCAB = ["malaria", "pregnan", "hiv", "symptom", "kuzuia", "dalili", "prevent", "cause"]

    def encode(self, texts, **kw):
        import numpy as np

        v = np.array([[float(w in t.lower()) for w in self.VOCAB] for t in texts]) + 1e-6
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_hybrid_blends_two_retrievers():
    ctx = Ctx()
    ctx.get_embedder = lambda name: FakeDense()
    dense, _ = E.run({"mode": "retrieval", "embedder": "fake-dense"}, EVAL, POOL, ctx)
    hyb, meta = E.run({"mode": "retrieval", "embedder": "fake-dense", "hybrid_with": "tfidf-char", "hybrid_alpha": 0.5}, EVAL, POOL, ctx)
    assert hyb["q1"] == "use a bed net" and hyb["q3"] == "tumia chandarua"
    assert all(0 < m["sim"] <= 1.0001 for m in meta.values())


def test_rerank_reorders_candidates():
    class FakeCE:
        # prefers candidates whose text mentions "cause": flips q1 from "use a bed net" to "mosquito bites"
        def predict(self, pairs, **kw):
            return [1.0 if "cause" in cand else 0.1 for _, cand in pairs]

    ctx = Ctx()
    ctx.cache[("cross_encoder", "fake-ce")] = FakeCE()
    answers, meta = E.run({"mode": "retrieval", "embedder": "tfidf-char", "rerank_model": "fake-ce", "rerank_k": 3}, EVAL, POOL, ctx)
    assert answers["q1"] == "mosquito bites"
    assert meta["q1"]["rerank_score"] == 1.0 and meta["q1"]["rerank_id"] == "p2"


def test_training_rows_exclude_self_and_stay_in_subset():
    pool = POOL.assign(ID=POOL["ID"])
    rows = E.training_rows({**E.DEFAULT_CONFIG, "embedder": "tfidf-char", "few_shot_k": 2}, pool, Ctx())
    assert len(rows) == len(pool)
    for (q, a, ex, subset), (_, r) in zip(rows, pool.iterrows()):
        assert q == r["input"] and a == r["output"] and subset == r["subset"]
        assert (r["input"], r["output"][:400]) not in ex  # never its own Q&A
        same = set(pool[pool.subset == subset]["input"])
        assert all(eq in same for eq, _ in ex) and 1 <= len(ex) <= 2


def test_training_rows_respect_lora_train_subsets():
    keep = sorted(POOL.subset.unique())[:1]
    rows = E.training_rows({**E.DEFAULT_CONFIG, "embedder": "tfidf-char", "few_shot_k": 1, "lora_train_subsets": keep}, POOL, Ctx())
    assert rows and {sub for *_, sub in rows} == set(keep)
    assert len(rows) == int((POOL.subset == keep[0]).sum())


def test_lora_targets_skip_vision_towers():
    import re

    class Fake:  # torch-free stand-in for nn.Module.named_modules()
        def __init__(self, names):
            self.names = names

        def named_modules(self):
            return [(n, None) for n in self.names]

    assert E.lora_targets(Fake(["", "model.layers.0.self_attn.q_proj"])) == "all-linear"
    names = ["model.language_model.layers.0.self_attn.q_proj", "model.language_model.layers.0.mlp.down_proj",
             "model.vision_tower.encoder.layers.0.self_attn.q_proj", "model.embed_vision.embedding_projection"]
    pat = E.lora_targets(Fake(names))
    assert [n for n in names if re.fullmatch(pat, n)] == names[:2]


def test_setup_is_noop_for_non_lora_and_lora_rag_uses_trained_adapter(monkeypatch):
    ctx = Ctx()
    ctx.run_id = "r1"
    E.setup({"mode": "retrieval"}, {"held_out": (EVAL, POOL)}, ctx)
    assert ("lora_adapter", "r1") not in ctx.cache
    monkeypatch.setattr(E, "train_lora", lambda cfg, pool, c: f"/adapters/{len(pool)}")
    E.setup({"mode": "lora_rag", "embedder": "tfidf-char"}, {"val": (EVAL, POOL.iloc[:4]), "held_out": (EVAL, POOL)}, ctx)
    assert ctx.cache[("lora_adapter", "r1")] == "/adapters/6"  # held_out's pool wins over val's
    seen = {}

    def fake_generate(cfg, rows, examples_for, c, answers, pool_df):
        seen["adapter"], seen["ex"] = cfg["adapter"], examples_for(rows["ID"].iloc[0], rows["subset"].iloc[0])
        answers.update({i: "GEN" for i in rows["ID"]})

    monkeypatch.setattr(E, "generate", fake_generate)
    answers, _ = E.run({"mode": "lora_rag", "embedder": "tfidf-char", "few_shot_k": 2}, EVAL, POOL, ctx)
    assert seen["adapter"] == "/adapters/6" and len(seen["ex"]) == 2 and set(answers.values()) == {"GEN"}


def test_lora_rag_trains_inline_on_runners_without_setup(monkeypatch):
    calls = []
    monkeypatch.setattr(E, "train_lora", lambda cfg, pool, c: calls.append(len(pool)) or "/adapters/inline")
    monkeypatch.setattr(E, "generate", lambda cfg, rows, ex, c, answers, pool: answers.update({i: cfg["adapter"] for i in rows["ID"]}))
    ctx = Ctx()  # old runner: no run_id, no setup, no upload
    cfg = {"mode": "lora_rag", "embedder": "tfidf-char", "few_shot_k": 2}
    a1, _ = E.run(cfg, EVAL, POOL, ctx)
    a2, _ = E.run(cfg, EVAL, POOL, ctx)  # second eval set reuses the adapter
    assert calls == [6] and set(a1.values()) == set(a2.values()) == {"/adapters/inline"}


def test_fallback_replaces_short_generations_with_retrieval(monkeypatch):
    # first eval row gets an empty generation, the others a long one
    def fake_generate(cfg, rows, ex, c, answers, pool):
        for n, i in enumerate(rows["ID"]):
            answers[i] = "" if n == 0 else "word " * 20

    monkeypatch.setattr(E, "generate", fake_generate)
    cfg = {"mode": "rag_few_shot", "embedder": "tfidf-char", "few_shot_k": 2, "fallback_below_frac": 0.5}
    answers, meta = E.run(cfg, EVAL, POOL, Ctx())
    assert answers["q1"] == "use a bed net" and meta["q1"]["fallback"] is True
    assert answers["q2"].startswith("word") and "fallback" not in meta["q2"]
    off, _ = E.run({**cfg, "fallback_below_frac": 0.0}, EVAL, POOL, Ctx())
    assert off["q1"] == ""


def test_min_len_pct_sets_min_new_tokens():
    class Tok:
        def __call__(self, texts, add_special_tokens=False):
            return {"input_ids": [t.split() for t in texts]}

    pool = pd.DataFrame({"ID": [f"p{i}" for i in range(10)], "input": ["q"] * 10,
                         "output": [" ".join(["w"] * (i + 1) * 10) for i in range(10)], "subset": ["Aka_Gha"] * 10})
    cfg = {**E.DEFAULT_CONFIG}
    assert E.length_bounds(cfg, Tok(), pool)["Aka_Gha"]["min_new_tokens"] == 1
    b = E.length_bounds({**cfg, "min_len_pct": 50}, Tok(), pool)["Aka_Gha"]
    assert b["min_new_tokens"] == 55 and b["max_new_tokens"] > b["min_new_tokens"]


def test_resolve_adapter_passes_paths_through_and_fetches_runs(monkeypatch, tmp_path):
    assert E.resolve_adapter(None, Ctx()) is None and E.resolve_adapter("/a/b", Ctx()) == "/a/b"
    src = tmp_path / "hf" / "runs" / "r9" / "adapter"
    src.mkdir(parents=True)
    (src / "adapter_config.json").write_text("{}")
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: str(tmp_path / "hf"))
    monkeypatch.chdir(tmp_path)
    out = E.resolve_adapter("run:r9", Ctx())
    assert out.endswith("runner_work/adapters/r9") and (tmp_path / "runner_work/adapters/r9/adapter_config.json").exists()


def test_diag_k_records_candidate_ids_before_and_after_rerank():
    class FakeCE:
        def predict(self, pairs, **kw):
            return [1.0 if "cause" in cand else 0.1 for _, cand in pairs]

    ctx = Ctx()
    ctx.cache[("cross_encoder", "fake-ce")] = FakeCE()
    _, meta = E.run({"mode": "retrieval", "embedder": "tfidf-char", "rerank_model": "fake-ce", "rerank_k": 3, "diag_k": 3}, EVAL, POOL, ctx)
    ret, cand = meta["q1"]["ret_ids"].split("|"), meta["q1"]["cand_ids"].split("|")
    assert len(ret) == 3 and set(ret) == set(cand) and cand[0] == "p2" and ret[0] != "p2"
    _, meta = E.run({"mode": "retrieval", "embedder": "tfidf-char"}, EVAL, POOL, Ctx())
    assert "ret_ids" not in meta["q1"]


def test_reranker_groups_use_same_answer_positives_and_different_answer_negatives():
    pool = pd.DataFrame({
        "ID": [f"p{i}" for i in range(8)],
        "input": ["prevent malaria pregnancy", "avoid malaria when pregnant", "malaria prevention pregnant women",
                  "hiv symptoms adults", "signs of hiv in adults", "what causes malaria", "malaria causes", "treat malaria"],
        "output": ["net", "net", "net", "rash", "rash", "mosquito", "mosquito", "drugs"],
        "subset": ["Eng_Uga"] * 8,
    })
    cfg = {**E.DEFAULT_CONFIG, "embedder": "tfidf-char", "rerank_train_negs": 2}
    groups = E.reranker_groups(cfg, pool, Ctx())
    ans = dict(zip(pool["input"], pool["output"]))
    assert groups and all(q != cands[0] for q, cands, _ in groups)
    for q, cands, _ in groups:
        assert ans[cands[0]] == ans[q]                       # positive shares the answer
        assert all(ans[c] != ans[q] for c in cands[1:])       # negatives don't
        assert len(cands) == 3
    assert "treat malaria" not in [q for q, _, _ in groups]     # unique answer: no positive, skipped


def test_rerank_train_wires_trained_model_into_rerank(monkeypatch):
    seen = {}
    monkeypatch.setattr(E, "train_rerankers", lambda cfg, pool, c: seen.setdefault("pool", len(pool)) and "/rr/trained")

    class FakeCE:
        def predict(self, pairs, **kw):
            return [1.0 if "cause" in cand else 0.1 for _, cand in pairs]

    ctx = Ctx()
    ctx.run_id = "r7"
    ctx.cache[("cross_encoder", "/rr/trained")] = FakeCE()
    E.setup({"mode": "retrieval", "embedder": "tfidf-char", "rerank_model": "base-ce", "rerank_train": True},
            {"held_out": (EVAL, POOL)}, ctx)
    assert ctx.cache[("trained_reranker", "r7")] == "/rr/trained" and seen["pool"] == 6
    answers, meta = E.run({"mode": "retrieval", "embedder": "tfidf-char", "rerank_model": "base-ce", "rerank_train": True, "rerank_k": 3},
                          EVAL, POOL, ctx)
    assert answers["q1"] == "mosquito bites" and meta["q1"]["rerank_score"] == 1.0


TRIP_POOL = pd.DataFrame({
    "ID": [f"p{i}" for i in range(8)],
    "input": ["prevent malaria pregnancy", "avoid malaria when pregnant", "malaria prevention pregnant women",
              "hiv symptoms adults", "signs of hiv in adults", "what causes malaria", "malaria causes", "treat malaria"],
    "output": ["net", "net", "net", "rash", "rash", "mosquito", "mosquito", "drugs"],
    "subset": ["Lug_Uga"] * 5 + ["Eng_Uga"] * 3,
})


def test_embedder_triplets_pair_same_answers_and_respect_subset_filter():
    cfg = {**E.DEFAULT_CONFIG, "embedder": "tfidf-char"}
    trips = E.embedder_triplets(cfg, TRIP_POOL, Ctx())
    ans = dict(zip(TRIP_POOL["input"], TRIP_POOL["output"]))
    assert trips and all(a != p and ans[a] == ans[p] and ans[n] != ans[a] for a, p, n in trips)
    assert "treat malaria" not in [a for a, _, _ in trips]  # unique answer: no positive
    lug = E.embedder_triplets({**cfg, "embedder_train_subsets": ["Lug_Uga"]}, TRIP_POOL, Ctx())
    lug_qs = set(TRIP_POOL[TRIP_POOL.subset == "Lug_Uga"]["input"])
    assert lug and all(a in lug_qs for a, _, _ in lug)


def test_embedder_train_feeds_the_trained_embedder_to_the_reranker(monkeypatch):
    seen = {}
    monkeypatch.setattr(E, "train_embedder", lambda cfg, pool, c: "/emb/trained")
    monkeypatch.setattr(E, "train_rerankers", lambda cfg, pool, c: seen.setdefault("emb", cfg["embedder"]) and "/rr/t")
    ctx = Ctx()
    ctx.run_id = "r8"
    E.setup({"mode": "retrieval", "embedder_train": True, "rerank_train": True, "rerank_model": "base"},
            {"held_out": (EVAL, POOL)}, ctx)
    assert ctx.cache[("trained_embedder", "r8")] == "/emb/trained" and seen["emb"] == "/emb/trained"


def test_rerank_on_both_shows_question_and_answer_in_training_and_inference():
    cfg = {**E.DEFAULT_CONFIG, "embedder": "tfidf-char", "rerank_train_negs": 2, "rerank_on": "both", "rerank_answer_chars": 3}
    groups = E.reranker_groups(cfg, TRIP_POOL, Ctx())
    assert groups and all(" || " in c for _, cands, _ in groups for c in cands)
    assert all(len(c.split(" || ")[1]) <= 3 for _, cands, _ in groups for c in cands)
    seen = []

    class SpyCE:
        def predict(self, pairs, **kw):
            seen.extend(c for _, c in pairs)
            return [0.5] * len(pairs)

    ctx = Ctx()
    ctx.cache[("cross_encoder", "spy")] = SpyCE()
    E.run({"mode": "retrieval", "embedder": "tfidf-char", "rerank_model": "spy", "rerank_k": 2, "rerank_on": "both"}, EVAL, POOL, ctx)
    assert seen and all(" || " in c for c in seen)


def test_graded_labels_give_near_duplicate_answers_high_targets():
    pool = TRIP_POOL.copy()
    pool.loc[pool["ID"] == "p5", "output"] = "net"          # "what causes malaria" shares no answer...
    pool.loc[pool["ID"] == "p6", "output"] = "net please"   # ...but "malaria causes" has a near-copy of "net"
    cfg = {**E.DEFAULT_CONFIG, "embedder": "tfidf-char", "rerank_train_negs": 2, "rerank_train_graded": True}
    ans = dict(zip(pool["input"], pool["output"].str.strip()))
    groups = E.reranker_groups(cfg, pool, Ctx())
    for q, cands, labels in groups:
        assert labels[0] == 1.0 and len(labels) == len(cands)
        for c, lab in zip(cands[1:], labels[1:]):
            assert abs(lab - E.answer_overlap(ans[c], ans[q])) < 1e-9 and 0.0 <= lab < 1.0
    assert any(lab > 0 for _, _, labels in groups for lab in labels[1:])  # some negative got partial credit
    plain = E.reranker_groups({**cfg, "rerank_train_graded": False}, pool, Ctx())
    assert all(labels[1:] == [0.0] * 2 for _, _, labels in plain)


def test_answer_overlap_is_one_for_identical_and_zero_for_disjoint():
    assert E.answer_overlap("use a bed net", "use a bed net") == 1.0
    assert E.answer_overlap("use a bed net", "drink water") == 0.0


def test_rerank_averages_scores_across_an_ensemble():
    class CE:
        def __init__(self, fav):
            self.fav = fav

        def predict(self, pairs, **kw):
            return [1.0 if self.fav in cand else 0.0 for _, cand in pairs]

    ctx = Ctx()
    ctx.cache[("cross_encoder", "a")] = CE("cause")    # prefers p2 ("what causes malaria")
    ctx.cache[("cross_encoder", "b")] = CE("prevent")  # prefers p0/p1
    _, meta = E.run({"mode": "retrieval", "embedder": "tfidf-char", "rerank_model": ["a", "b"], "rerank_k": 3}, EVAL, POOL, ctx)
    assert meta["q1"]["rerank_score"] == 0.5  # every candidate averages to 0.5: one model each likes it


def test_train_rerankers_trains_one_per_seed(monkeypatch):
    monkeypatch.setattr(E, "train_reranker", lambda cfg, pool, c, seed=0: f"/rr/{seed}")
    assert E.train_rerankers({**E.DEFAULT_CONFIG}, POOL, Ctx()) == "/rr/0"
    assert E.train_rerankers({**E.DEFAULT_CONFIG, "rerank_ensemble": 3}, POOL, Ctx()) == ["/rr/0", "/rr/1", "/rr/2"]


def test_mbr_pick_returns_the_consensus_candidate():
    cands = ["drink clean water and rest", "zzz unrelated text", "drink clean water and rest well", "rest and drink clean water"]
    assert E.mbr_pick(cands) in (0, 2)
    assert E.mbr_pick(["only one"]) == 0
    assert E.mbr_pick(["a b", "x y", "a b", "a b c"]) in (0, 2)


def test_merge_vllm_output_greedy_only_and_mbr():
    ans, cands = E.merge_vllm_output({"greedy": " drink water ", "samples": []}, "Eng_Uga")
    assert ans == cands[0] == E.postprocess(" drink water ", "Eng_Uga") and len(cands) == 1
    rec = {"greedy": "zzz unrelated text", "samples": ["drink clean water and rest", "drink clean water and rest well", "rest and drink clean water"]}
    ans, cands = E.merge_vllm_output(rec, "Eng_Uga")
    assert len(cands) == 4 and ans.startswith("drink clean water")


def test_generate_dispatches_to_vllm(monkeypatch):
    seen = {}
    monkeypatch.setattr(E, "generate_vllm", lambda *a, **k: seen.setdefault("called", True))
    cfg = {**E.DEFAULT_CONFIG, "gen_engine": "vllm"}
    E.generate(cfg, pd.DataFrame(), None, Ctx(), {}, pd.DataFrame())
    assert seen.get("called")


def test_clean_env_drops_kernel_python_paths(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/tmp/uv-venv/lib/python3.13/site-packages")
    monkeypatch.setenv("HF_HOME", "/cache")
    env = E.clean_env()
    assert "PYTHONPATH" not in env and env["HF_HOME"] == "/cache"


def test_distinct_options_keeps_first_id_per_answer():
    answer_of = {"a": "Drink water.", "b": " Drink water.", "c": "Rest.", "d": "See a doctor."}
    assert E.distinct_options(["a", "x", "b", "c", "d"], answer_of, 2) == ["a", "c"]
    assert E.distinct_options(["x"], answer_of, 3) == []


def test_choose_scores_undoes_the_cyclic_shift():
    # option 1 wins everywhere: shift s shows option (s + j) % 3 at position j, so it sits at position (1 - s) % 3
    big, small = -0.01, -6.0
    shifts = {s: [big if (s + j) % 3 == 1 else small for j in range(3)] for s in range(3)}
    sc = E.choose_scores(shifts, 3)
    assert max(range(3), key=sc.__getitem__) == 1 and abs(sum(sc) - 1) < 1e-9
    # a letter missing from the top logprobs counts as very unlikely, not as an error
    assert E.choose_scores({0: [None, -0.1]}, 2)[1] > 0.99


def test_llm_choose_is_dispatched_before_retrieval(monkeypatch):
    seen = {}
    monkeypatch.setattr(E, "llm_choose", lambda cfg, ev, pool, ctx, answers, meta: seen.setdefault("called", cfg["mode"]))
    E.run({"mode": "llm_choose", "choose_from": "run:x"}, pd.DataFrame({"ID": [], "input": [], "subset": []}),
          pd.DataFrame({"ID": [], "input": [], "output": [], "subset": []}), Ctx())
    assert seen.get("called") == "llm_choose"


def test_choose_messages_lists_every_option_with_letters():
    m = E.choose_messages(" Is malaria contagious? ", "Lug_Uga", ["No.", "Yes."])
    assert len(m) == 1 and "Luganda" in m[0]["content"] and "A. No.\n\nB. Yes." in m[0]["content"]
    assert "(A-B)" in m[0]["content"] and "Question: Is malaria contagious?" in m[0]["content"]


CHOOSE_POOL = pd.DataFrame({
    "ID": [f"c{i}" for i in range(5)],
    "input": ["malaria prevention pregnancy", "prevent malaria pregnant women", "malaria pregnancy bed net",
              "hiv symptoms adults", "hiv signs in adults"],
    "output": ["sleep under a treated bed net", "sleep under a treated bed net", "take preventive malaria drugs",
               "fever rash and weight loss", "fever rash and weight loss"],
    "subset": ["Eng_Uga"] * 5,
})


def test_chooser_rows_leave_self_out_and_target_the_gold_answer():
    cfg = {**E.DEFAULT_CONFIG, "mode": "llm_choose", "embedder": "tfidf-char", "choose_k": 3, "choose_questions": True}
    rows = E.chooser_rows(cfg, CHOOSE_POOL, Ctx())
    assert rows, "every row with a same-answer neighbour yields a list"
    for msgs, letter in rows:
        text = msgs[0]["content"]
        opts = {L: text.split(f"\n\n{L}. ", 1)[1].split("\n", 1)[0] for L in "ABC" if f"\n\n{L}. " in text}
        q = text.split("Question: ", 1)[1].split("\n", 1)[0]
        gold = CHOOSE_POOL.set_index("input").loc[q, "output"]
        assert opts[letter] == gold  # target letter points at the row's own answer, wherever the shift put it
        assert f"(Dataset question with this answer: {q})" not in text  # the row itself is never an option
    low = E.chooser_rows({**cfg, "choose_train_min_overlap": 1.01}, CHOOSE_POOL, Ctx())
    assert low == []


def test_choose_train_feeds_the_trained_adapter_to_the_chooser(monkeypatch):
    ctx = Ctx()
    ctx.run_id = "r2"
    monkeypatch.setattr(E, "train_lora", lambda cfg, pool, c: f"/chooser/{cfg['mode']}/{len(pool)}")
    cfg = {"mode": "llm_choose", "choose_from": "run:x", "choose_train": True, "embedder": "tfidf-char"}
    E.setup(cfg, {"held_out": (EVAL, POOL)}, ctx)
    seen = {}
    monkeypatch.setattr(E, "llm_choose", lambda c, ev, pool, cx, answers, meta: seen.setdefault("adapter", c["adapter"]))
    E.run(cfg, EVAL, POOL, ctx)
    assert seen["adapter"] == "/chooser/llm_choose/6"
    E.setup({**cfg, "choose_train": False}, {"held_out": (EVAL, POOL)}, Ctx())  # zero-shot chooser: no training


def test_lora_sequences_pick_chooser_lists_or_rag_answers(monkeypatch):
    monkeypatch.setattr(E, "chooser_rows", lambda cfg, pool, ctx: [("M", "B")])
    assert E.lora_sequences({**E.DEFAULT_CONFIG, "mode": "llm_choose"}, POOL, Ctx()) == [("M", "B")]
    seqs = E.lora_sequences({**E.DEFAULT_CONFIG, "mode": "lora_rag", "embedder": "tfidf-char", "few_shot_k": 1}, POOL, Ctx())
    assert len(seqs) == len(POOL) and seqs[0][1] == POOL["output"].iloc[0] and seqs[0][0][-1]["role"] == "user"
