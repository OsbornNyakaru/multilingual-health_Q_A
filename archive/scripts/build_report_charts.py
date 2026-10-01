"""Render the summary charts for the Afro-Health-QA retrospective report.

All competition numbers below are the verified figures from the data audit and
``autoresearch_nlp/COMPETITION_INTEL.md``. The retrieval-comparison and
score-ladder charts also fold in the freshly *measured* dense-retrieval result
written by ``scripts/retrieval_baseline_bge.py`` (read from the scores JSON).

Output: docs/competition_report/figures/*.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO = Path(__file__).resolve().parents[1]
FIG = REPO / "docs" / "competition_report" / "figures"
DATA = REPO / "docs" / "competition_report" / "data"
FIG.mkdir(parents=True, exist_ok=True)

# ---- palette ----------------------------------------------------------------
INK = "#1d2530"
MUTED = "#6b7686"
GRID = "#e6e9ee"
ENG = "#3a6ea5"      # English subsets
AFR = "#d1603d"      # African-language subsets
ACCENT = "#2a9d8f"
WARN = "#c44536"
GOLD = "#e0a32e"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "figure.dpi": 140,
    }
)

# ---- verified competition numbers ------------------------------------------
SPLIT = {"Train": 29_815, "Validation": 6_686, "Test": 2_618}

TRAIN_SUBSET = {  # full Train.csv (data audit)
    "Eng_Uga": 7624, "Aka_Gha": 4455, "Eng_Gha": 4443, "Eng_Eth": 3915,
    "Lug_Uga": 3383, "Eng_Ken": 2080, "Swa_Ken": 2070, "Amh_Eth": 1845,
}
ENGLISH = {"Eng_Uga", "Eng_Gha", "Eng_Eth", "Eng_Ken"}

ANSWER_LEN = {  # mean answer length in words
    "Amh_Eth": 20.2, "Eng_Eth": 24.5, "Eng_Gha": 75.1, "Eng_Ken": 78.7,
    "Lug_Uga": 79.7, "Swa_Ken": 84.3, "Eng_Uga": 95.4, "Aka_Gha": 105.6,
}

# crude token-overlap TF-IDF retrieval, whitespace ROUGE (COMPETITION_INTEL.md)
CRUDE = {
    "Aka_Gha": (0.294, 0.174), "Amh_Eth": (0.117, 0.110), "Eng_Eth": (0.569, 0.555),
    "Eng_Gha": (0.249, 0.165), "Eng_Ken": (0.459, 0.406), "Eng_Uga": (0.472, 0.426),
    "Lug_Uga": (0.443, 0.416), "Swa_Ken": (0.544, 0.505),
}
CRUDE_ALL = (0.404, 0.350)
CRUDE_COMB = 0.279  # 0.37*R1 + 0.37*RL, ROUGE-only

# 11th-place (Koleshjr) reported BGE-M3 retrieval-only
KOLESH = {"r1": 0.5548, "rl": 0.4823, "judge": 0.7379, "combined": 0.576}


def _measured() -> dict | None:
    p = DATA / "bge_retrieval_scores.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _style(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[chart] {name}")


# ---- 1. dataset split -------------------------------------------------------
def chart_dataset_split():
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    names = list(SPLIT)
    vals = list(SPLIT.values())
    bars = ax.bar(names, vals, color=[ENG, ACCENT, GOLD], width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 400, f"{v:,}", ha="center", fontweight="bold")
    _style(ax)
    ax.set_ylabel("rows (question–answer pairs)")
    ax.set_ylim(0, 33_000)
    ax.set_title("How much data the competition gave us")
    ax.text(0, -0.22, "Test answers are hidden — we submit predictions for 2,618 questions and Zindi grades them.",
            transform=ax.transAxes, color=MUTED, fontsize=9)
    _save(fig, "dataset_split.png")


# ---- 2. language mix --------------------------------------------------------
def chart_language_mix():
    items = sorted(TRAIN_SUBSET.items(), key=lambda kv: -kv[1])
    labels = [k for k, _ in items]
    vals = [v for _, v in items]
    colors = [ENG if k in ENGLISH else AFR for k in labels]
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    bars = ax.bar(labels, vals, color=colors, width=0.68)
    total = sum(vals)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 120, f"{v/total*100:.0f}%", ha="center", fontsize=9)
    _style(ax)
    ax.set_ylabel("training rows")
    ax.set_title("The data is mostly English — despite being an 'African languages' challenge")
    eng_share = sum(v for k, v in TRAIN_SUBSET.items() if k in ENGLISH) / total * 100
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=ENG, label=f"English variants ({eng_share:.0f}%)"),
                       Patch(color=AFR, label=f"African languages ({100-eng_share:.0f}%)")],
              frameon=False, loc="upper right")
    _save(fig, "language_mix.png")


# ---- 3. scoring formula -----------------------------------------------------
def chart_scoring_formula():
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    sizes = [37, 37, 26]
    labels = ["ROUGE-1\n(word overlap)", "ROUGE-L\n(sequence overlap)", "LLM judge\n(quality)"]
    colors = [ENG, "#5a8bbf", GOLD]
    wedges, _ = ax.pie(sizes, colors=colors, startangle=90, counterclock=False,
                       wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2))
    for w, lab, s in zip(wedges, labels, sizes):
        ang = (w.theta2 + w.theta1) / 2
        x, y = np.cos(np.radians(ang)) * 1.18, np.sin(np.radians(ang)) * 1.18
        ax.text(x, y, f"{lab}\n{s}%", ha="center", va="center", fontsize=10, fontweight="bold")
    ax.text(0, 0.12, "74%", ha="center", fontsize=22, fontweight="bold", color=INK)
    ax.text(0, -0.16, "is plain\nword-matching", ha="center", fontsize=9, color=MUTED)
    ax.set_title("How answers are scored")
    ax.text(0, -1.45, "AfroLM-BertScore = 0% on the leaderboard. Matching the reference wording matters most.",
            ha="center", color=MUTED, fontsize=9)
    _save(fig, "scoring_formula.png")


# ---- 4. answer length -------------------------------------------------------
def chart_answer_length():
    items = sorted(ANSWER_LEN.items(), key=lambda kv: kv[1])
    labels = [k for k, _ in items]
    vals = [v for _, v in items]
    colors = [ENG if k in ENGLISH else AFR for k in labels]
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    bars = ax.barh(labels, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(v + 1.5, b.get_y() + b.get_height() / 2, f"{v:.0f}", va="center", fontsize=9)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, linewidth=1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("mean answer length (words)")
    ax.set_title("Reference answers vary 5× in length — so length is a 'free' lever")
    ax.text(0, -0.2, "ROUGE is an F1 score: too-short or too-long answers both lose points. Match each language's typical length.",
            transform=ax.transAxes, color=MUTED, fontsize=9)
    _save(fig, "answer_length.png")


# ---- 5. retrieval per subset ------------------------------------------------
def chart_retrieval_per_subset():
    order = sorted(CRUDE, key=lambda k: -CRUDE[k][0])
    r1 = [CRUDE[k][0] for k in order]
    rl = [CRUDE[k][1] for k in order]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    ax.bar(x - 0.2, r1, 0.4, label="ROUGE-1", color=ENG)
    ax.bar(x + 0.2, rl, 0.4, label="ROUGE-L", color=ACCENT)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=20)
    _style(ax)
    ax.set_ylabel("F1")
    ax.set_ylim(0, 0.65)
    ax.legend(frameon=False)
    ax.set_title("Even crude retrieval already 'solves' the high-overlap subsets")
    ax.annotate("41.6% of these questions\nare near-identical to train",
                xy=(0, 0.57), xytext=(1.4, 0.6), fontsize=8.5, color=MUTED,
                arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.annotate("Amharic / Akan are the\nhard floor — a model must earn these",
                xy=(len(order) - 1, 0.12), xytext=(len(order) - 3.4, 0.27), fontsize=8.5, color=WARN,
                arrowprops=dict(arrowstyle="->", color=WARN))
    _save(fig, "retrieval_per_subset.png")


# ---- 6. retrieval comparison (the lift) ------------------------------------
def chart_retrieval_compare(meas):
    groups = ["Crude TF-IDF\n(our only measurement)", None, "BGE-M3 dense\n(11th place, reported)"]
    r1 = [CRUDE_ALL[0], None, KOLESH["r1"]]
    rl = [CRUDE_ALL[1], None, KOLESH["rl"]]
    if meas:
        mk = meas.get("model_key", "dense")
        tag = "BGE-M3 dense" if mk == "bge-m3" else f"{mk} dense (proxy)"
        groups[1] = f"{tag}\n(measured here)"
        r1[1] = meas["overall"]["rouge1"]
        rl[1] = meas["overall"]["rougeL"]
    else:
        groups[1] = "dense retrieval\n(measured here — pending)"
        r1[1] = 0.0
        rl[1] = 0.0
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    b1 = ax.bar(x - 0.2, r1, 0.4, label="ROUGE-1", color=ENG)
    b2 = ax.bar(x + 0.2, rl, 0.4, label="ROUGE-L", color=ACCENT)
    for bars in (b1, b2):
        for b in bars:
            h = b.get_height()
            if h:
                ax.text(b.get_x() + b.get_width() / 2, h + 0.008, f"{h:.3f}", ha="center", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=9.5)
    _style(ax)
    ax.set_ylabel("F1 (held-out)")
    ax.set_ylim(0, 0.66)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Better embeddings ≈ +0.15 ROUGE for free — the lever we left on the table")
    _save(fig, "retrieval_compare.png")


# ---- 7. score ladder --------------------------------------------------------
JUDGE_EST = KOLESH["judge"]  # illustrative judge level (11th place reported)


def chart_score_ladder(meas):
    # Each rung: label, measured/known ROUGE-only-or-combined, est-judge-addon, color, note
    rungs = [
        ("Our actual submitted score", 0.0, 0.0, WARN, "nothing was ever submitted"),
        ("Crude retrieval", CRUDE_COMB, 0.0, MUTED, "ROUGE-only, our prior probe"),
    ]
    if meas:
        mc = meas["overall"]["rouge_only_combined"]
        mk = meas.get("model_key", "dense")
        tag = "Dense retrieval here" + ("" if mk == "bge-m3" else " (proxy)")
        rungs.append((tag, mc, 0.26 * JUDGE_EST, ACCENT, "ROUGE measured + est. judge"))
    rungs += [
        ("BGE-M3 retrieval-only (11th place)", KOLESH["combined"], 0.0, ENG, "their reported combined"),
        ("Pure fine-tuning (11th place)", 0.66, 0.0, "#5a8bbf", "their reported number"),
        ("RAG + fine-tuning (11th place)", 0.665, 0.0, GOLD, "the winning combo"),
        ("Top of leaderboard", 0.70, 0.0, "#2f6b3c", "where top-10 lived"),
    ]
    labels = [r[0] for r in rungs]
    base = [r[1] for r in rungs]
    addon = [r[2] for r in rungs]
    colors = [r[3] for r in rungs]
    notes = [r[4] for r in rungs]
    y = np.arange(len(rungs))[::-1]
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    ax.barh(y, base, color=colors, height=0.5)
    # hatched estimated-judge extension (only where addon > 0)
    for yi, b, a in zip(y, base, addon):
        if a > 0:
            ax.barh(yi, a, left=b, height=0.5, facecolor="none", edgecolor=ACCENT,
                    hatch="////", linewidth=0)
    for yi, b, a, n in zip(y, base, addon, notes):
        total = b + a
        txt = "—  none" if total == 0 else (f"{b:.3f}" + (f"  +{a:.2f} ≈ {total:.2f}" if a > 0 else ""))
        ax.text(total + 0.01, yi, txt, va="center", fontsize=8.6, fontweight="bold")
        ax.text(0.01, yi + 0.30, n, va="center", fontsize=7.4, color=MUTED, style="italic")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.3)
    ax.set_xlim(0, 0.86)
    ax.set_ylim(-0.6, len(rungs) - 0.3)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, linewidth=1)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.axvline(KOLESH["combined"], color=ENG, linestyle="--", linewidth=1, alpha=0.45)
    ax.set_xlabel("competition score (combined)")
    ax.set_title("The score ladder: where each approach lands")
    ax.text(0.99, -0.14, "Hatched = estimated LLM-judge add-on (0.26 × judge); not measured locally.",
            transform=ax.transAxes, ha="right", fontsize=7.6, color=MUTED)
    _save(fig, "score_ladder.png")


# ---- 8. our status (built vs run) ------------------------------------------
def chart_our_status():
    rows = [
        ("Data splits & audit", 1, 1),
        ("Evaluation harness (ROUGE/judge)", 1, 1),
        ("Model + training configs (×3)", 1, 0),
        ("Mac/MLX + autoresearch scaffold", 1, 0),
        ("Retrieval baseline", 0, 0),
        ("Fine-tuned model", 0, 0),
        ("Scored experiment logged", 0, 0),
        ("Zindi submission made", 0, 0),
    ]
    labels = [r[0] for r in rows]
    built = [r[1] for r in rows]
    run = [r[2] for r in rows]
    y = np.arange(len(rows))[::-1]
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    for yi, b, r in zip(y, built, run):
        ax.scatter(0, yi, s=240, marker="s",
                   color=ACCENT if b else "#dfe3e9", edgecolor="white", zorder=3)
        ax.scatter(1, yi, s=240, marker="s",
                   color=ACCENT if r else "#dfe3e9", edgecolor="white", zorder=3)
        if b and not r:
            ax.text(1, yi, "✗", ha="center", va="center", color=WARN, fontsize=12, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["BUILT", "ACTUALLY RUN"], fontweight="bold")
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Our gap in one picture: we built a lot, and ran almost none of it")
    _save(fig, "our_status.png")


# ---- 9. roadmap -------------------------------------------------------------
def chart_roadmap():
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    phases = [
        ("1 · Retrieve", "BGE-M3 nearest-answer\nbaseline. Ship it.", "~0.57", ENG),
        ("2 · RAG + fine-tune", "Retrieve k=3 as context,\nfine-tune to adapt (LoRA).", "~0.66", ACCENT),
        ("3 · Route per subset", "High-overlap → retrieve,\nhard → fine-tuned LLM.", "↑", GOLD),
        ("4 · Squeeze ROUGE", "Length calibration,\ndecoding, ensemble, hedge.", "top-10", "#2f6b3c"),
    ]
    w, x0, gap = 2.0, 0.35, 0.25
    for i, (title, body, tag, col) in enumerate(phases):
        x = x0 + i * (w + gap)
        box = FancyBboxPatch((x, 1.4), w, 3.0, boxstyle="round,pad=0.08,rounding_size=0.12",
                             linewidth=1.5, edgecolor=col, facecolor=col + "18")
        ax.add_patch(box)
        ax.text(x + w / 2, 4.0, title, ha="center", fontweight="bold", color=col, fontsize=11)
        ax.text(x + w / 2, 2.95, body, ha="center", va="center", fontsize=8.6, color=INK)
        ax.text(x + w / 2, 1.75, tag, ha="center", fontweight="bold", color=col, fontsize=12)
        if i < 3:
            ax.add_patch(FancyArrowPatch((x + w + 0.02, 2.9), (x + w + gap - 0.02, 2.9),
                                         arrowstyle="-|>", mutation_scale=16, color=MUTED))
    ax.text(5, 5.4, "Roadmap to top 10", ha="center", fontweight="bold", fontsize=14, color=INK)
    ax.text(5, 0.7, "Each phase ships a scored submission before the next begins — measure, don't estimate.",
            ha="center", color=MUTED, fontsize=9)
    _save(fig, "roadmap.png")


def main():
    meas = _measured()
    if meas:
        mk = meas.get("model_key")
        print(f"[info] measured retrieval found: {mk} -> "
              f"R1={meas['overall']['rouge1']} RL={meas['overall']['rougeL']}")
    else:
        print("[info] no measured retrieval JSON yet — compare/ladder use placeholders.")
    chart_dataset_split()
    chart_language_mix()
    chart_scoring_formula()
    chart_answer_length()
    chart_retrieval_per_subset()
    chart_retrieval_compare(meas)
    chart_score_ladder(meas)
    chart_our_status()
    chart_roadmap()
    print(f"\n[done] charts in {FIG}")


if __name__ == "__main__":
    main()
