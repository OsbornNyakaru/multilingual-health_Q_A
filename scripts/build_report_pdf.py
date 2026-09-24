"""Assemble the single combined retrospective PDF.

Parts:
  Cover
  A. The competition in plain English          (layman)
  B. What we built and where we landed         (honest status)
  C. The 11th-place solution and our gap        (comparison)
  D. Roadmap to top 10                          (concrete plan)
  Appendix. Exact numbers + file references

Pulls the *measured* dense-retrieval numbers from
docs/competition_report/data/bge_retrieval_scores.json so the narrative quotes
real figures. ROUGE figures are measured; the LLM-judge component is always
labelled estimated (never measured locally).

Output: docs/competition_report/afro_health_qa_report.pdf
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.flowables import HRFlowable

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "docs" / "competition_report"
FIG = BASE / "figures"
DATA = BASE / "data"
OUT = BASE / "afro_health_qa_report.pdf"

INK = colors.HexColor("#1d2530")
MUTED = colors.HexColor("#6b7686")
ENG = colors.HexColor("#3a6ea5")
WARN = colors.HexColor("#c44536")
ACCENT = colors.HexColor("#11796b")
GOLD = colors.HexColor("#9c6f12")
BOXBG = colors.HexColor("#f4f6f9")
BOXBG_WARN = colors.HexColor("#fbeeea")
BOXBG_GOOD = colors.HexColor("#eef7f4")
LINE = colors.HexColor("#d8dde4")

PAGE_W, PAGE_H = A4
MARGIN = 1.8 * cm
CONTENT_W = PAGE_W - 2 * MARGIN

# ---------------------------------------------------------------- styles -----
_ss = getSampleStyleSheet()


def _style(name, **kw):
    base = kw.pop("parent", _ss["Normal"])
    return ParagraphStyle(name, parent=base, **kw)


S = {
    "title": _style("title", fontName="Helvetica-Bold", fontSize=26, leading=30,
                    textColor=INK, alignment=TA_LEFT),
    "subtitle": _style("subtitle", fontName="Helvetica", fontSize=13, leading=17,
                       textColor=MUTED),
    "part": _style("part", fontName="Helvetica-Bold", fontSize=18, leading=22,
                   textColor=ENG, spaceBefore=6, spaceAfter=4),
    "h2": _style("h2", fontName="Helvetica-Bold", fontSize=13, leading=16,
                 textColor=INK, spaceBefore=10, spaceAfter=3),
    "body": _style("body", fontName="Helvetica", fontSize=10.5, leading=15.5,
                   textColor=INK, spaceAfter=6),
    "caption": _style("caption", fontName="Helvetica-Oblique", fontSize=8.6, leading=11,
                      textColor=MUTED, alignment=TA_CENTER, spaceAfter=10),
    "callout": _style("callout", fontName="Helvetica", fontSize=10.5, leading=15,
                      textColor=INK),
    "callout_h": _style("callout_h", fontName="Helvetica-Bold", fontSize=10.5, leading=15,
                        textColor=INK),
    "bullet": _style("bullet", fontName="Helvetica", fontSize=10.5, leading=15,
                     textColor=INK, leftIndent=14, spaceAfter=3, bulletIndent=2),
    "small": _style("small", fontName="Helvetica", fontSize=9, leading=12.5, textColor=MUTED),
    "cell": _style("cell", fontName="Helvetica", fontSize=9, leading=12, textColor=INK),
    "cellb": _style("cellb", fontName="Helvetica-Bold", fontSize=9, leading=12, textColor=INK),
    "coverkick": _style("coverkick", fontName="Helvetica-Bold", fontSize=11, leading=14,
                        textColor=ACCENT),
}


def measured() -> dict | None:
    p = DATA / "bge_retrieval_scores.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


# ------------------------------------------------------------- flowables -----
def fig(name, width=CONTENT_W, caption=None, story=None):
    path = FIG / name
    img = Image(str(path))
    iw, ih = img.imageWidth, img.imageHeight
    img.drawWidth = width
    img.drawHeight = ih * (width / iw)
    img.hAlign = "CENTER"
    block = [img]
    if caption:
        block.append(Spacer(1, 3))
        block.append(Paragraph(caption, S["caption"]))
    else:
        block.append(Spacer(1, 10))
    story.append(KeepTogether(block))


def para(text, story, style="body"):
    story.append(Paragraph(text, S[style]))


def bullets(items, story, style="bullet"):
    for it in items:
        story.append(Paragraph(f"•&nbsp;&nbsp;{it}", S[style]))
    story.append(Spacer(1, 6))


def callout(title, body_html, story, kind="info"):
    bg = {"info": BOXBG, "warn": BOXBG_WARN, "good": BOXBG_GOOD}[kind]
    bar = {"info": ENG, "warn": WARN, "good": ACCENT}[kind]
    inner = []
    if title:
        inner.append(Paragraph(title, S["callout_h"]))
        inner.append(Spacer(1, 2))
    inner.append(Paragraph(body_html, S["callout"]))
    t = Table([[inner]], colWidths=[CONTENT_W])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("LINEBEFORE", (0, 0), (0, -1), 3, bar),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
    ]))
    story.append(KeepTogether([t]))
    story.append(Spacer(1, 10))


def part_header(letter, title, story):
    story.append(Paragraph(f"Part {letter} &nbsp;·&nbsp; {title}", S["part"]))
    story.append(HRFlowable(width="100%", thickness=1.4, color=ENG, spaceAfter=8))


# ----------------------------------------------------------------- pages -----
def cover(story):
    story.append(Spacer(1, 2.2 * cm))
    story.append(Paragraph("AFRO-HEALTH-QA", S["coverkick"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph("What the competition was, where we landed, "
                           "and the road to top 10", S["title"]))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "A plain-English retrospective on the Zindi <b>Multilingual Health Question Answering "
        "in Low-Resource African Languages</b> challenge — and an honest comparison against the "
        "published 11th-place solution.", S["subtitle"]))
    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", thickness=1, color=LINE))
    story.append(Spacer(1, 10))
    meta = Table([
        [Paragraph("Team", S["small"]), Paragraph("Chiromo Forge — Osborn Nyakaru", S["cellb"])],
        [Paragraph("Prepared", S["small"]), Paragraph(date.today().strftime("%d %B %Y"), S["cell"])],
        [Paragraph("One-line thesis", S["small"]),
         Paragraph("We engineered a great car and never drove it. The winners drove a "
                   "simpler one — and the fuel was <b>retrieval</b>.", S["cell"])],
    ], colWidths=[3.4 * cm, CONTENT_W - 3.4 * cm])
    meta.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 1),
    ]))
    story.append(meta)
    story.append(Spacer(1, 14))
    callout("How to read this report",
            "Part A explains the competition with no jargon. Part B is an honest look at what we "
            "built. Part C compares us to the 11th-place finisher and names what we missed. "
            "Part D is the concrete plan to reach top 10. Numbers shown as ROUGE are "
            "<b>measured</b>; any LLM-judge figure is clearly labelled <b>estimated</b>.",
            story, kind="info")
    story.append(PageBreak())


def part_a(story):
    part_header("A", "The competition, in plain English", story)
    para("Imagine a free health helpline for young people across Africa — covering questions about "
         "the body, relationships, pregnancy, and sexually transmitted infections. People write in "
         "their own language. The machine has to write back a good answer <b>in that same "
         "language</b>. That is the whole competition: read a health question, write a helpful "
         "answer in the right language.", story)
    para("Eight 'subsets' were in play — the same kind of health questions asked in different "
         "languages and countries: English (in Uganda, Ghana, Kenya, Ethiopia), plus "
         "<b>Luganda</b> (Uganda), <b>Kiswahili</b> (Kenya), <b>Akan/Twi</b> (Ghana) and "
         "<b>Amharic</b> (Ethiopia).", story)
    fig("dataset_split.png", width=13.5 * cm, caption="Figure 1. We were handed ~30,000 example "
        "Q&A pairs to learn from, and 2,618 test questions whose answers are hidden.", story=story)

    para("A surprising fact shaped everything: <b>despite the 'African languages' billing, about "
         "61% of the data is English.</b> A quarter of it is English from Uganda alone. So the "
         "single biggest way to score points was simply to be very good at the English "
         "questions first.", story)
    fig("language_mix.png", width=15.5 * cm, caption="Figure 2. The language mix. Blue = English "
        "variants (~61%), orange = African languages (~39%).", story=story)

    para("<b>How answers are graded.</b> Your answer is compared to a reference 'model answer'. "
         "Most of the score — 74% — is just <i>word overlap</i> with that reference (a metric "
         "called ROUGE). The remaining 26% is an AI 'judge' rating quality. A third metric, "
         "AfroLM-BertScore, sounds important but is worth <b>zero</b> on the leaderboard.", story)
    fig("scoring_formula.png", width=12 * cm, caption="Figure 3. The scoring recipe. Because 74% is "
        "word-matching, the goal is to produce the reference's words — not the prettiest essay.",
        story=story)

    callout("Why this matters",
            "If the score is mostly word-overlap, then an answer that already exists in the "
            "training data — and matches a new question — is worth a lot. Hold that thought; it is "
            "the whole story of Part C.", story, kind="info")

    para("One more lever: <b>length</b>. ROUGE is an F1 score, so an answer that is too short or "
         "too long both lose points. Reference answers range from ~20 words (Amharic) to ~106 "
         "words (Akan) — a 5× spread. Matching each language's typical length is nearly-free "
         "points.", story)
    fig("answer_length.png", width=15.5 * cm, caption="Figure 4. Typical answer length by subset. "
        "Tuning output length per language is one of the cheapest ways to gain ROUGE.", story=story)
    story.append(PageBreak())


def part_b(story, m):
    part_header("B", "What we built — and where we actually landed", story)
    para("Here is the uncomfortable truth, stated plainly. We built an <b>excellent engineering "
         "scaffold</b>: clean train/validation/held-out splits, a faithful copy of the scoring "
         "metric, three model configurations with QLoRA fine-tuning recipes, an Apple-Silicon "
         "inference path, a reproducibility harness, and pages of strategy docs.", story)
    callout("…but the engine was never started",
            "Our experiment log contains exactly one row: the bootstrap commit. All ten of our "
            "planned experiments are still marked 'Pending'. <b>We never fine-tuned a model, never "
            "logged a single scored experiment, and never made one submission to Zindi.</b> Our "
            "only real measurement was a crude retrieval probe.",
            story, kind="warn")
    fig("our_status.png", width=15.5 * cm, caption="Figure 5. Built versus actually run. Green "
        "squares on the left, almost empty on the right. The whole story of our result is in that "
        "second column.", story=story)
    para("This is not a tooling failure — the tools are good. It is an <b>execution</b> failure: "
         "we polished the workshop instead of shipping a car. And the cost of that is exact and "
         "measurable, which Part C makes painfully clear.", story)
    story.append(PageBreak())


def part_c(story, m):
    part_header("C", "The 11th-place solution — and the gap", story)
    para("The 11th-place finisher (Koleshjr) wrote up their approach. They were refreshingly "
         "honest: they only got serious in the <b>final four days</b>. They did not test many "
         "models or fancy decoders. They did one thing well, and it was enough for 11th place.", story)

    para("<b>Their core insight: the training data is also a knowledge base.</b> Many test "
         "questions are near-duplicates or paraphrases of questions already answered in the "
         "training set. So instead of always generating an answer, you can <i>retrieve</i> the "
         "closest known question and reuse its answer. Because the score is mostly word-overlap, "
         "a real reference answer scores very high.", story)

    judge_txt = ("0.7379" if m is None else "0.7379")
    callout("Retrieval alone — no fine-tuning — was already strong (their numbers)",
            "Using BGE-M3 (a strong multilingual embedding model) to retrieve the nearest answer: "
            "<b>ROUGE-1 = 0.5548, ROUGE-L = 0.4823, LLM-judge = " + judge_txt + "</b> → a combined "
            "score of about <b>0.576</b>. That is a serious score from <i>copying</i> the right "
            "existing answer. Their best fine-tuned and RAG+fine-tuned models then reached ~0.66.",
            story, kind="good")

    # --- the measured-here lift ---
    if m:
        ov = m["overall"]
        mk = m.get("model_key")
        modtag = "BGE-M3" if mk == "bge-m3" else f"{mk} (a lighter proxy for BGE-M3)"
        proxy_note = "" if mk == "bge-m3" else (
            " We used a lighter embedding model than BGE-M3 here for speed, so treat this as a "
            "conservative floor — a full BGE-M3 run would score higher, as the 11th-place numbers show.")
        para(f"<b>We reproduced this with our own data.</b> Retrieving the nearest training answer "
             f"per language on our sealed held-out slice with {modtag} scores "
             f"<b>ROUGE-1 = {ov['rouge1']:.3f}, ROUGE-L = {ov['rougeL']:.3f}</b> "
             f"(ROUGE-only combined ≈ {ov['rouge_only_combined']:.3f}, before the judge term). "
             f"Compare that to the crude token-overlap probe we had been quoting — ROUGE-1 0.404, "
             f"ROUGE-L 0.350. <b>Swapping in proper dense embeddings lifts ROUGE-1 by roughly "
             f"+{ov['rouge1']-0.404:+.2f} for essentially no extra work.</b>{proxy_note}", story)
    else:
        para("<b>We reproduced this with our own data.</b> (Run "
             "<font face='Courier'>scripts/retrieval_baseline_bge.py</font> to fill in the measured "
             "numbers here.)", story)
    fig("retrieval_compare.png", width=13.5 * cm, caption="Figure 6. The same idea, three "
        "embedders. Better embeddings are worth ~+0.15 ROUGE for free — the lever we identified "
        "and never pulled.", story=story)

    para("And here is the part that stings. <b>Our own intelligence document already said all of "
         "this.</b> Months ago, <font face='Courier'>COMPETITION_INTEL.md</font> measured retrieval "
         "as strong, flagged that 41.6% of the English-Ethiopia questions were near-identical to "
         "training questions, and explicitly recommended a <i>retrieval + generation hybrid</i> — "
         "the exact shape of the winning approach. We wrote the winning idea down and then "
         "filed it away.", story)
    fig("retrieval_per_subset.png", width=15 * cm, caption="Figure 7. Even our crude retrieval "
        "already 'solved' the high-overlap English subsets; Amharic and Akan are the genuine hard "
        "floor.", story=story)

    para("Stacking the approaches on one ladder makes the gap concrete:", story)
    fig("score_ladder.png", caption="Figure 8. The score ladder. Our actual submitted score is the "
        "bottom rung — there isn't one. Retrieval alone would have cleared most of the climb.",
        story=story)

    story.append(Paragraph("What we missed", S["h2"]))
    bullets([
        "<b>Retrieval as a knowledge base.</b> The single highest-leverage idea — and it needed no "
        "GPU training to start scoring.",
        "<b>Dense embeddings.</b> We measured a weak crude-TF-IDF lower bound (0.40) and believed "
        "it; a proper embedder (BGE-M3) is far stronger.",
        "<b>RAG + fine-tuning.</b> Retrieve similar examples, put them in the prompt, and fine-tune "
        "the model to <i>adapt</i> them — the combination that reached ~0.66.",
        "<b>Shipping anything at all.</b> A finisher who started 4 days out still placed 11th. "
        "Submitting beats perfecting.",
    ], story)

    story.append(Paragraph("Where we went wrong", S["h2"]))
    bullets([
        "We invested in <b>infrastructure</b> (three model configs, a Mac path, a reproducibility "
        "harness) before running one real experiment.",
        "We <b>dismissed retrieval</b>, half-conflating 'don't chase external data' with 'don't "
        "reuse the labelled pool' — and threw out the most valuable signal in the data.",
        "We <b>trusted a crude measurement</b> instead of spending an afternoon on a proper one.",
        "Metric confusion (optimising the wrong weights, including a zero-weight metric) burned "
        "time early — documented in our own LESSONS.md.",
        "We never closed the loop: <b>no submission, no leaderboard feedback, no iteration.</b>",
    ], story)
    story.append(PageBreak())


def part_d(story):
    part_header("D", "The roadmap to top 10", story)
    para("The competition has closed, so this is a <b>practice plan</b> — the repeatable playbook "
         "that would have put us in contention, and will next time. The rule above all others: "
         "<b>each phase ships a scored submission before the next begins.</b> Measure, never "
         "estimate.", story)
    fig("roadmap.png", caption="Figure 9. Four phases, each ending in a real submission.", story=story)

    story.append(Paragraph("Phase 1 — Retrieve, and ship it (target ≈ 0.57)", S["h2"]))
    para("Build the BGE-M3 nearest-answer baseline (the very script in this repo). Retrieve from "
         "train + validation, per language, and submit. This locks a strong floor on day one and "
         "proves the whole pipeline end-to-end — exactly what we never did.", story)

    story.append(Paragraph("Phase 2 — RAG-enriched fine-tuning (target ≈ 0.66)", S["h2"]))
    para("For each training row, retrieve the k=3 most similar examples (excluding itself), put "
         "them in the prompt as context, and fine-tune a strong multilingual model (Aya-Expanse / "
         "AfriqueLlama / Sunbird-Sunflower) with LoRA so it learns to <i>adapt</i> the retrieved "
         "text rather than copy it. Serve with vLLM. This is the 11th-place winner's exact recipe.", story)

    story.append(Paragraph("Phase 3 — Route per subset (climb the board)", S["h2"]))
    para("Some subsets (high-overlap English) are won by retrieving directly; others (Amharic, "
         "Akan) need the fine-tuned model to generate. Build a simple router: if the nearest "
         "retrieved question is similar enough, return its answer; otherwise generate. Tune the "
         "similarity threshold on the held-out slice <b>by ROUGE</b> — never by an unscored metric.", story)

    story.append(Paragraph("Phase 4 — Squeeze the last ROUGE (into top 10)", S["h2"]))
    para("Per-subset length calibration, a short decoding sweep, a light two-model ensemble, and a "
         "final hedged pair (best-public vs best-held-out). These are small, compounding gains on "
         "top of a strong base.", story)

    callout("The one-sentence lesson",
            "We did not lose on ideas — our own notes held the winning strategy. We lost on "
            "<b>execution and humility</b>: trust the data, build the simplest thing that scores, "
            "submit it, and iterate from a real number.", story, kind="good")
    story.append(PageBreak())


def appendix(story, m):
    story.append(Paragraph("Appendix · Exact numbers & references", S["part"]))
    story.append(HRFlowable(width="100%", thickness=1.2, color=ENG, spaceAfter=8))

    story.append(Paragraph("Scoring formula (from src/afro_health_qa/evaluation/combined.py)", S["h2"]))
    para("combined = 0.37·ROUGE-1 F1 + 0.37·ROUGE-L F1 + 0.26·LLM-judge &nbsp;&nbsp;"
         "(AfroLM-BertScore = 0.00).", story)

    story.append(Paragraph("Retrieval, measured here vs reported", S["h2"]))
    rows = [["", "ROUGE-1", "ROUGE-L", "ROUGE-only comb.", "+ judge"]]
    rows.append(["Crude TF-IDF (our prior probe)", "0.404", "0.350", "0.279", "—"])
    if m:
        ov = m["overall"]
        lab = f"Dense ({m.get('model_key')}), measured"
        rows.append([lab, f"{ov['rouge1']:.3f}", f"{ov['rougeL']:.3f}",
                     f"{ov['rouge_only_combined']:.3f}", "not run"])
    rows.append(["BGE-M3, 11th place (reported)", "0.5548", "0.4823", "0.384", "0.576"])
    t = Table(rows, colWidths=[6.6 * cm, 2.4 * cm, 2.4 * cm, 3.2 * cm, 2.0 * cm])
    t.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8.6),
        ("FONT", (0, 1), (-1, -1), "Helvetica", 8.6),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("BACKGROUND", (0, 0), (-1, 0), BOXBG),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, LINE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BOXBG]),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t)
    story.append(Spacer(1, 6))
    para("ROUGE-only comb. = 0.37·R1 + 0.37·RL. The '+ judge' column adds the 0.26 LLM-judge term; "
         "we did <b>not</b> run the judge locally, so our dense row leaves it blank and the "
         "11th-place row uses their reported judge value. Note our local ROUGE uses rouge_score "
         "with stemmer=True, which can differ slightly from the leaderboard's whitespace "
         "tokenizer — treat absolute values as indicative, the gaps as real.", story, "small")

    story.append(Paragraph("Data at a glance", S["h2"]))
    para("Train 29,815 · Validation 6,686 · Test 2,618 · sealed held-out 1,491. Eight subsets; "
         "~61% English (Eng_Uga alone ~26%). Answer length 20–106 words by subset.", story, "small")

    story.append(Paragraph("Key files", S["h2"]))
    para("Charts: scripts/build_report_charts.py · Retrieval baseline: "
         "scripts/retrieval_baseline_bge.py · This report: scripts/build_report_pdf.py · "
         "Source-of-truth intel: autoresearch_nlp/COMPETITION_INTEL.md · Scoring: "
         "src/afro_health_qa/evaluation/combined.py.", story, "small")


# ----------------------------------------------------------------- footer ----
def _footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(LINE)
    canvas.line(MARGIN, 1.3 * cm, PAGE_W - MARGIN, 1.3 * cm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(MARGIN, 0.9 * cm, "Afro-Health-QA · retrospective & roadmap")
    canvas.drawRightString(PAGE_W - MARGIN, 0.9 * cm, f"{doc.page}")
    canvas.restoreState()


def build():
    m = measured()
    story = []
    cover(story)
    part_a(story)
    part_b(story, m)
    part_c(story, m)
    part_d(story)
    appendix(story, m)

    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN, topMargin=1.6 * cm, bottomMargin=1.7 * cm,
        title="Afro-Health-QA — retrospective & roadmap",
        author="Chiromo Forge — Osborn Nyakaru",
    )
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    print(f"[done] {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    build()
