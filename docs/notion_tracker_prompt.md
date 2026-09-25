# notion ai prompt: experiment archive

paste everything inside the box below into notion ai, on a new empty page.

```
you are setting up the research archive for a two-person team (osborn and vera) running controlled experiments on a multilingual health question-answering model. treat this like a lab notebook at a top research lab: every run is recorded the same way, nothing is lost, and the best result is always obvious. use lowercase for all titles, headings, property names and options.

context
- task: zindi "multilingual health question answering in low-resource african languages". models answer health questions in akan, amharic, luganda, swahili and english.
- we run one notebook on molab (a cloud gpu). each run changes exactly one setting, scores the model on a fixed 2,088-question held-out set (or a fixed 200/500-question sample of it), and we decide keep or drop.
- score shown in the notebook: combined = 0.37 × rouge-1 f1 + 0.37 × rouge-l f1 (max 0.74, the ai-judge part is not measured locally). higher is better.
- a change is kept only if it beats the current best by at least 0.003 on the full set (rows = all). smaller gaps are noise.
- bar to beat: retrieval baseline ≈ 0.37 combined.
- the 8 subsets: aka_gha, amh_eth, eng_eth, eng_gha, eng_ken, eng_uga, lug_uga, swa_ken.

build the following.

1. a page called "afro health qa lab" with this structure:
   - a callout at the top called "current best" with: run name, model, mode, settings, combined score, date. leave placeholders; we update it by hand.
   - a short "rules" toggle listing: one change per run; new run name every run; record before moving on; keep only if ≥ +0.003 on all rows; never submit to zindi without a full-set win.
   - the database below, with its views embedded.
   - a "glossary" toggle explaining in one line each: held-out, zero-shot, few-shot, k, answer cap, num_beams, precision, refusal rate, rouge-1, rouge-l, combined, lora, checkpoint.

2. a database called "runs" with these properties (exact names, lowercase):
   - run name (title). pattern exp<number>_<model>_<mode>[_k<k>][_c<cap>][_b<beams>]_n<rows>, e.g. exp014_qwen_fs_k2_n500
   - exp id (text), e.g. exp014
   - date (date)
   - run by (person)
   - status (select): planned, running, done, failed, cancelled
   - stage (select): 1 dry run, 2 gpu smoke, 3 zero-shot screen, 4 few-shot screen, 5 confirm full, 6 tune few-shot, 7 beams, 8 fine-tune, 9 submission
   - hypothesis (text): what we expect and why, one sentence
   - changed setting (select): model, mode, k, answer cap, num_beams, batch size, precision, lora adapter, rows, none
   - changed from → to (text), e.g. k 2 → 3
   - compared against (relation to runs): the run this one is judged against
   - model (select): afrique (McGill-NLP/AfriqueLlama-8B), qwen (Qwen/Qwen2.5-7B-Instruct), gemma (google/gemma-2-9b-it), llama (meta-llama/Llama-3.1-8B-Instruct), aya (CohereLabs/aya-expanse-8b), other
   - mode (select): dry_run, zero_shot, few_shot
   - precision (select): auto, bf16, 4bit
   - k (number), answer cap (number), num_beams (number), batch size (number)
   - rows (select): 200, 500, all
   - lora adapter (text)
   - combined (number, 4 decimals)
   - rouge-1 (number, 4 decimals)
   - rouge-l (number, 4 decimals)
   - one number property per subset, 4 decimals: aka_gha, amh_eth, eng_eth, eng_gha, eng_ken, eng_uga, lug_uga, swa_ken (the "combined_no_judge" column of the notebook's per-subset table)
   - refusal rate (number, percent)
   - seconds (number): eval time from the notebook
   - current best combined (number): the current best's full-set score at the time of this run
   - delta vs best (formula): combined minus current best combined, 4 decimals, shown with a + or − sign
   - verdict (formula): if status is not done → "—"; if rows is not all → "screen only"; if delta vs best ≥ 0.003 → "keep"; otherwise "drop"
   - beats retrieval (formula): combined ≥ 0.37 → "yes", otherwise "no"
   - decision (select): keep, drop, rerun, investigate. we set this by hand; the verdict formula is a suggestion
   - public lb score (number, 4 decimals): zindi leaderboard score, only for submitted runs
   - greppable report (text): the raw block copied from the notebook
   - observations (text): what we saw in the sample predictions
   - error (text): full error text if failed

3. a page template for new runs in "runs", with these headings in the body:
   - hypothesis
   - what changed (one setting only)
   - settings snapshot (a small table of every setting in the notebook's config cell)
   - results (paste the greppable report in a code block)
   - per-subset notes (which languages moved up or down, and by how much)
   - sample predictions (paste 3–5 examples: question, reference, prediction, and what is wrong or right)
   - decision and why
   - next run this suggests

4. views of "runs":
   - "leaderboard": table, filter status = done and rows = all, sort combined descending. show run name, model, mode, k, answer cap, num_beams, combined, delta vs best, verdict, decision, date.
   - "screening": table, filter rows = 200 or 500, group by stage, sort combined descending.
   - "per language": table, status = done, rows = all, showing run name, model, mode and the 8 subset columns, sort combined descending. conditional colour on each subset column if notion supports it (low red, high green).
   - "queue": board grouped by status, sorted by exp id.
   - "by model": table grouped by model, sorted by combined descending.
   - "failures": table, filter status = failed, show run name, error, run by, date.
   - "submitted": table, filter public lb score is not empty, showing combined vs public lb score.

5. pre-fill "runs" with the planned runs below, status = planned, with stage, model, mode, k, answer cap (150), num_beams (1), batch size (32), precision (auto) and rows filled in:
   - dry_01: stage 1, dry_run, rows all
   - exp007_afrique_zs_n200: stage 2, afrique, zero_shot, rows 200
   - exp008_afrique_zs_n500, exp009_qwen_zs_n500, exp010_gemma_zs_n500, exp011_llama_zs_n500, exp012_aya_zs_n500: stage 3, zero_shot, rows 500
   - exp013_afrique_fs_k2_n500, exp014_qwen_fs_k2_n500, exp015_gemma_fs_k2_n500, exp016_llama_fs_k2_n500, exp017_aya_fs_k2_n500: stage 4, few_shot, k 2, rows 500
   - exp018 and exp019: stage 5, rows all, model and mode "to decide" (top two from stages 3–4), titles exp018_tbd_nall and exp019_tbd_nall
   - exp020_tbd_fs_k1_nall, exp021_tbd_fs_k3_nall, exp022_tbd_fs_k4_nall: stage 6
   - exp023_tbd_fs_c80_nall, exp024_tbd_fs_c300_nall: stage 6
   - exp025_tbd_b3_nall, exp026_tbd_b5_nall: stage 7
   add the hypothesis for each in one sentence (e.g. "few-shot examples teach answer length and style, so few_shot should beat zero_shot on rouge").
   in the llama rows, write in observations: "blocked until hugging face licence access is approved". in the aya rows: "reference only, can't be a final pick (non-commercial licence)".

6. a second, small database called "insights" with: insight (title), evidence (relation to runs), confidence (select: low, medium, high), date. a finding goes here once two or more runs support it, e.g. "few-shot helps amharic more than english".

keep the design plain and minimal: no emojis, no cover images, no decorative icons. consistent lowercase. make every number column right-aligned with 4 decimals so runs are easy to compare.
```
