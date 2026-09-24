---
type: decision
id: D-006
created: 2026-09-24
status: confirmed
links: ["[[F-004-rules]]", "[[repo-state]]", "[[00_INDEX]]"]
---
# D-006 Competition data lives in a private HF dataset, not GitHub

**Decision:** The GitHub repo `OsbornNyakaru/multilingual-health_Q_A` is **public**, and Zindi's terms forbid redistributing the data. So:
- `.gitignore` excludes `data/multilingual_data/`, `autoresearch_nlp/data/` and `docs/competition_report/data/bge_retrieval_heldout.csv` (commit `c579a17`).
- `scripts/push_data_to_hf.py` uploads `data/raw/{Train,Val,Test,SampleSubmission}.csv` plus a SHA-256 `manifest.json` to a **private** dataset `nyakaruosborn/afro-health-qa-data`. It refuses to upload if the repo already exists and is public.
- On molab, set `HF_DATA_REPO=nyakaruosborn/afro-health-qa-data` and `HF_TOKEN=<read token>`. `notebooks/molab_afro_health_qa.py` then `snapshot_download`s `*.csv` into `data/raw/`.

**Status:** confirmed 2026-09-24. Uploaded to private `nyakaruosborn/afro-health-qa-data`; a round-trip `snapshot_download` matched all 4 SHA-256 hashes. On molab: `HF_DATA_REPO=nyakaruosborn/afro-health-qa-data`.

**Incident (2026-09-24):** A real HF token was hardcoded in `notebooks/uploaded_nb_code.py` and `notebooks/Copy_of_notebook26836f548f.ipynb`. GitHub push protection blocked the push; the token was replaced with `os.environ["HF_TOKEN"]` before anything reached the remote. That token must be revoked. Never hardcode tokens in notebooks.

**Alternatives rejected:** uploading via the molab sidebar each session (molab storage isn't durable); making the GitHub repo private (the repo is meant to be shareable).

## Links
- [[F-004-rules]]
- [[repo-state]]
- [[00_INDEX]]
