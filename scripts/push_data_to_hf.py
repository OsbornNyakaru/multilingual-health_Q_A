"""Upload the Zindi competition CSVs to a PRIVATE Hugging Face dataset repo for molab.

The GitHub repo is public and Zindi data must not be redistributed, so molab pulls
the data from this private repo instead (see notebooks/molab_afro_health_qa.py,
which reads HF_DATA_REPO + HF_TOKEN and snapshot_downloads *.csv).

Usage (after `hf auth login` with a write token):
    python scripts/push_data_to_hf.py                     # -> <you>/afro-health-qa-data
    python scripts/push_data_to_hf.py --repo org/name
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
FILES = ("Train.csv", "Val.csv", "Test.csv", "SampleSubmission.csv")

CARD = """---
license: other
pretty_name: afro-health-qa competition data (private mirror)
---

Private mirror of the Zindi *Multilingual Health Question Answering in Low-Resource
African Languages* challenge data, for use from molab notebooks only.
Do **not** make this repo public: Zindi's terms forbid redistributing the data.

Files: {files}. SHA-256 in `manifest.json`.
"""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=None, help="dataset repo id (default: <whoami>/afro-health-qa-data)")
    args = ap.parse_args()

    missing = [f for f in FILES if not (RAW_DIR / f).exists()]
    if missing:
        print(f"Missing in {RAW_DIR}: {missing}", file=sys.stderr)
        return 1

    api = HfApi()
    repo_id = args.repo or f"{api.whoami()['name']}/afro-health-qa-data"

    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    if not api.repo_info(repo_id, repo_type="dataset").private:
        print(f"Refusing to upload: {repo_id} exists and is PUBLIC.", file=sys.stderr)
        return 1

    manifest = {f: sha256(RAW_DIR / f) for f in FILES}
    (RAW_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (RAW_DIR / "README.md").write_text(CARD.format(files=", ".join(FILES)), encoding="utf-8")

    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=RAW_DIR,
        allow_patterns=[*FILES, "manifest.json", "README.md"],
        commit_message="Upload competition CSVs",
    )
    remote = {s.rfilename for s in api.dataset_info(repo_id, files_metadata=False).siblings}
    absent = [f for f in FILES if f not in remote]
    print(f"repo: {repo_id} (private)  uploaded: {sorted(remote)}")
    if absent:
        print(f"MISSING after upload: {absent}", file=sys.stderr)
        return 1
    print(f"On molab set: HF_DATA_REPO={repo_id}  HF_TOKEN=<read token>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
