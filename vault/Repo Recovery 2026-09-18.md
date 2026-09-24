---
tags: [ops, git]
date: 2026-09-18
---
# Repo Recovery 2026-09-18

## Symptom
`git status` → `error: bad tree object HEAD`. `git fsck` showed the root tree `562e78ce…` and ~50 blobs missing from `.git/objects`. `.git` was 36 KB. Working tree was intact except the whole `src/afro_health_qa/` package below `__init__.py`/`seeding.py` and `configs/decoding/beam_rerank.yaml` were gone (their blobs were the missing ones).

Likely cause: the folder lives under `~/Documents` (iCloud Drive eligible); `du` reported 0 B for several directories with real files, which is the signature of dataless/offloaded files. The `.git/objects` loose files were probably evicted or never synced.

## Fix (non-destructive)
1. Confirmed remote `origin/main` = same commit `00d444b` and healthy (`git clone` into scratchpad, fsck clean).
2. `git fetch --refetch origin` — forces the server to resend every object; fsck clean afterwards. (Direct copying into `.git` was blocked by the auto-mode safety classifier; the git-native path worked.)
3. `git ls-files --deleted | xargs git restore --` — restored the 34 lost tracked files without touching the modified `pyproject.toml`.

## After
- `main` tracks `origin/main`, fsck clean.
- Only diff vs GitHub: `pyproject.toml` `requires-python = ">=3.11"` (local) vs `">=3.11,<3.12"` (remote).
- Lots of untracked local work (see [[Repo Layout]]); nothing on GitHub beyond the scaffold commit. **GitHub does not have "the latest"; the local tree does.**

Recommendation: keep the repo out of iCloud-synced paths, or set the folder to "always keep on this device".

Related: [[Session Log]].
