---
tags: [platform, molab]
source: https://docs.marimo.io/guides/molab/
---
# Molab Platform

Free cloud-hosted marimo notebooks: https://molab.marimo.io/notebooks

## Compute (as documented 2026-09-18)
- Default: 4 CPUs, 32 GB RAM.
- GPU option: NVIDIA RTX Pro 6000 Blackwell, **96 GB VRAM**, 125 TFLOPS. Enough for an 8B model in bf16 with large batches, or LoRA on 8B–27B.
- **Session cap 12 h; idle > 90 min = shutdown.** Checkpoint everything (see [[Molab Notebook Plan]]).
- Free.

## Packages
- torch, numpy, polars pre-installed. marimo's package manager auto-installs on import.
- We also declare deps as PEP 723 inline script metadata at the top of the notebook file so `uv run` / molab resolve the same set.

## Storage — the important rule
- "Only files uploaded through the sidebar, or cached with `mo.persistent_cache`, are persisted." Everything else in the working dir is ephemeral per session.
- Remote storage integrations: Google Drive, Hugging Face, S3-compatible buckets.
- Practical consequence: upload the 4 CSVs (~24 MB) once via the sidebar into a `data/` folder next to the notebook, and push submissions/adapters to the Hugging Face Hub (or download them) before the session dies. See [[File Organisation]].

## Getting a notebook in / out
- New notebook dropdown → paste a GitHub URL to a `.py` marimo notebook → becomes a **synced notebook with GitHub as source of truth**. This is how we deploy: push `notebooks/molab_afro_health_qa.py` to GitHub, import by URL.
- Download: download button, or `marimo edit https://molab.marimo.io/notebooks/<ID>` locally.
- Sharing: static/WASM previews, embed, "open in molab" badge.

## Agents
- Notebook actions menu → **"Pair with an agent"** gives the connection instructions for Claude Code. See [[Marimo MCP and Pairing]].

Related: [[Molab Notebook Plan]], [[File Organisation]].
