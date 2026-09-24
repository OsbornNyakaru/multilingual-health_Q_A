<h1 align="center">AfroHealth QA — Mac Local Edition</h1>

<p align="center">
  <b>Blazing fast, native Apple Silicon inference for the AfroHealth QA generation pipeline.</b>
</p>

<p align="center">
  <a href="#why">Why?</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#memory-management">Memory Management</a>
</p>

---

## Why?

Running large language models sequentially on free cloud tiers (like Google Colab T4) is bottlenecked by PyTorch's heavy CUDA memory fragmentation, restrictive 4-bit `bitsandbytes` compute overhead, and strict session timeouts. 

This standalone execution module abandons PyTorch in favor of **Apple MLX** (`mlx-lm`). By leveraging the native Unified Memory architecture of M-Series chips (M1/M2/M3/M4), this pipeline can generate predictions for all 2,600+ test records sequentially in minutes rather than hours, completely offline.

## Architecture

* **Model**: `McGill-NLP/AfriqueLlama-8B` (A state-of-the-art fine-tune of Llama-3 8B for Swahili, Luganda, Akan, and Amharic).
* **Framework**: Apple `mlx-lm` (Replaces `transformers` and `accelerate`).
* **Prompting Strategy**: Dynamic Few-Shot Extraction. The engine automatically extracts perfect, concise doctor responses from `Train.csv` and injects them into the strict Llama-3 `<|eot_id|>` chat template.
* **Decoding Strategy**: Pure Greedy Decoding (`do_sample=False`). Eliminates medical hallucinations and bypasses the severe latency of repetition penalties.

## Quick Start

### 1. Prepare the Data
Ensure your raw Zindi CSV files are placed in the local data directory:
```text
mac_local_execution/
└── data/
    └── raw/
        ├── Train.csv
        ├── Test.csv
        └── SampleSubmission.csv
```

### 2. Install Dependencies
You only need Apple's MLX library and standard data processing tools. You do **not** need PyTorch.
```bash
pip install mlx-lm pandas tqdm jupyter
```

### 3. Run the Pipeline
Launch the Jupyter Notebook and execute the cells sequentially:
```bash
jupyter notebook exp002_mlx_macbook.ipynb
```
The engine will cache the model weights on its first run. Final predictions and iterative checkpoints will automatically compile into `data/submissions/`.

## Memory Management (Crucial)

By default, MLX loads the Hugging Face model in **16-bit precision**, requiring ~16GB of Unified Memory. 

**If your Mac has 24GB+ of RAM:**
Run the notebook exactly as-is. Enjoy maximum native throughput.

**If your Mac has 16GB (or less) of RAM:**
You must compress the model into 4-bit before running the notebook to prevent severe memory swapping. Run this command in your terminal:
```bash
mlx_lm.convert --hf-path McGill-NLP/AfriqueLlama-8B -q --q-bits 4
```
This will create a local folder named `mlx_model`. Open the Jupyter notebook and change the model path from `"McGill-NLP/AfriqueLlama-8B"` to `"mlx_model"`.

---
*If you are using Claude Code or another Agentic AI to help develop this codebase further, pass it the contents of `prompt.md` to establish the correct architectural context.*
