"""HF Trainer wrapper with the knobs our config exposes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from transformers import TrainingArguments


def build_training_args(cfg: dict[str, Any], output_dir: str | Path) -> TrainingArguments:
    """Map configs/training/*.yaml's ``training`` block to TrainingArguments."""
    t = cfg["training"]
    return TrainingArguments(
        output_dir=str(output_dir),
        run_name=cfg.get("run_name", "afro-health-qa"),
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_eval_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        gradient_checkpointing=t.get("gradient_checkpointing", True),
        learning_rate=float(t["learning_rate"]),
        lr_scheduler_type=t.get("lr_scheduler_type", "cosine"),
        warmup_ratio=t.get("warmup_ratio", 0.03),
        weight_decay=t.get("weight_decay", 0.0),
        max_grad_norm=t.get("max_grad_norm", 0.3),
        optim=t.get("optim", "paged_adamw_8bit"),
        bf16=t.get("bf16", True),
        fp16=t.get("fp16", False),
        logging_steps=t.get("logging_steps", 10),
        eval_strategy=t.get("eval_strategy", "steps"),
        eval_steps=t.get("eval_steps", 100),
        save_strategy=t.get("save_strategy", "steps"),
        save_steps=t.get("save_steps", 200),
        save_total_limit=t.get("save_total_limit", 2),
        load_best_model_at_end=t.get("load_best_model_at_end", True),
        metric_for_best_model=t.get("metric_for_best_model", "eval_loss"),
        greater_is_better=t.get("greater_is_better", False),
        report_to=t.get("report_to", "wandb"),
        seed=int(t.get("seed", 42)),
        dataloader_num_workers=t.get("dataloader_num_workers", 2),
        remove_unused_columns=t.get("remove_unused_columns", False),
    )


def format_for_sft(example: dict, tokenizer, max_length: int = 1024) -> dict:
    """Tokenise a single training example with the prompt+answer template."""
    from afro_health_qa.models.prompts import build_prompt

    prompt = build_prompt(example["Question"], example["Language"], answer=example["Response"])
    ids = tokenizer(
        prompt + tokenizer.eos_token,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )
    ids["labels"] = ids["input_ids"].copy()
    return ids
