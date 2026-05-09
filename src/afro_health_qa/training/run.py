"""Entry point for fine-tuning.

    python -m afro_health_qa.training.run --config configs/training/qlora_default.yaml

The run:
    1. Seeds all five RNGs (+ transformers.set_seed).
    2. Loads base + data configs referenced from the training config.
    3. Loads the data, splits stratified, tokenises.
    4. Loads the model in 4-bit, attaches LoRA, trains.
    5. Saves adapter to ``output_dir``.
    6. Writes a run-metadata JSON sidecar for the experiment log.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def _load_yaml(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="configs/training/<name>.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Parse configs, skip training.")
    args = parser.parse_args()

    # --- 1. Seed ---
    from afro_health_qa.seeding import set_all_seeds

    base_cfg = _load_yaml("configs/base.yaml")
    set_all_seeds(base_cfg.get("seed", 42))
    from transformers import set_seed

    set_seed(base_cfg.get("seed", 42))

    # --- 2. Load configs ---
    train_cfg = _load_yaml(args.config)
    model_cfg_path = train_cfg["model_config"]
    data_cfg = _load_yaml(train_cfg.get("data_config", "configs/data.yaml"))

    from afro_health_qa.models.registry import resolve_model_config

    model_cfg = resolve_model_config(model_cfg_path)

    run_name = train_cfg.get("run_name", "qlora_run")
    output_dir = Path(train_cfg.get("output_dir", f"models/{run_name}"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 3. Write run metadata sidecar up-front ---
    meta = {
        "run_name": run_name,
        "config": args.config,
        "model_hf_id": model_cfg.hf_id,
        "git_hash": _git_hash(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "seed": base_cfg.get("seed", 42),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if args.dry_run:
        print(f"[dry-run] loaded configs ok. output_dir={output_dir}")
        print(json.dumps(meta, indent=2))
        return 0

    # --- 4. Training ---
    # Kept as a top-level import so dry-run does not require torch.
    from afro_health_qa.data.load import load_competition_data, to_hf_dataset
    from afro_health_qa.data.split import SplitRatios, stratified_split
    from afro_health_qa.models.qlora import attach_adapters, spec_from_yaml
    from afro_health_qa.models.registry import load_model, load_tokenizer
    from afro_health_qa.training.callbacks import build_early_stopping
    from afro_health_qa.training.trainer import build_training_args, format_for_sft

    comp = load_competition_data(raw_dir=base_cfg["paths"]["data_raw"])
    ratios = SplitRatios(**data_cfg["split"]["ratios"])
    splits = stratified_split(
        comp.train,
        by=data_cfg["split"]["stratify_by"],
        ratios=ratios,
        seed=int(data_cfg["split"]["seed"]),
    )
    # Persist splits for reproducibility.
    proc_dir = Path(base_cfg["paths"]["data_processed"])
    proc_dir.mkdir(parents=True, exist_ok=True)
    splits.train.to_csv(proc_dir / "train.csv", index=False)
    splits.val.to_csv(proc_dir / "val.csv", index=False)
    splits.heldout.to_csv(proc_dir / "heldout.csv", index=False)

    tokenizer = load_tokenizer(model_cfg)
    model = load_model(model_cfg, for_training=True)
    lora_spec = spec_from_yaml(train_cfg, model_cfg)
    model = attach_adapters(model, lora_spec)

    max_len = int(train_cfg["sequence"]["max_length"])

    train_ds = to_hf_dataset(splits.train)
    val_ds = to_hf_dataset(splits.val)
    train_ds = train_ds.map(
        lambda ex: format_for_sft(ex, tokenizer, max_length=max_len),
        remove_columns=[c for c in train_ds.column_names if c not in ("Language",)],
    )
    val_ds = val_ds.map(
        lambda ex: format_for_sft(ex, tokenizer, max_length=max_len),
        remove_columns=[c for c in val_ds.column_names if c not in ("Language",)],
    )

    from transformers import DataCollatorForLanguageModeling, Trainer

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = build_training_args(train_cfg, output_dir=output_dir)

    callbacks = []
    es_cfg = train_cfg.get("early_stopping")
    if es_cfg:
        callbacks.append(build_early_stopping(es_cfg["patience"], es_cfg["threshold"]))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        callbacks=callbacks,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Done. Adapter saved to {output_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
