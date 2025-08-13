# train.py
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import argparse
from pathlib import Path
import yaml, sys
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from loguru import logger


project_root = Path().resolve().parent
sys.path.append(str(project_root / "src"))

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# Config
def parse_args():
    ap = argparse.ArgumentParser("Train DistilBERT from a YAML config (with 80/10/10 auto-split).")
    ap.add_argument("-c", "--config", required=True, help="Path to YAML config.")
    return ap.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Data
def build_datasets(cfg):
    """
    Load ONE CSV via pandas and create a stratified 80/10/10 split.
    Assumes a text column and a label column. String labels like 'pos'/'neg' are mapped to 1/0.
    Returns: DatasetDict with 'train'/'validation'/'test' (Hugging Face datasets)
    """
    csv_path = cfg["dataset_csv"]  # required
    seed = int(cfg.get("seed", 42))
    text_col = cfg.get("text_col", "text")
    label_col = cfg.get("label_col", "label")

    # 1) Read CSV with pandas
    df = pd.read_csv(csv_path)

    # 2) Rename to 'text'/'label' if needed
    rename_map = {}
    if text_col in df.columns and text_col != "text":
        rename_map[text_col] = "text"
    if label_col in df.columns and label_col != "label":
        rename_map[label_col] = "label"
    if rename_map:
        df = df.rename(columns=rename_map)

    # Basic sanity check
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain 'text' and 'label' columns (or configure text_col/label_col).")

    # 3) Clean and map labels
    #    - if strings: map pos/positive -> 1, neg/negative -> 0
    #    - if numeric: force to int, and validate values are 0/1
    if df["label"].dtype == object:
        lab = (
            df["label"].astype(str).str.strip().str.lower()
            .replace({"pos": 1, "positive": 1, "neg": 0, "negative": 0})
        )
        if not set(lab.unique()).issubset({0, 1}):
            raise ValueError("String labels must be one of {pos, positive, neg, negative}.")
        df["label"] = lab.astype(int)
    else:
        df["label"] = df["label"].astype(int)
        bad = set(df["label"].unique()) - {0, 1}
        if bad:
            raise ValueError(f"Numeric labels must be 0/1; got unexpected values: {sorted(bad)}")

    # Drop NA and empty texts
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    df = df[df["text"].astype(str).str.strip().ne("")].reset_index(drop=True)

    # 4) Stratified 80/20, then 12.5% of 80% -> 10% overall for validation
    X = df["text"].values
    y = df["label"].values

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    # 12.5% of 80% = 10% overall
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.125, random_state=seed, stratify=y_train_full
    )

    # 5) Build HF datasets from pandas DataFrames
    train_df = pd.DataFrame({"text": X_train, "label": y_train})
    val_df = pd.DataFrame({"text": X_val, "label": y_val})
    test_df = pd.DataFrame({"text": X_test, "label": y_test})

    ds_train = Dataset.from_pandas(train_df, preserve_index=False)
    ds_val = Dataset.from_pandas(val_df, preserve_index=False)
    ds_test = Dataset.from_pandas(test_df, preserve_index=False)

    return DatasetDict({"train": ds_train, "validation": ds_val, "test": ds_test})


def tokenize_builder(tokenizer, max_length: int):
    def tok(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding=False,  # dynamic padding via collator
            max_length=max_length,
        )

    return tok


def compute_metrics(eval_pred):
    logits, y_true = eval_pred
    y_pred = np.argmax(logits, axis=1)
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}


# -----------------------
# Main
# -----------------------
def main():
    args = parse_args()
    cfg = load_config(args.config)

    logger.success(f"Config Loaded {cfg}")

    out_dir = Path(cfg.get("output_dir", "runs/distilbert"))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    model_name = cfg.get("model_name", "distilbert-base-uncased")
    lr = float(cfg.get("lr", 2e-5))
    bsz = int(cfg.get("bsz", 16))
    epochs = float(cfg.get("epochs", 3))
    wd = float(cfg.get("wd", 0.01))
    max_len = int(cfg.get("max_length", 256))
    seed = int(cfg.get("seed", 42))

    eval_strategy = cfg.get("eval_strategy", "epoch")
    save_strategy = cfg.get("save_strategy", "no")
    disable_tqdm = bool(cfg.get("disable_tqdm", False))
    fp16 = bool(cfg.get("fp16", torch.cuda.is_available()))

    # Early stopping (optional)
    use_es = bool(cfg.get("early_stopping", False))
    es_patience = int(cfg.get("early_stopping_patience", 2))
    load_best = bool(cfg.get("load_best_model_at_end", use_es))
    metric_best = cfg.get("metric_for_best_model", "accuracy")

    # 1) Data
    logger.info("Building datasets (with stratified 80/10/10 split)…")
    ds = build_datasets(cfg)

    # 2) Tokenizer & tokenization
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tok_fn = tokenize_builder(tokenizer, max_len)
    ds = ds.map(tok_fn, batched=True)
    # rename label -> labels for HF Trainer
    ds = ds.map(lambda ex: {"labels": ex["label"]})
    # keep only necessary columns
    keep_cols = {"input_ids", "attention_mask", "labels"}
    ds = DatasetDict({k: v.remove_columns([c for c in v.column_names if c not in keep_cols]) for k, v in ds.items()})

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 3) Model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # 4) Training args
    args_hf = TrainingArguments(
        output_dir=out_dir,
        learning_rate=lr,
        per_device_train_batch_size=bsz,
        per_device_eval_batch_size=bsz,
        num_train_epochs=epochs,
        weight_decay=wd,

        # must match
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        save_total_limit=cfg.get("save_total_limit", 2),

        # pick metric to monitor
        load_best_model_at_end=load_best,
        metric_for_best_model=metric_best,
        greater_is_better=True,

        logging_steps=cfg.get("logging_steps", 50),
        report_to=[],
        fp16=fp16,
        disable_tqdm=disable_tqdm,
        logging_strategy="no"
    )

    callbacks = []
    if use_es:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=es_patience))

    # 5) Trainer
    trainer = Trainer(
        model=model,
        args=args_hf,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    # 6) Train / Evaluate / Test
    logger.info("Starting training…")
    trainer.train()
    logger.info("Evaluating on validation…")
    val_metrics = trainer.evaluate(eval_dataset=ds["validation"])
    for k, v in val_metrics.items():
        logger.info(f"val_{k}: {v:.6f}" if isinstance(v, (int, float)) else f"val_{k}: {v}")

    logger.info("Evaluating on held-out test…")
    test_metrics = trainer.evaluate(eval_dataset=ds["test"])
    for k, v in test_metrics.items():
        logger.info(f"test_{k}: {v:.6f}" if isinstance(v, (int, float)) else f"test_{k}: {v}")

    # Save final artifacts
    save_dir = out_dir / "final_model"
    trainer.save_model(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    logger.success(f"Saved model + tokenizer to: {save_dir}")


if __name__ == "__main__":
    main()
