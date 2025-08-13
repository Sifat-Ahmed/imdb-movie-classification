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
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)
import json
import matplotlib.pyplot as plt

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

def parse_args():
    ap = argparse.ArgumentParser("Train DistilBERT from a YAML config (with 80/10/10 auto-split).")
    ap.add_argument("-c", "--config", required=True, help="Path to YAML config.")
    return ap.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _save_curve_png(x, y, xlabel, ylabel, title, out_path):
    plt.figure()
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _make_roc_grid(y_true, probs, thresholds, auc_val, out_path):
    """2x2 grid: each subplot shows full ROC curve + operating point at a given threshold."""
    fpr, tpr, _ = roc_curve(y_true, probs)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()

    for ax, th in zip(axes, thresholds):
        y_pred = (probs >= th).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        # FPR = FP / (FP + TN); TPR = TP / (TP + FN)
        tn, fp, fn, tp = cm.ravel()
        fpr_pt = fp / (fp + tn + 1e-12)
        tpr_pt = tp / (tp + fn + 1e-12)

        ax.plot(fpr, tpr)
        ax.plot([0, 1], [0, 1], linestyle="--", alpha=0.3)
        ax.scatter([fpr_pt], [tpr_pt], marker="o")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC @ t={th:.2f} (AUC={auc_val:.3f})")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def _make_pr_grid(y_true, probs, thresholds, ap_val, out_path):
    """2x2 grid: each subplot shows full PR curve + operating point at a given threshold."""
    precision, recall, _ = precision_recall_curve(y_true, probs)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()

    for ax, th in zip(axes, thresholds):
        y_pred = (probs >= th).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        # Precision = TP/(TP+FP);
        # Recall = TP/(TP+FN)
        prec_pt = tp / (tp + fp + 1e-12)
        rec_pt = tp / (tp + fn + 1e-12)

        ax.plot(recall, precision)
        ax.scatter([rec_pt], [prec_pt], marker="o")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"PR @ t={th:.2f} (AP={ap_val:.3f})")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def _make_cm_grid(y_true, probs, thresholds, out_path):
    """2x2 grid of confusion matrices for the given thresholds, with counts annotated."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()

    for ax, th in zip(axes, thresholds):
        y_pred = (probs >= th).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"Confusion Matrix @ t={th:.2f}")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"])
        ax.set_yticklabels(["True 0", "True 1"])

        # annotate counts
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


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

    #  Read CSV with pandas
    df = pd.read_csv(csv_path)

    # Rename to 'text'/'label' if needed
    rename_map = {}
    if text_col in df.columns and text_col != "text":
        rename_map[text_col] = "text"
    if label_col in df.columns and label_col != "label":
        rename_map[label_col] = "label"
    if rename_map:
        df = df.rename(columns=rename_map)

    # Basic check
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain 'text' and 'label' columns (or configure text_col/label_col).")

    # Clean and map labels
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

    # Stratified 80/20, then 12.5% of 80% -> 10% overall for validation
    X = df["text"].values
    y = df["label"].values

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.125, random_state=seed, stratify=y_train_full
    )

    # Build HF datasets from pandas DataFrames
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

    #Data
    logger.info("Building datasets (with stratified 80/10/10 split)…")
    ds = build_datasets(cfg)

    #Tokenizer & tokenization
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tok_fn = tokenize_builder(tokenizer, max_len)
    ds = ds.map(tok_fn, batched=True)
    ds = ds.map(lambda ex: {"labels": ex["label"]})
    keep_cols = {"input_ids", "attention_mask", "labels"}
    ds = DatasetDict({k: v.remove_columns([c for c in v.column_names if c not in keep_cols]) for k, v in ds.items()})

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # 4) Training args
    args_hf = TrainingArguments(
        output_dir=str(out_dir),
        learning_rate=lr,
        per_device_train_batch_size=bsz,
        per_device_eval_batch_size=bsz,
        num_train_epochs=epochs,
        weight_decay=wd,

        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        save_total_limit=cfg.get("save_total_limit", 2),

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

    # --- Analysis on test set ---
    logger.info("Running ROC/PR analysis and threshold sweep on the test set…")
    pred = trainer.predict(ds["test"])
    logits = pred.predictions  # shape (N, 2)
    y_true = pred.label_ids.astype(int)

    # Convert logits to probabilities for class 1 (positive)
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]

    # Directory for artifacts
    save_dir = Path(cfg.get("output_dir", "runs/distilbert")) / "final_model"
    save_dir.mkdir(parents=True, exist_ok=True)

    # AUCs
    roc_auc = roc_auc_score(y_true, probs)
    pr_auc = average_precision_score(y_true, probs)
    with open(save_dir / "auc_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"roc_auc": float(roc_auc), "pr_auc": float(pr_auc)}, f, indent=2)


    fpr, tpr, _ = roc_curve(y_true, probs)
    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
    roc_df.to_csv(save_dir / "roc_curve_points.csv", index=False)
    _save_curve_png(
        fpr, tpr, xlabel="False Positive Rate", ylabel="True Positive Rate",
        title=f"ROC Curve (AUC={roc_auc:.3f})",
        out_path=save_dir / "roc_curve.png"
    )

    precision, recall, _ = precision_recall_curve(y_true, probs)
    pr_df = pd.DataFrame({"precision": precision, "recall": recall})
    pr_df.to_csv(save_dir / "pr_curve_points.csv", index=False)
    _save_curve_png(
        recall, precision, xlabel="Recall", ylabel="Precision",
        title=f"Precision–Recall Curve (AP={pr_auc:.3f})",
        out_path=save_dir / "pr_curve.png"
    )

    # Threshold sweep and confusion matrices
    thresholds = np.array([0.50, 0.60, 0.70, 0.80])
    rows = []
    for th in thresholds:
        y_pred = (probs >= th).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        acc = (tp + tn) / (tp + tn + fp + fn + 1e-12)
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)

        cm_path = save_dir / f"confusion_matrix_t{str(th).replace('.', '_')}.csv"
        pd.DataFrame(cm, index=["true_neg", "true_pos"], columns=["pred_neg", "pred_pos"]).to_csv(cm_path)

        rows.append({
            "threshold": float(th),
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        })

    th_df = pd.DataFrame(rows)
    th_df.to_csv(save_dir / "threshold_sweep_0p50_0p80.csv", index=False)

    # Quick F1 vs threshold line
    _save_curve_png(
        th_df["threshold"].values, th_df["f1"].values,
        xlabel="Threshold", ylabel="F1",
        title="F1 vs Threshold (Test Set)",
        out_path=save_dir / "f1_vs_threshold.png"
    )


    _make_roc_grid(y_true, probs, thresholds, roc_auc, save_dir / "roc_grid.png")

    _make_pr_grid(y_true, probs, thresholds, pr_auc, save_dir / "pr_grid.png")

    _make_cm_grid(y_true, probs, thresholds, save_dir / "cm_grid.png")

    logger.success(f"Saved AUC metrics, curves, grids, threshold sweep, and confusion matrices to: {save_dir}")

    # Save final artifacts (model + tokenizer)
    final_dir = out_dir / "final_model"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    logger.success(f"Saved model + tokenizer to: {final_dir}")


if __name__ == "__main__":
    main()
