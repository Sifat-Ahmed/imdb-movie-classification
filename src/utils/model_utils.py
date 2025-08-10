from __future__ import annotations

from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score, roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
)

from utils.model_dataclass import ModelEval


def evaluate_model(model: torch.nn.Module,
                   name: str,
                   loader: torch.utils.data.DataLoader,
                   device: torch.device,
                   thresholds: List[float]) -> ModelEval:
    """Run full evaluation for a single model on a loader."""
    y_true, y_prob = predict_probs(model, loader, device)
    by_t = eval_at_thresholds(y_true, y_prob, thresholds)

    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    fpr, tpr, thr_roc = roc_curve(y_true, y_prob)
    prec, rec, thr_pr = precision_recall_curve(y_true, y_prob)

    return ModelEval(
        name=name,
        y_true=y_true,
        y_prob=y_prob,
        thresholds=thresholds,
        by_threshold=by_t,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        roc_curve=(fpr, tpr, thr_roc),
        pr_curve=(prec, rec, thr_pr),
    )


@torch.no_grad()
def predict_probs(model: torch.nn.Module,
                  loader: torch.utils.data.DataLoader,
                  device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """Return (y_true, y_prob) for a model on a given loader.
    y_prob are sigmoid(logits) in [0,1].
    """
    model.eval()
    ys, ps = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        prob = torch.sigmoid(logits).detach().cpu().numpy()
        ys.append(yb.numpy())
        ps.append(prob)
    y_true = np.concatenate(ys).astype(int)
    y_prob = np.concatenate(ps).astype(float)
    return y_true, y_prob


def eval_at_thresholds(y_true: np.ndarray,
                       y_prob: np.ndarray,
                       thresholds: List[float]) -> Dict[float, Dict[str, float]]:
    """Compute metrics for a list of thresholds.
    Returns a mapping threshold -> metrics dict including confusion matrix.
    """
    out: Dict[float, Dict[str, float]] = {}
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        out[t] = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "cm": confusion_matrix(y_true, y_pred, labels=[0, 1]),
        }
    return out


def plot_cm_quadrant(model_eval: ModelEval) -> None:
    fig, axes = plt.subplots(2, 2)
    axes = axes.ravel()

    for ax, t in zip(axes, model_eval.thresholds):
        m = model_eval.by_threshold[t]
        cm = m["cm"].astype(float)

        hm = sns.heatmap(
            cm,
            annot=False,
            cmap="Blues",
            xticklabels=["neg", "pos"],
            yticklabels=["neg", "pos"],
            cbar=False,
            square=True,
            linewidths=0.1,
            linecolor="white",
            ax=ax,
        )

        vmax = cm.max() if cm.size else 1.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = int(cm[i, j])
                # choose white text if background is dark
                frac = (cm[i, j] / vmax) if vmax else 0.0
                color = "white" if frac > 0.5 else "black"
                ax.text(j + 0.5, i + 0.5, f"{val}", ha="center", va="center",
                        color=color, fontsize=6)

        ax.set_title(
            f"{model_eval.name} @ t={t:.2f}\n"
            f"Acc={m['accuracy']:.3f} | F1={m['f1']:.3f} | P={m['precision']:.3f} | R={m['recall']:.3f}", fontsize=6
        )
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")

    fig.suptitle(f"{model_eval.name}: Confusion Matrices at 4 Thresholds",
                 y=1.02, fontsize=14)
    plt.tight_layout()
    plt.show()



def plot_model_rocs(evals: List[ModelEval]) -> None:
    plt.figure(figsize=(8, 6))
    for ev in evals:
        fpr, tpr, _ = ev.roc_curve
        plt.plot(fpr, tpr, label=f"{ev.name} (AUC={ev.roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_model_prs(evals: List[ModelEval]) -> None:
    plt.figure(figsize=(8, 6))
    for ev in evals:
        prec, rec, _ = ev.pr_curve
        plt.plot(rec, prec, label=f"{ev.name} (AP={ev.pr_auc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curves")
    plt.legend()
    plt.tight_layout()
    plt.show()


def show_threshold_points_on_roc(ev: ModelEval) -> None:
    """Overlay the 4 threshold operating points on the ROC curve for a model."""
    fpr, tpr, thr = ev.roc_curve
    pts = []
    # Note: roc_curve thresholds are sorted from high->low and exclude 0 and 1 at ends sometimes
    for t in ev.thresholds:
        if len(thr) == 0:
            continue
        idx = int(np.abs(thr - t).argmin())
        if 0 <= idx < len(fpr):
            pts.append((fpr[idx], tpr[idx], t))

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"{ev.name} (AUC={ev.roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", alpha=0.5)
    for x, y, t in pts:
        plt.scatter([x], [y], s=30)
        plt.text(x + 0.02, y, f"t={t:.2f}", fontsize=9)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC with Threshold Points — {ev.name}")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_f1_vs_threshold(evals: List[ModelEval]) -> None:
    """Plot F1 as a function of threshold for each model on one chart."""
    plt.figure(figsize=(8, 6))
    for ev in evals:
        ts = sorted(ev.by_threshold.keys())
        f1s = [ev.by_threshold[t]["f1"] for t in ts]
        plt.plot(ts, f1s, marker="o", label=ev.name)
    plt.xlabel("Threshold")
    plt.ylabel("F1 score")
    plt.title("F1 vs Threshold")
    plt.xticks(ts)
    plt.legend()
    plt.tight_layout()
    plt.show()


def make_threshold_table(ev: ModelEval) -> pd.DataFrame:
    rows = []
    for t, m in ev.by_threshold.items():
        rows.append(
            {
                "model": ev.name,
                "threshold": t,
                "accuracy": m["accuracy"],
                "f1": m["f1"],
                "precision": m["precision"],
                "recall": m["recall"],
            }
        )
    df = pd.DataFrame(rows).sort_values(["model", "threshold"]).reset_index(drop=True)
    return df


def summarize_models(evals: List[ModelEval]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    per_threshold = pd.concat([make_threshold_table(ev) for ev in evals], ignore_index=True)

    best_rows = []
    for ev in evals:
        # choose best threshold by F1; change to accuracy if needed
        t_best = max(ev.by_threshold, key=lambda t: ev.by_threshold[t]["f1"]) if len(ev.by_threshold) else None
        if t_best is not None:
            m = ev.by_threshold[t_best]
            best_rows.append(
                {
                    "model": ev.name,
                    "best_threshold": t_best,
                    "best_acc": m["accuracy"],
                    "best_f1": m["f1"],
                    "roc_auc": ev.roc_auc,
                    "pr_auc": ev.pr_auc,
                }
            )
    leaderboard = (
        pd.DataFrame(best_rows)
        .sort_values(["best_f1", "best_acc", "roc_auc"], ascending=False)
        .reset_index(drop=True)
    )
    return per_threshold, leaderboard
