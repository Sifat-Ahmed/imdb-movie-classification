from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class ModelEval:
    name: str
    y_true: np.ndarray
    y_prob: np.ndarray
    thresholds: List[float]
    by_threshold: Dict[float, Dict[str, float]]
    roc_auc: float
    pr_auc: float
    roc_curve: Tuple[np.ndarray, np.ndarray, np.ndarray]  # fpr, tpr, thr
    pr_curve: Tuple[np.ndarray, np.ndarray, np.ndarray]   # precision, recall, thr
