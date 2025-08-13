import os, sys
import torch
from pathlib import Path
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.utils import logging as hf_logging

project_root = Path().resolve().parent
sys.path.append(str(project_root / "src"))
hf_logging.set_verbosity_error()

MODEL_DIR = os.environ.get("MODEL_DIR", "runs/distilbert_auto/final_model")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD = float(os.environ.get("THRESHOLD", "0.5"))

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()

app = FastAPI(title="IMDB Sentiment API", version="1.0.0")


class PredictIn(BaseModel):
    texts: List[str] = Field(..., description="List of raw review texts")


class ItemResult(BaseModel):
    label: int
    sentiment: str
    score_pos: float
    score_neg: float


class PredictOut(BaseModel):
    results: List[ItemResult]


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "model_dir": MODEL_DIR}


def _predict_batch(texts: List[str], max_length: int, threshold: float) -> List[ItemResult]:
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)

    pos = probs[:, 1].cpu().numpy()
    neg = probs[:, 0].cpu().numpy()
    th = threshold if threshold is not None else THRESHOLD
    labels = (pos >= th).astype(int)
    sentiments = ["positive" if l == 1 else "negative" for l in labels]

    return [ItemResult(label=int(l), sentiment=str(s), score_pos=float(p), score_neg=float(n))
            for l, s, p, n in zip(labels, sentiments, pos, neg)]


@app.post("/predict", response_model=PredictOut)
def predict(payload: PredictIn):
    results = _predict_batch(payload.texts, 256, THRESHOLD)
    return PredictOut(results=results)
