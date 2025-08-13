import os
import sys
from pathlib import Path
from typing import List

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError, model_validator
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.utils import logging as hf_logging
from loguru import logger


# Config & Logger
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / "src"))

hf_logging.set_verbosity_error()

MODEL_DIR = os.environ.get("MODEL_DIR", "runs/distilbert_auto/final_model")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD = float(os.environ.get("THRESHOLD", "0.5"))

# Request limits (can be tweaked via env vars)
MAX_BATCH = int(os.environ.get("MAX_BATCH", "64"))
MAX_CHARS = int(os.environ.get("MAX_CHARS", "5000"))
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "500"))  # tokenizer max_length


logger.info("Starting IMDB Sentiment API…")


try:
    logger.info(f"Loading tokenizer and model from: {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)
    model.eval()
    logger.success("Model loaded successfully.")
except Exception as e:
    logger.exception(f"Failed to load model from {MODEL_DIR}: {e}")
    # If model fails to load at startup, raise and stop the app from serving
    raise RuntimeError(f"Model load error: {e}") from e


app = FastAPI(title="IMDB Sentiment API", version="1.0.0")


class PredictIn(BaseModel):
    texts: List[str] = Field(..., description="List of raw review texts")

    @model_validator(mode="after")
    def _validate_batch(self):
        # Basic type and emptiness checks
        if not isinstance(self.texts, list):
            raise ValueError("`texts` must be a list of strings.")
        if len(self.texts) == 0:
            raise ValueError("`texts` is empty. Provide at least one text.")
        if len(self.texts) > MAX_BATCH:
            raise ValueError(f"Batch too large. Max allowed: {MAX_BATCH}.")

        # Strip & length limits; also ensure all items are non-empty strings
        cleaned = []
        for i, t in enumerate(self.texts):
            if not isinstance(t, str):
                raise ValueError(f"Item at index {i} is not a string.")
            s = t.strip()
            if len(s) == 0:
                raise ValueError(f"Item at index {i} is empty after trimming.")
            if len(s) > MAX_CHARS:
                raise ValueError(
                    f"Item at index {i} exceeds {MAX_CHARS} characters. "
                    "Shorten the input or increase MAX_CHARS."
                )
            cleaned.append(s)
        self.texts = cleaned
        return self


class ItemResult(BaseModel):
    label: int
    sentiment: str
    score_pos: float
    score_neg: float


class PredictOut(BaseModel):
    results: List[ItemResult]


@app.get("/health")
def health():
    logger.info("Running health check with dummy review...")
    dummy_review = ["This is a test review. The movie was fantastic and I enjoyed every moment!"]

    try:
        # Run quick prediction
        _ = _predict_batch(dummy_review, MAX_LENGTH, THRESHOLD)
        logger.success("Health check passed.")
        return {
            "status": "ok",
            "device": DEVICE,
            "model_dir": MODEL_DIR,
            "max_batch": MAX_BATCH,
            "max_chars": MAX_CHARS,
            "max_length": MAX_LENGTH,
            "threshold": THRESHOLD,
            "model_test": "passed"
        }
    except Exception as e:
        logger.exception(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed. Model is not responding.")



def _predict_batch(texts: List[str], max_length: int, threshold: float) -> List[ItemResult]:
    # Tokenize
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    # Forward pass
    with torch.no_grad():
        logits = model(**enc).logits  # shape (B, 2)
        probs = torch.softmax(logits, dim=-1)

    pos = probs[:, 1].cpu().numpy()
    neg = probs[:, 0].cpu().numpy()
    th = threshold if threshold is not None else THRESHOLD
    labels = (pos >= th).astype(int)
    sentiments = ["positive" if l == 1 else "negative" for l in labels]

    return [
        ItemResult(label=int(l), sentiment=s, score_pos=float(p), score_neg=float(n))
        for l, s, p, n in zip(labels, sentiments, pos, neg)
    ]

@app.post("/predict", response_model=PredictOut)
def predict(payload: PredictIn):
    logger.info(f"/predict called with batch={len(payload.texts)}")
    try:
        results = _predict_batch(payload.texts, MAX_LENGTH, THRESHOLD)
        logger.info("Inference successful.")
        return PredictOut(results=results)
    except ValidationError as ve:
        logger.warning(f"Validation error: {ve}")
        raise HTTPException(status_code=422, detail=str(ve)) from ve
    except torch.cuda.OutOfMemoryError:
        logger.exception("CUDA OOM during inference.")
        raise HTTPException(
            status_code=503,
            detail="Inference failed due to insufficient GPU memory. Try a smaller batch.",
        )
    except Exception as e:
        logger.exception(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail="Model inference failed.") from e
