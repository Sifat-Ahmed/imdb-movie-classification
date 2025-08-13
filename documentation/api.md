# IMDB Sentiment Classification API Documentation

## Overview

This API provides real-time sentiment classification for movie reviews using a fine-tuned DistilBERT model. The service can process individual reviews or batches of reviews, returning sentiment predictions with confidence scores.

**Base URL**: `http://localhost:8000` (default)  
**Framework**: FastAPI  
**Model**: DistilBERT fine-tuned on IMDB dataset  
**Performance**: 91.6% F1-score on test set  

---

## Quick Start

### 1. Start the API Server
```bash
# Using uvicorn
uvicorn app:app --host 0.0.0.0 --port 8000

# Using Docker
docker run -p 8000:8000 imdb-sentiment-api
```

### 2. Test the API
```bash
# Health check
curl http://localhost:8000/health

# Sample prediction
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"texts": ["This movie was amazing!", "Terrible film, waste of time"]}'
```

---

## API Endpoints

### 1. Health Check

**Endpoint**: `GET /health`

**Description**: Check if the API service is running and perform a test prediction to verify model functionality.

**Example Request**:
```bash
curl -X GET "http://localhost:8000/health"
```

**Response**:
```json
{
  "status": "ok",
  "device": "cuda",
  "model_dir": "runs/distilbert_auto/final_model",
  "max_batch": 64,
  "max_chars": 5000,
  "max_length": 500,
  "threshold": 0.5,
  "model_test": "passed"
}
```

**Response Fields**:
- `status`: Service status ("ok" if healthy)
- `device`: Computing device ("cuda" or "cpu")
- `model_dir`: Path to the loaded model
- `max_batch`: Maximum number of texts per request
- `max_chars`: Maximum characters per text
- `max_length`: Maximum token length for model processing
- `threshold`: Classification threshold
- `model_test`: Result of internal model test ("passed" if successful)

---

### 2. Sentiment Prediction

**Endpoint**: `POST /predict`

**Description**: Classify sentiment of one or more movie reviews.

#### Input Validation Rules

- **Batch Size**: 1 to 64 texts per request (configurable via `MAX_BATCH`)
- **Text Length**: Maximum 5,000 characters per text (configurable via `MAX_CHARS`)
- **Token Length**: Automatically truncated to 500 tokens (configurable via `MAX_LENGTH`)
- **Content**: Non-empty strings only (whitespace-only texts are rejected)
- **Format**: All items must be valid strings

#### Request Schema

```json
{
  "texts": [
    "string"
  ]
}
```

**Fields**:
- `texts` (required): Array of review texts to classify
  - Type: List[str]
  - Min items: 1
  - Max items: 64 (default, configurable)
  - Max characters per text: 5,000 (default, configurable)

#### Response Schema

```json
{
  "results": [
    {
      "label": 1,
      "sentiment": "positive",
      "score_pos": 0.8934,
      "score_neg": 0.1066
    }
  ]
}
```

**Fields**:
- `results`: Array of prediction results (same order as input)
  - `label`: Predicted class (0=negative, 1=positive)
  - `sentiment`: Human-readable label ("positive" or "negative")
  - `score_pos`: Confidence score for positive sentiment [0-1]
  - `score_neg`: Confidence score for negative sentiment [0-1]

---

## Usage Examples

### Single Review Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"texts": ["This movie was absolutely fantastic! Great acting and storyline."]}'
```

**Response**:
```json
{
  "results": [
    {
      "label": 1,
      "sentiment": "positive",
      "score_pos": 0.943,
      "score_neg": 0.057
    }
  ]
}
```

### Batch Processing

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "texts": [
         "Amazing cinematography and stellar performances!",
         "Boring plot, couldn't wait for it to end.",
         "Pretty good movie, would recommend to friends.",
         "Worst film I have ever seen, total disaster."
       ]
     }'
```

**Response**:
```json
{
  "results": [
    {
      "label": 1,
      "sentiment": "positive", 
      "score_pos": 0.956,
      "score_neg": 0.044
    },
    {
      "label": 0,
      "sentiment": "negative",
      "score_pos": 0.109,
      "score_neg": 0.891
    },
    {
      "label": 1,
      "sentiment": "positive",
      "score_pos": 0.723,
      "score_neg": 0.277
    },
    {
      "label": 0,
      "sentiment": "negative",
      "score_pos": 0.022,
      "score_neg": 0.978
    }
  ]
}
```

### Large Text Example

```bash
# Long review (under 5,000 character limit)
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"texts": ["This movie was an incredible cinematic experience that left me speechless... [continues for several paragraphs]"]}'
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_DIR` | `runs/distilbert_auto/final_model` | Path to the trained model directory |
| `THRESHOLD` | `0.5` | Classification threshold (0.0-1.0) |
| `MAX_BATCH` | `64` | Maximum number of texts per request |
| `MAX_CHARS` | `5000` | Maximum characters per individual text |
| `MAX_LENGTH` | `500` | Maximum token length for model processing |

### Setting Environment Variables

```bash
# Linux/Mac
export MAX_BATCH=32
export MAX_CHARS=3000
export THRESHOLD=0.6

# Windows
set MAX_BATCH=32
set MAX_CHARS=3000
set THRESHOLD=0.6

# Docker
docker run -e MAX_BATCH=32 -e MAX_CHARS=3000 imdb-sentiment-api
```
---

## Error Handling

#### Empty or Invalid Input
```bash
# Empty texts array
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"texts": []}'
```

**Response**:
```json
{
  "detail": "`texts` is empty. Provide at least one text."
}
```

#### Text Too Long
```bash
# Text exceeding character limit
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"texts": ["Very long text exceeding 5000 characters..."]}'
```

**Response**:
```json
{
  "detail": "Item at index 0 exceeds 5000 characters. Shorten the input or increase MAX_CHARS."
}
```

#### Batch Too Large
```bash
# Batch exceeding size limit
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"texts": ["review1", "review2", ..., "review100"]}'  # 100 items
```

**Response**:
```json
{
  "detail": "Batch too large. Max allowed: 64."
}
```

#### Whitespace-Only Text
```bash
# Empty or whitespace-only text
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"texts": ["   \n\t   "]}'
```