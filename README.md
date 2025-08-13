# IMDB Sentiment Classification

End-to-end pipeline for IMDB movie review sentiment analysis:
- EDA and preprocessing
- Classic ML baselines (scikit-learn)
- Deep models (DNN/GRU/LSTM, GloVe-LSTM)
- DistilBERT fine-tuning
- K-fold CV + hyperparameter search
- FastAPI inference server
- Dockerized deployment

> - See [Report](REPORT.md) for extensive analysis.
> - [Documentation](/documentation) contains specific docs for the repo.


## Environment

- Python 3.10+
- PyTorch (CPU or CUDA)
- Hugging Face transformers, datasets, accelerate
- FastAPI + Uvicorn

## Installation & Usage

### Local Development
```bash
pip install -r requirements-dev.txt
jupyter-lab
```

### Training (DistilBERT via YAML)
1. Edit your config (example: configs/distilbert.yaml)

```yaml
model_name: distilbert-base-uncased
dataset_csv: data/IMDB_Dataset.csv
text_col: review
label_col: sentiment

seed: 42
lr: 5e-5
bsz: 32
epochs: 2
wd: 0.01
max_length: 256

eval_strategy: epoch
save_strategy: epoch
disable_tqdm: false
log_level: error
log_level_replica: error
logging_steps: 50
report_to: ["none"]
fp16: true

early_stopping: true
early_stopping_patience: 2
load_best_model_at_end: true
metric_for_best_model: accuracy
greater_is_better: true

output_dir: runs/distilbert_auto
```
2. Train

```shell
python src/train.py --config configs/distilbert.yaml
```
> Outputs (by default): `runs/distilbert_auto/final_model/` with the tokenizer and model weights.


### API Server

### Pre-trained Weights 
> Download the weights from https://drive.google.com/drive/folders/16tlUwq_1BX8Jlwb4JDjjfjrCCILdzW1H?usp=sharing and paste the contents inside runs folder

#### Running the App

```bash
set MODEL_DIR=runs/distilbert_auto/final_model   # Windows PowerShell: $env:MODEL_DIR="..."
uvicorn src.app:app --host 0.0.0.0 --port 8000
```

> Extensive documentation is available [Here. (API.md)](/documentation/api.md)

## Endpoints

- `GET /health` – service status
- `POST /predict` – batch prediction

**Example Request**
```commandline
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "An outstanding film with soulful performances.",
      "Boring plot and terrible acting."
    ]
  }'
```
**Example Response**
```json
{
  "results": [
    {"label": 1, "sentiment": "pos", "score_pos": 0.97, "score_neg": 0.03},
    {"label": 0, "sentiment": "neg", "score_pos": 0.08, "score_neg": 0.92}
  ]
}
```

## Docker
**Build**
```shell
docker build -t imdb-sentiment:latest .
```

### Docker Deployment
#### Run (Windows PowerShell)
Use an absolute path for the model mount:
```bash
docker build -t imdb-sentiment .
docker run --rm -p 8000:8000 -e MODEL_DIR=/models/distilbert/final_model -v "{{YOUR_PROJECT_ROOT}}/runs/distilbert_auto/":/models/distilbert imdb-sentiment:latest
```
#### Run (Linux/macOS)
```shell
docker run --rm -p 8000:8000 \
  -e MODEL_DIR=/models/distilbert/final_model \
  -v "$(pwd)/runs/distilbert_auto/final_model":/models/distilbert \
  imdb-sentiment:latest
```
> The container expects MODEL_DIR to point inside the container to a directory containing the HF model files (config.json, tokenizer.json, vocab.txt or merges.txt/bpe, model.safetensors, etc.).

## Notebooks
- [Step_1.eda_preprocessing.ipynb](/notebooks/Step_1.eda_preprocessing.ipynb) – EDA and cleaning
- [Step_2.ML_classification.ipynb](/notebooks/Step_2.ML_classification.ipynb) – scratch + sklearn baselines
- [Step_3.DNN_Classification.ipynb](/notebooks/Step_3.DNN_Classification.ipynb) – DNN/GRU/LSTM + DistilBERT training
- [Step_4.CrossVal_Tuning.ipynb](/notebooks/Step_4.CrossVal_Tuning.ipynb) – K-fold CV and grid search

## Report
- See [Report](REPORT.md) for detailed analysis.