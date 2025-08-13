# DistilBERT Training Pipeline Documentation

This document provides comprehensive documentation for the DistilBERT training pipeline consisting of `train.py` (training script) and `distilbert.yaml` (configuration file).

## Table of Contents

1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Configuration File (distilbert.yaml)](#configuration-file-distilbertyaml)
4. [Training Script (train.py)](#training-script-trainpy)
5. [Usage Examples](#usage-examples)
6. [Data Format](#data-format)
7. [Output Structure](#output-structure)
8. [Troubleshooting](#troubleshooting)

## Overview

This pipeline provides a complete solution for fine-tuning DistilBERT models on text classification tasks. Key features include:

- **Automatic Data Splitting**: Stratified 80/10/10 train/validation/test split
- **Flexible Configuration**: YAML-based configuration system
- **Multiple Model Support**: Any Hugging Face AutoModel compatible model
- **Advanced Training Features**: Early stopping, mixed precision (FP16), dynamic padding
- **Comprehensive Evaluation**: Accuracy, precision, recall, and F1-score metrics
- **Label Mapping**: Automatic conversion of string labels to numeric format

### Pipeline Architecture

```
CSV Dataset → Data Processing → Tokenization → Model Training → Evaluation → Model Saving
```

## Requirements

### Dependencies

```bash
pip install torch transformers datasets scikit-learn pandas pyyaml loguru
```

### Hardware Requirements

- **Minimum**: 4GB GPU memory (for batch_size=8)
- **Recommended**: 8GB+ GPU memory (for batch_size=16+)
- **CPU**: Works but significantly slower

### File Structure

```
project/
├── src/
│   └── train.py
├── config/
│   └── distilbert.yaml
├── data/
│   └── IMDB_Dataset.csv
└── runs/
    └── distilbert_auto/  # Output directory
```

## Configuration File (distilbert.yaml)

The YAML configuration file controls all aspects of training. Each parameter is documented below:

### Complete Configuration Example

```yaml
# Model Configuration
model_name: distilbert-base-uncased

# Dataset Configuration
dataset_csv: data/IMDB_Dataset.csv
text_col: review        # optional (defaults to "text")
label_col: sentiment    # optional (defaults to "label")
seed: 42

# Training Hyperparameters
lr: 0.00001            # Learning rate
bsz: 16                # Batch size
epochs: 3              # Number of training epochs
wd: 0.01               # Weight decay
max_length: 256        # Maximum sequence length

# Training Strategy
eval_strategy: "epoch"          # When to evaluate
save_strategy: "epoch"          # When to save checkpoints
disable_tqdm: false            # Show progress bars
log_level: "error"             # Logging level
log_level_replica: "error"     # Replica logging level
logging_steps: 50              # Log every N steps
report_to: ["none"]            # Experiment tracking
fp16: true                     # Mixed precision training

# Early Stopping
early_stopping: true                    # Enable early stopping
early_stopping_patience: 2             # Patience epochs
load_best_model_at_end: true           # Load best checkpoint
metric_for_best_model: "accuracy"      # Metric to monitor
greater_is_better: true                # Higher metric = better

# Output
output_dir: runs/distilbert_auto       # Output directory
```

### Parameter Details

#### Model Configuration

- **model_name** (str): Hugging Face model identifier
  - Examples: `distilbert-base-uncased`, `bert-base-uncased`, `roberta-base`
  - Must support AutoModelForSequenceClassification

#### Dataset Configuration

- **dataset_csv** (str): Path to CSV file containing dataset
- **text_col** (str, default="text"): Column name containing text data
- **label_col** (str, default="label"): Column name containing labels
- **seed** (int, default=42): Random seed for reproducible splits

#### Training Hyperparameters

- **lr** (float, default=2e-5): Learning rate for optimizer
- **bsz** (int, default=16): Training and evaluation batch size
- **epochs** (float, default=3): Number of training epochs
- **wd** (float, default=0.01): Weight decay for regularization
- **max_length** (int, default=256): Maximum input sequence length

#### Training Strategy

- **eval_strategy** (str, default="epoch"): Evaluation frequency
  - Options: "no", "steps", "epoch"
- **save_strategy** (str, default="no"): Checkpoint saving frequency
  - Options: "no", "steps", "epoch"
- **fp16** (bool, default=auto): Enable mixed precision training
- **disable_tqdm** (bool, default=false): Hide progress bars

#### Early Stopping

- **early_stopping** (bool, default=false): Enable early stopping
- **early_stopping_patience** (int, default=2): Epochs to wait before stopping
- **load_best_model_at_end** (bool): Load best checkpoint after training
- **metric_for_best_model** (str, default="accuracy"): Metric to optimize

## Training Script (train.py)

The training script handles the complete pipeline from data loading to model evaluation.

### Script Structure

```python
# Main components:
1. Configuration parsing and validation
2. Data loading and preprocessing  
3. Dataset splitting (80/10/10)
4. Tokenization and padding
5. Model initialization
6. Training loop with evaluation
7. Final testing and model saving
```

### Key Functions

#### parse_args()

```python
def parse_args():
    ap = argparse.ArgumentParser("Train DistilBERT from a YAML config")
    ap.add_argument("-c", "--config", required=True, help="Path to YAML config")
    return ap.parse_args()
```

Parses command line arguments to get configuration file path.

#### load_config(path)

```python
def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
```

Loads and parses YAML configuration file.

#### build_datasets(cfg)

```python
def build_datasets(cfg):
    """
    Load CSV and create stratified 80/10/10 split.
    Handles label mapping and data cleaning.
    Returns: DatasetDict with 'train'/'validation'/'test'
    """
```

**Process:**
1. Load CSV file using pandas
2. Rename columns if needed (text_col → "text", label_col → "label")
3. Map string labels to numeric:
   - `pos`, `positive` → 1
   - `neg`, `negative` → 0
4. Clean data (remove NaN, empty strings)
5. Perform stratified split: 80% train, 10% validation, 10% test
6. Convert to Hugging Face Dataset format

**Input Requirements:**
- CSV must contain text and label columns
- Labels can be string ("pos"/"neg") or numeric (0/1)
- Text should be non-empty strings

#### tokenize_builder(tokenizer, max_length)

```python
def tokenize_builder(tokenizer, max_length: int):
    def tok(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding=False,  # Dynamic padding via collator
            max_length=max_length,
        )
    return tok
```

Creates tokenization function with specified parameters.

#### compute_metrics(eval_pred)

```python
def compute_metrics(eval_pred):
    logits, y_true = eval_pred
    y_pred = np.argmax(logits, axis=1)
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}
```

Calculates evaluation metrics during training and testing.

### Data Processing Pipeline

#### 1. Label Mapping

```python
# String labels to numeric
if df["label"].dtype == object:
    lab = (
        df["label"].astype(str).str.strip().str.lower()
        .replace({"pos": 1, "positive": 1, "neg": 0, "negative": 0})
    )
    df["label"] = lab.astype(int)
```

#### 2. Data Cleaning

```python
# Remove missing data and empty texts
df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
df = df[df["text"].astype(str).str.strip().ne("")].reset_index(drop=True)
```

#### 3. Stratified Splitting

```python
# 80/20 split
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=seed, stratify=y
)

# 12.5% of 80% = 10% overall validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.125, random_state=seed, stratify=y_train_full
)
```

### Training Pipeline

#### 1. Model Initialization

```python
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
```

#### 2. Training Arguments

```python
args_hf = TrainingArguments(
    output_dir=out_dir,
    learning_rate=lr,
    per_device_train_batch_size=bsz,
    per_device_eval_batch_size=bsz,
    num_train_epochs=epochs,
    weight_decay=wd,
    eval_strategy=eval_strategy,
    save_strategy=save_strategy,
    load_best_model_at_end=load_best,
    metric_for_best_model=metric_best,
    fp16=fp16,
    # ... other parameters
)
```

#### 3. Trainer Setup

```python
trainer = Trainer(
    model=model,
    args=args_hf,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
    callbacks=callbacks,  # Early stopping if enabled
)
```

#### 4. Training and Evaluation

```python
# Train the model
trainer.train()

# Evaluate on validation set
val_metrics = trainer.evaluate(eval_dataset=ds["validation"])

# Evaluate on test set
test_metrics = trainer.evaluate(eval_dataset=ds["test"])

# Save final model
trainer.save_model(str(save_dir))
tokenizer.save_pretrained(str(save_dir))
```

## Usage Examples

### Basic Usage

```bash
# Run training with configuration file
python train.py --config distilbert.yaml
```