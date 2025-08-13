# Sentiment Classification on IMDB Movie Reviews
**A Comprehensive Study: From Traditional ML to Transformer Models**

---

## Summary

This study presents a comprehensive comparison of machine learning and deep learning approaches for sentiment classification on the IMDB movie reviews dataset. We implemented and evaluated 15+ different models ranging from scratch implementations to state-of-the-art transformers. **DistilBERT achieved the best performance (91.6% F1)**, while LinearSVC provided the best traditional ML baseline (91.1% F1). The study demonstrates that while transformer models excel in accuracy, traditional ML approaches remain competitive and efficient for production environments.

---

## Table of Contents
1. [Problem Statement](#problem-statement)
2. [Dataset and Preprocessing](#dataset-and-preprocessing)
3. [Methodology](#methodology)
4. [Experimental Results](#experimental-results)
5. [Performance Analysis](#performance-analysis)
6. [Implementation Details](#implementation-details)
7. [Conclusions and Future Work](#conclusions-and-future-work)
8. [Appendix](#appendix)

---

## Problem Statement

The goal is to classify IMDB movie reviews as positive or negative sentiment using various machine learning approaches. This binary classification task serves as a benchmark for comparing traditional ML algorithms, deep learning models, and transformer-based approaches.

**Key Objectives:**
- Implement models from scratch to understand underlying mechanics
- Compare scratch implementations with optimized sklearn versions
- Evaluate deep learning models with different architectures
- Fine-tune transformer models for optimal performance
- Provide comprehensive cross-validation and hyperparameter optimization

---

## Dataset and Preprocessing

### Dataset Overview
- IMDB Dataset of 50K Movie Reviews (Kaggle)
- 50,000 movie reviews (25,000 positive, 25,000 negative)
- Balanced binary classification dataset 
- Variable length reviews with HTML tags, contractions, and informal language

### Preprocessing Pipeline

**Traditional ML Models (Applied):**
```
# Comprehensive text cleaning pipeline
1. HTML tag removal: <br />, <p>, etc.
2. Newline and space normalization
3. Emoji conversion to text descriptions
4. Contraction expansion: "don't" → "do not"
5. Rating normalization: "8/10" → "RATING_POS"
6. Lowercasing and punctuation removal
7. Stop word removal
8. Optional: Stemming/Lemmatization
```

**Transformer Models (Minimal Processing):**
- Raw text
- Leverages pre-trained tokenizers and embeddings
> Please refer to [EDA Notebook](/notebooks/Step_1.eda_preprocessing.ipynb) for details. There is also documentation related to preprocessing available on [Docs](/documentation/preprocessor.md).

<figure>
  <img width="1340" height="449" alt="image" src="https://github.com/user-attachments/assets/8e8c4b98-eda7-42aa-8eb7-ff0925d91a71" />
  <p style="text-align: center;">Fig 1: Data counts for  positive, negative classes</p>
</figure>

---

## Methodology

The idea of this methodology is to explore several approaches to create an app that can detect positive/negative sentiment from a given movie review.

This problem has been approached methodically. For an example, doing Exploratory Data Analysis, building ML models from scratch, training pre-built models and then incrementally increasing model complexity from Classical methods to neural networks and finally ending in Transformer based approach. The goal here is to identify the proper model that can be used depending on the scenario, reasoning behind choosing parameters overall doing Research and Development. 

> There is a notebook for every step. WHich has been also linked below. These notebooks contain all the images, tables, graphs needed. This report only contains a summary. 

### Step 1: Baseline Implementation (Scratch Models)
Implemented foundational algorithms from scratch to understand core mechanisms:
- Naive Bayes: Multinomial with Laplace smoothing
- K-Nearest Neighbors: Cosine similarity with sparse matrix optimization
- Custom Vectorizers: Bag-of-Words and TF-IDF implementations

Optimized implementations using scikit-learn:
- Models: Logistic Regression, LinearSVC, SGDClassifier, Multinomial/Bernoulli NB
- Vectorization: TF-IDF with n-gram combinations (unigrams, bigrams, trigrams)
- Optimization: Grid search across regularization, vocabulary size, and feature selection

> Please refer to [ML Classification Notebook](/notebooks/Step_2.ML_classification.ipynb) for details. There is also documentation related to Models available on [Docs](/documentation/models.md).

### Step 2: Deep Learning Models
Neural network architectures with PyTorch:
- DNN: Dense layers with dropout regularization
- RNN/LSTM: Recurrent architectures with bidirectional processing
- Embeddings: Custom embeddings vs. pre-trained GloVe (100-dim)
- Architecture: Embedding → RNN/LSTM → Dropout → Linear classifier

> Please refer to [DNN Classification Notebook](/notebooks/Step_3.DNN_Classification.ipynb) for details. There is also documentation related to Models available on [Docs](/documentation/models.md).


### Step 3: Cross-Validation and Hyperparameter Optimization
Rigorous evaluation methodology:
- Cross-Validation: 5-fold stratified CV for all models
- Grid Search: Comprehensive hyperparameter exploration
- Metrics: Accuracy, F1-score, Precision, Recall, ROC-AUC
- Statistical Analysis: Mean ± standard deviation across folds

State-of-the-art approach using Hugging Face Transformers:
- Model: DistilBERT (distilbert-base-uncased)
- Strategy: Fine-tuning with task-specific classification head
- Optimization: Learning rate scheduling, weight decay, early stopping

> Please refer to [Cross Validation & Tuning Notebook](/notebooks/Step_4.CrossVal_Tuning.ipynb) for details.

The final model was trained based on the best parameter and results found. For the app, DIstilBERT was trained with the best parameters set.

---

## Experimental Results

### Comprehensive Model Comparison

| Rank | Model Category | Model | Accuracy | F1 Score | Training Time | Notes |
|------|---------|-------|----------|----------|---------------|--------|
| 1 | **Transformer** | DistilBERT (Tuned) | **0.916** | **0.916** | ~45 min | Best overall performance |
| 2 | Deep Learning | GloVe + LSTM (CV) | 0.913 | 0.913 | ~25 min | Strong contextual learning |
| 3 | Traditional ML | LinearSVC (Tuned) | 0.911 | 0.911 | ~2 min | Best traditional ML |
| 4 | Traditional ML | SGDClassifier (Tuned) | 0.908 | 0.908 | ~1 min | Fast and competitive |
| 5 | Deep Learning | DNN | 0.890 | 0.889 | ~15 min | Dense neural network |
| 6 | Deep Learning | RNN | 0.887 | 0.890 | ~20 min | Basic recurrent model |
| 7 | Deep Learning | LSTM | 0.882 | 0.880 | ~22 min | Sequential processing |
| 8 | Traditional ML | Logistic Regression | 0.880 | 0.880 | ~1 min | Interpretable baseline |
| 9 | Scratch | Naive Bayes (TF-IDF) | 0.868 | 0.867 | ~30 sec | Custom implementation |
| 10 | Scratch | KNN (TF-IDF) | 0.815 | 0.826 | ~5 min | Distance-based classifier |

<div>
  <p style="text-align: center;">Table 1: Comprehensive result comparison</p>
</div>


<figure>
  <img width="1341" height="664" alt="image" src="https://github.com/user-attachments/assets/1ba119fc-e379-4190-a08a-783c3a68b59a" />
  <p style="text-align: center;"s>Fig 2: Top features selected by Linear SVC</p>
</figure>

### Cross-Validation Results (Top Models)

| Model | Mean Accuracy | Std Accuracy | Mean F1 | Std F1 | Stability |
|-------|--------------|--------------|---------|--------|-----------|
| DistilBERT | 0.916 ± 0.003 | 0.003 | 0.916 ± 0.003 | 0.003 | Excellent |
| LinearSVC | 0.911 ± 0.005 | 0.005 | 0.911 ± 0.004 | 0.004 | Excellent |
| GloVe + LSTM | 0.879 ± 0.007 | 0.007 | 0.879 ± 0.006 | 0.006 | Good |

<div>
  <p style="text-align: center;">Table 2: Cross validation statistics</p>
</div>
---

## Performance Analysis

### Threshold Analysis for Deep Learning Models

Deep learning models showed varying sensitivity to classification thresholds:

| Model | Optimal Threshold | Accuracy | F1 | Precision | Recall | Trade-off |
|-------|------------------|----------|----|-----------|---------|-----------| 
| DNN | 0.5 | 0.890 | 0.889 | 0.899 | 0.879 | Balanced |
| RNN | 0.6 | 0.887 | 0.888 | 0.881 | 0.896 | Recall-focused |
| LSTM | 0.5 | 0.882 | 0.880 | 0.898 | 0.862 | Precision-focused |
| GloVe-LSTM | 0.5 | 0.888 | 0.888 | 0.895 | 0.881 | Balanced |

<div>
  <p style="text-align: center;">Table 3: Performance metrices of DL methods</p>
</div>


### Hyperparameter Optimization Results

#### Traditional ML (Grid Search + 5-Fold CV)
```
Best Configuration (LinearSVC):
- C: 0.5 (regularization strength)
- ngram_range: (1, 2) (unigrams + bigrams)  
- min_df: 3 (minimum document frequency)
- max_features: None (use full vocabulary)

Grid Search: 750 total fits (150 candidates × 5 folds)
Best F1: 0.9110
```

#### GloVe + LSTM (Parameter Search)
```
Parameter Exploration:
- Embedding dim: 100 (optimal for GloVe)
- Hidden dims: [128, 256] → 128 optimal
- Dropout: [0.4, 0.5] → 0.4 optimal  
- Bidirectional: True (captures both directions)
- Learning rate: [0.001, 0.0005] → 0.001 optimal

Results:
- Config 1: hidden=128, dropout=0.4 → F1: 0.879 ± 0.006
- Config 2: hidden=256, dropout=0.5 → F1: 0.877 ± 0.009
```

#### DistilBERT (Fine-tuning Grid Search)
```
Hyperparameter Grid:
- Learning rates: [2e-5, 5e-5, 1e-4]
- Batch sizes: [16, 32]  
- Epochs: [2, 3]
- Weight decay: [0.01, 0.1]

Best Configuration:
- Learning rate: 5e-5
- Batch size: 32
- Epochs: 2
- Weight decay: 0.01
- F1: 0.9163

Performance Stability: <0.4% variance between top configs
```

### Key Insights and Observations

#### 1. **Traditional ML Efficiency**
- LinearSVC achieved 91.1% F1 with ~2 minutes training time
- Competitive with deep learning while being 10-20x faster
- Better choice for production environments with resource constraints

#### 2. **Deep Learning Trade-offs**  
- LSTM models showed better contextual understanding than bag-of-words
- GloVe embeddings provided significant boost over random initialization
- Higher variance across folds compared to traditional ML
- Required careful hyperparameter tuning to avoid overfitting

#### 3. **Transformer Excellence**
- DistilBERT achieved best absolute performance (91.6% F1)
- Remarkable stability across different hyperparameter configurations
- Minimal preprocessing required due to robust pre-training
- Higher computational cost but superior accuracy

#### 4. **Threshold Sensitivity**
- RNN models performed better at threshold=0.6 (recall-optimized)
- DNN and LSTM optimal at threshold=0.5 (balanced)
- Suggests different confidence calibration across architectures

### Insights of DistilBERT
#### ROC Curve analysis
- AUC Score: The model achieved an ROC AUC of 0.975, indicating excellent ranking ability between positive and negative samples.
- The ROC curve remains close to the top-left corner, confirming a low false positive rate for high true positive rates.
- The stability of the curve suggests the model is well-calibrated for a variety of operating thresholds.

<img width="908" height="724" alt="image" src="https://github.com/user-attachments/assets/2a776855-de32-4715-9bd8-72201f79169c" />

Fig 3: ROC Curve for DistilBERT with different thresholds

#### Precision-Recall (PR) Curve Analysis
- Average Precision (AP) remains consistent at 0.975 across thresholds, indicating that the model maintains a high level of precision even as recall varies.
- This consistency implies the model's confidence scores are reliable, with minimal degradation when the decision threshold is adjusted.
<img width="907" height="721" alt="image" src="https://github.com/user-attachments/assets/b30ad44e-3812-4cbb-8836-9701a2a77b9c" />

Fig 4: PR Curve of DistilBERT with different thresholds

#### Confusion Matrix Trends
- Threshold 0.50 : Balanced performance between false positives (416) and false negatives (393). Suitable when both precision and recall are equally important.
- Threshold 0.60 : Slight reduction in false positives (377) with a small increase in false negatives (443). Appropriate if precision is marginally more important than recall.
- Threshold 0.70 : Significant drop in false positives (337) but noticeable increase in false negatives (493). Ideal for scenarios where minimizing false alarms is critical.
- Threshold 0.80 : Minimal false positives (287) but higher false negatives (559). Best suited for applications where incorrect positive predictions have a high operational or financial cost.

<img width="891" height="718" alt="image" src="https://github.com/user-attachments/assets/4b399eac-d3f4-46ba-b408-d7511a00d962" />

Fig 5: Confusion matrix for DistilBERT with different thresholds

---

## Implementation Details

### Project Structure
```
movie-sentiment-classification/
├── configs/
│   └── distilbert.yaml
├── data/
│   ├── glove.6B.100d.txt
│   ├── IMDB_Dataset.csv
│   └── imdb_reviews.parquet
├── documentation/
│   ├── API_DOCUMENTATION.md      # Your comprehensive API docs
│   ├── preprocessor.md
│   ├── vectorizer.md
│   └── classifier.md
├── notebooks/
│   ├── Step_1.eda_preprocessing.ipynb
│   ├── Step_2.ML_classification.ipynb
│   ├── Step_3.DNN_Classification.ipynb
│   └── Step_4.CrossVal_Tuning.ipynb
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── dnn.py               # Deep Neural Network
│   │   ├── lstm.py              # LSTM models
│   │   ├── rnn.py               # RNN models
│   │   ├── naive_bayes.py       # Scratch Naive Bayes
│   │   └── knn.py               # Scratch KNN
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── constants.py         # Contractions dictionary
│   │   ├── preprocessing.py     # Text preprocessing
│   │   ├── pos_tag.py          # POS tagging utilities
│   │   ├── embeddings.py        # GloVe embeddings loader
│   │   ├── model_dataclass.py   # Model evaluation dataclass
│   │   └── model_utils.py       # Evaluation and plotting functions
│   ├── vectorizer/
│   │   ├── __init__.py
│   │   ├── bag_of_words.py      # Custom BoW implementation
│   │   └── tfidf.py             # Custom TF-IDF implementation
│   └── app.py                   # FastAPI application
├── models/
│   └── saved_models/            # Trained model artifacts
├── runs/
│   └── distilbert_auto/
│       └── final_model/         # DistilBERT fine-tuned model
├── requirements-dev.txt         # Development dependencies
├── requirements-prod.txt        # Production dependencies
├── Dockerfile                   # Container configuration
├── REPORT.md                   # Comprehensive project report
└── README.md                   # Setup and usage instructions

```

### Technical Implementation Notes

#### Scratch vs. Sklearn Performance Gap
- Sklearn 5-50x faster due to optimized implementations
- Sklearn more efficient with sparse matrix operations
- Sklearn typically 2-5% better due to advanced optimization
- Scratch implementations crucial for understanding algorithms


---

## Conclusions and Future Work

### Key Findings

1. Model Performance Hierarchy: DistilBERT > LinearSVC > GloVe+LSTM > Traditional ML > Scratch Models
2. Efficiency vs. Accuracy Trade-off**: LinearSVC provides 98% of DistilBERT's performance at 5% of the computational cost
3. Preprocessing Impact: Proper text cleaning improved traditional ML by 3-5% but had minimal impact on transformers
4. Cross-validation Necessity: Single train/test splits can be misleading; CV revealed true model stability


| Use Case | Recommended Model | Reason                                  |
|----------|-------------------|-----------------------------------------|
| High Accuracy Required | DistilBERT | Best performance, acceptable latency    |
| Real-time Applications | LinearSVC | Fast inference, competitive accuracy    |
| Resource Constrained | Naive Bayes | Minimal memory, decent performance      |
| Interpretability Needed | Logistic Regression | Coefficients provide feature importance |

<div>
  <p style="text-align: center;">Table 4: Use case for models. </p>
</div>


### Future Work

1. Ensemble Methods: Combine top models using voting or stacking,bagging, boosting
2. Advanced Transformers: Experiment with RoBERTa, ELECTRA, or newer architectures

### Model Degradation (Bonus Task)
1. **Early Detection**
- Data Quality Monitoring (Vocabulary, Review Patterns)
- User feedback 
- Log/Prediction monitoring

2. **Automated re-training**
- Data collection from several sources (when a new movie is released)
- Continuous training and Testing (Running parallely with production model) , A/B Testing

3. **LLM /Agentic Supervising**
- Synthetic data for drifted regions
- Teacher-student system to score/justify predictions on recent data and re-train with the new data
- LLMs can be costly, so minimal usage is preferred

## Appendix

### A. Hardware and Environment
- GPU: NVIDIA RTX 4070 Ti (12GB VRAM)
- RAM: 64GB DDR5
- Framework: `PyTorch`, `Scikit-learn`, `Transformers`, `FastAPI`
- Training Time: Total ~6 hours across all experiments

### B. Reproducibility
- Seeds: Fixed random seeds for all frameworks
- Dependencies: requirements-dev.txt with exact versions
- Data: Original Kaggle dataset with preprocessing scripts

---

Note: This report represents a comprehensive exploration of sentiment classification approaches. Code, models, and detailed notebooks are available in the project repository.

***Some values may not be precise due to running the notebooks after writing the report.***
