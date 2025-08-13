# Sentiment Classification on IMDB Movie Reviews
## Problem Statement
The goal of this study is to classify IMDB movie reviews as positive or negative. Various machine learning and deep learning models are applied and compared to determine the most effective approach for sentiment classification. The study follows a structured process: exploratory data analysis (EDA), implementation of baseline and advanced models, and hyperparameter optimization.

### Table of Contents
- Overview
- Data Preprocessing and EDA (Step 1)
- Baseline Machine Learning Models (Step 2)
- Deep Learning and Transformer Models (Step 3)
- Cross-validation and Hyperparameter Tuning (Step 4)
- Results and Observations
- Conclusion

### Overview
The project begins with EDA to understand the dataset and clean the text data. Basic machine learning models are implemented first in both scratch and scikit-learn to establish a performance baseline. Deep learning models (DNN, RNN, LSTM) and a transformer-based model (DistilBERT) are then applied. Finally, the best models from the earlier stages are tuned using k-fold cross-validation and grid search to optimize their performance.

### Methods
#### Step 1: Data Preprocessing and EDA
- Removal of HTML tags, newline characters, and extra spaces.
- Optional conversion of emojis to text descriptions.
- Token distribution, class balance, and review length were analyzed.
- The processed dataset was saved for modeling.

#### Step 2: Baseline Machine Learning Models
- Implemented from scratch and with scikit-learn: Logistic Regression, Naive Bayes, LinearSVC, and SGDClassifier.
- TF-IDF vectorization with varying ngram_range and vocabulary size.
- Found that scikit-learn implementations performed better than scratch implementations in both speed and accuracy.

#### Step 3: Deep Learning and Transformer Models
Deep Neural Networks (DNN) with dense layers and dropout.

- Recurrent models (RNN, LSTM) with GloVe embeddings.
- DistilBERT fine-tuning for classification.
- DistilBERT was chosen instead of base BERT for reduced computational cost while retaining most of BERT’s accuracy.

#### Step 4: Cross-validation and Hyperparameter Tuning
Applied k-fold cross-validation with grid search on top-performing models from Steps 2 and 3.

- Parameters tuned included regularization strength for LinearSVC, loss functions and learning rates for SGDClassifier, and hidden dimensions/dropout for LSTM.
- Selection criteria were based on highest mean F1 score.

### Results and Observations
| Model                         | Mean Accuracy | Mean F1 | Notes                               |
|-------------------------------|--------------:|--------:|-------------------------------------|
| Naive Bayes (Scratch)         |         0.868 |   0.867 | Baseline scratch implementation     |
| KNN (Scratch)                 |         0.815 |   0.825 | Baseline scratch implementation     |
| Logistic Regression (Sklearn) |         0.880 |   0.880 | Improved over scratch               |
| LinearSVC (Tuned)             |         0.911 |   0.911 | Best tuned ML model                 |
| SGDClassifier (Tuned)         |         0.908 |   0.908 | Close to LinearSVC                  |
| DNN                           |         0.900 |   0.900 | Fully connected layers with dropout |
| RNN                           |         0.890 |   0.890 | Recurrent model without embeddings  |
| LSTM + GloVe                  |         0.913 |   0.913 | Competitive with best ML models     |
| DistilBERT                    |         0.916 |   0.916 | Best overall performance            |



### Observations

- EDA improved model stability by ensuring consistent text preprocessing.
Scratch models were significantly slower without performance gain.
- DistilBERT consistently outperformed other models with minimal overfitting.
- LSTM with pretrained embeddings was competitive but more resource-intensive.
- Cross-validation with grid search improved generalization and helped identify optimal configurations.

#### Effect of EDA
Data cleaning steps removed noise and standardized text, improving the performance of all models. Without EDA, models showed lower accuracy and higher variance across runs.

### Conclusion
The experiments demonstrate that transformer-based models like DistilBERT can achieve the best results on IMDB sentiment classification, even with reduced size compared to BERT. Classical ML algorithms, particularly LinearSVC, remain competitive when tuned, offering a faster and lighter alternative. Cross-validation and grid search are essential to reliably assess model performance and prevent overfitting. EDA plays a critical role in ensuring input quality and improving model stability.