# Machine Learning Models Documentation

This document provides comprehensive documentation for five machine learning models implemented for text classification: Naive Bayes, K-Nearest Neighbors, Deep Neural Network, Recurrent Neural Network, and Long Short-Term Memory networks.

## Table of Contents

1. [Naive Bayes](#naive-bayes)
2. [K-Nearest Neighbors (KNN)](#k-nearest-neighbors-knn)
3. [Deep Neural Network (DNN)](#deep-neural-network-dnn)
4. [Recurrent Neural Network (RNN)](#recurrent-neural-network-rnn)
5. [Long Short-Term Memory (LSTM)](#long-short-term-memory-lstm)
6. [Usage Examples](#usage-examples)
7. [Performance Comparison](#performance-comparison)

## Naive Bayes

A probabilistic classifier based on Bayes' theorem with strong independence assumptions between features. This implementation supports both sparse and dense matrices with Laplace smoothing.

### Class Definition

```python
class NaiveBayes:
    def __init__(self, alpha: float = 1.0):
```

### Parameters

- **alpha** (float, default=1.0): Laplace smoothing parameter to handle zero probabilities

### Mathematical Foundation

Naive Bayes classifies documents based on:

```
P(class|document) ∝ P(class) × ∏ P(word|class)
```

With Laplace smoothing:
```
P(word|class) = (count(word, class) + α) / (total_words_in_class + α × vocabulary_size)
```

### Methods

#### fit(X, y)
Trains the classifier on feature matrix X and labels y.

**Process:**
1. Calculate class priors: P(class) = count(class) / total_documents
2. Calculate feature likelihoods with Laplace smoothing
3. Store log probabilities for numerical stability

**Parameters:**
- **X**: Feature matrix (n_samples, n_features) - supports sparse or dense
- **y**: Target labels array

#### predict_log_proba(X)
Returns log probabilities for each class.

**Returns:** Array of shape (n_samples, n_classes) with log probabilities

#### predict(X)
Predicts class labels for input samples.

**Returns:** Array of predicted class indices

### Attributes

- **classes_**: Array of unique class labels
- **class_log_prior_**: Log probabilities of each class
- **feature_log_prob_**: Log probabilities of features given each class
- **is_trained**: Boolean indicating if model has been fitted

### Implementation Details

```python
# Example of log probability calculation
for each class c:
    class_log_prior[c] = log(P(c))
    for each feature f:
        feature_log_prob[c,f] = log((count(f,c) + alpha) / (total_count(c) + alpha * vocab_size))

# Prediction
log_prob = X @ feature_log_prob.T + class_log_prior
predicted_class = argmax(log_prob, axis=1)
```

## K-Nearest Neighbors (KNN)

A non-parametric classifier that assigns labels based on the majority vote of k nearest neighbors. This implementation is optimized for sparse matrices using cosine similarity.

### Class Definition

```python
class KNN:
    def __init__(self, k=7, distance_metric="cosine", batch_size=2000):
```

### Parameters

- **k** (int, default=7): Number of nearest neighbors to consider
- **distance_metric** (str, default="cosine"): Distance metric (currently only cosine supported)
- **batch_size** (int, default=2000): Batch size for processing large datasets

### Algorithm

1. **Training**: Store all training samples and labels
2. **Prediction**: 
   - Calculate similarities between test samples and all training samples
   - Find k most similar training samples
   - Return majority class among k neighbors

### Methods

#### fit(X, y)
Stores training data for lazy learning approach.

**Parameters:**
- **X**: Training feature matrix (supports sparse/dense)
- **y**: Training labels array

#### predict(X)
Predicts labels using k-nearest neighbors voting.

**Process:**
1. Calculate cosine similarity between test and training samples
2. For each test sample, find k nearest neighbors
3. Return majority class vote

**Parameters:**
- **X**: Test feature matrix

**Returns:** Array of predicted labels

#### _predict_batch(Xb)
Internal method for batch prediction to handle memory efficiently.

### Sparse Matrix Optimization

```python
# Efficient sparse matrix operations
sims = cosine_similarity(X_test, X_train, dense_output=False)
for each test sample:
    top_k_indices = argpartition(similarities, -k)[-k:]
    neighbor_labels = y_train[top_k_indices]
    prediction = majority_vote(neighbor_labels)
```

### Attributes

- **X_train**: Stored training features (CSR sparse matrix)
- **y_train**: Stored training labels
- **is_trained**: Boolean indicating if model has been fitted

## Deep Neural Network (DNN)

A feedforward neural network for text classification using word embeddings and average pooling.

### Class Definition

```python
class DNN(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hidden_dim=128, dropout=0.4):
```

### Parameters

- **vocab_size** (int): Size of vocabulary for embedding layer
- **emb_dim** (int, default=128): Dimension of word embeddings
- **hidden_dim** (int, default=128): Hidden layer dimension
- **dropout** (float, default=0.4): Dropout rate for regularization

### Architecture

```
Input (token indices) 
    ↓
Embedding Layer (vocab_size → emb_dim)
    ↓
Average Pooling (along sequence dimension)
    ↓
Linear Layer (emb_dim → hidden_dim)
    ↓
ReLU Activation
    ↓
Dropout (rate=0.4)
    ↓
Output Linear Layer (hidden_dim → 1)
    ↓
Output (single logit for binary classification)
```

### Forward Pass

```python
def forward(self, x):
    # x: (batch_size, sequence_length)
    emb = self.embedding(x)           # (B, L, E)
    avg_emb = emb.mean(dim=1)         # (B, E) - average pooling
    h = self.fc1(avg_emb)             # (B, H)
    h = self.relu(h)                  # (B, H)
    h = self.dropout(h)               # (B, H)
    out = self.fc2(h).squeeze(1)      # (B,) - single logit
    return out
```

### Input Format

- **Input**: Tensor of token indices, shape (batch_size, sequence_length)
- **Output**: Raw logits for binary classification, shape (batch_size,)
- **Training**: Use with BCEWithLogitsLoss
- **Inference**: Apply sigmoid to get probabilities

### Key Features

- **Average Pooling**: Simple but effective sequence aggregation
- **Dropout Regularization**: Prevents overfitting
- **Single Hidden Layer**: Efficient architecture
- **Padding Index**: Supports variable-length sequences with padding

## Recurrent Neural Network (RNN)

A recurrent neural network using GRU cells for sequential text processing with optional bidirectionality.

### Class Definition

```python
class RNN(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 128, hidden_size: int = 128, 
                 num_layers: int = 1, bidirectional: bool = False, dropout: float = 0.3):
```

### Parameters

- **vocab_size** (int): Vocabulary size for embeddings
- **emb_dim** (int, default=128): Embedding dimension
- **hidden_size** (int, default=128): GRU hidden state size
- **num_layers** (int, default=1): Number of GRU layers
- **bidirectional** (bool, default=False): Whether to use bidirectional GRU
- **dropout** (float, default=0.3): Dropout rate

### Architecture

```
Input (token indices)
    ↓
Embedding Layer (vocab_size → emb_dim, padding_idx=0)
    ↓
GRU Layers (emb_dim → hidden_size × num_layers)
    ↓
Last Hidden State Extraction
    ↓ (if bidirectional: concatenate forward + backward)
Dropout
    ↓
Linear Layer (hidden_size → 1)
    ↓
Output (single logit)
```

### Forward Pass

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    e = self.emb(x)                    # (B, T, E)
    out, h = self.rnn(e)               # h: (L*directions, B, H)
    
    if self.bidirectional:
        # Concatenate last forward and backward states
        last = torch.cat([h[-2], h[-1]], dim=1)  # (B, 2H)
    else:
        last = h[-1]                   # (B, H)
    
    z = self.drop(last)
    logit = self.fc(z).squeeze(1)      # (B,)
    return logit
```

### Key Features

- **GRU Cells**: More efficient than LSTM, less prone to vanishing gradients than vanilla RNN
- **Bidirectional Processing**: Optional forward and backward sequence processing
- **Variable Sequence Length**: Handles padded sequences (padding_idx=0)
- **Multiple Layers**: Supports stacked GRU layers with dropout

### Bidirectional Processing

When bidirectional=True:
- Forward GRU processes sequence left-to-right
- Backward GRU processes sequence right-to-left  
- Final representation concatenates both directions: [h_forward; h_backward]

## Long Short-Term Memory (LSTM)

LSTM networks designed for sequence modeling with memory cells to capture long-term dependencies. Includes both standard LSTM and GloVe-enhanced variants.

### Standard LSTM

#### Class Definition

```python
class LSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hidden_dim=128, num_layers=1,
                 bidirectional=False, dropout=0.4, pad_idx=0):
```

#### Parameters

- **vocab_size** (int): Vocabulary size
- **emb_dim** (int, default=128): Embedding dimension
- **hidden_dim** (int, default=128): LSTM hidden dimension
- **num_layers** (int, default=1): Number of LSTM layers
- **bidirectional** (bool, default=False): Bidirectional processing
- **dropout** (float, default=0.4): Dropout rate
- **pad_idx** (int, default=0): Padding token index

#### Architecture

```
Input Tokens
    ↓
Embedding Layer (with padding_idx=0)
    ↓
LSTM Layers (input_size=emb_dim, hidden_size=hidden_dim)
    ↓
Final Hidden State (last layer)
    ↓ (if bidirectional: concatenate forward + backward)
Dropout
    ↓
Linear Layer (hidden_dim → 1)
    ↓
Output Logit
```

#### LSTM Cell Components

LSTM uses three gates and a cell state:
- **Forget Gate**: Decides what information to discard
- **Input Gate**: Decides what new information to store
- **Output Gate**: Controls what parts of cell state to output
- **Cell State**: Long-term memory component

### GloVe LSTM

#### Class Definition

```python
class GloVeLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers, bidirectional, 
                 dropout, pad_idx, pretrained_embeddings=None):
```

#### Key Features

- **Pretrained Embeddings**: Supports GloVe or other pretrained embeddings
- **Frozen Weights**: Embedding weights can be frozen during training
- **Transfer Learning**: Leverages semantic knowledge from large corpora

#### Initialization with Pretrained Embeddings

```python
if pretrained_embeddings is not None:
    self.embedding.weight.data.copy_(pretrained_embeddings)
    self.embedding.weight.requires_grad = False  # Freeze embeddings
```

### Forward Pass (Both Variants)

```python
def forward(self, x):
    embedded = self.embedding(x)              # (B, L, E)
    output, (hidden, cell) = self.lstm(embedded)
    
    if self.lstm.bidirectional:
        # Concatenate final forward and backward hidden states
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
    else:
        hidden = hidden[-1]                   # Last layer, last time step
    
    return self.fc(self.dropout(hidden)).squeeze(1)
```

### LSTM vs RNN Comparison

| Feature | RNN (GRU) | LSTM |
|---------|-----------|------|
| Gates | Reset, Update | Forget, Input, Output |
| Memory | Hidden state only | Hidden state + Cell state |
| Parameters | Fewer | More |
| Long-term dependencies | Good | Excellent |
| Training speed | Faster | Slower |
| Vanishing gradients | Less prone | Most resistant |

## Usage Examples

### Naive Bayes Example

```python
from naive_bayes import NaiveBayes
from scipy.sparse import csr_matrix
import numpy as np

# Sample TF-IDF features (sparse matrix)
X_train = csr_matrix([[1, 0, 2, 1], [0, 1, 1, 0], [2, 1, 0, 1]])
y_train = np.array([1, 0, 1])  # Binary labels

# Train model
nb = NaiveBayes(alpha=1.0)
nb.fit(X_train, y_train)

# Predict
X_test = csr_matrix([[1, 1, 0, 1], [0, 0, 1, 2]])
predictions = nb.predict(X_test)
probabilities = nb.predict_log_proba(X_test)

print(f"Predictions: {predictions}")
print(f"Log probabilities: {probabilities}")
```

### KNN Example

```python
from knn import KNN
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample documents
docs = ["great movie excellent acting", "terrible film bad plot", 
        "amazing story wonderful characters", "awful script poor direction"]
labels = [1, 0, 1, 0]

# Vectorize
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(docs)

# Train KNN
knn = KNN(k=3, distance_metric="cosine")
knn.fit(X, labels)

# Test
test_docs = ["fantastic movie", "boring film"]
X_test = vectorizer.transform(test_docs)
predictions = knn.predict(X_test)

print(f"Predictions: {predictions}")
```

### DNN Example

```python
import torch
import torch.nn as nn
from dnn import DNN

# Model parameters
vocab_size = 10000
model = DNN(vocab_size=vocab_size, emb_dim=128, hidden_dim=128, dropout=0.3)

# Sample input (batch_size=2, sequence_length=10)
input_ids = torch.randint(1, vocab_size, (2, 10))

# Forward pass
logits = model(input_ids)
print(f"Output shape: {logits.shape}")  # Should be (2,)

# Training setup
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters())

# Sample training step
targets = torch.tensor([1.0, 0.0])  # Binary targets
loss = criterion(logits, targets)
loss.backward()
optimizer.step()
```

### RNN Example

```python
import torch
from rnn import RNN

# Initialize model
model = RNN(vocab_size=5000, emb_dim=100, hidden_size=64, 
           num_layers=2, bidirectional=True, dropout=0.2)

# Sample input
batch_size, seq_len = 4, 15
input_tokens = torch.randint(1, 5000, (batch_size, seq_len))

# Forward pass
logits = model(input_tokens)
print(f"Output shape: {logits.shape}")  # (4,)

# Convert to probabilities
probabilities = torch.sigmoid(logits)
predictions = (probabilities > 0.5).long()
```

### LSTM Example

```python
import torch
from lstm import LSTM, GloVeLSTM

# Standard LSTM
lstm_model = LSTM(vocab_size=8000, emb_dim=200, hidden_dim=256, 
                 num_layers=2, bidirectional=True, dropout=0.5)

# GloVe LSTM with pretrained embeddings
# Assume you have loaded GloVe embeddings
pretrained_emb = torch.randn(8000, 300)  # Example: 300-dim GloVe
glove_model = GloVeLSTM(vocab_size=8000, emb_dim=300, hidden_dim=256,
                       num_layers=1, bidirectional=False, dropout=0.3,
                       pad_idx=0, pretrained_embeddings=pretrained_emb)

# Sample input
input_seq = torch.randint(1, 8000, (3, 20))  # 3 samples, 20 tokens each

# Forward pass
lstm_output = lstm_model(input_seq)
glove_output = glove_model(input_seq)

print(f"LSTM output: {lstm_output.shape}")
print(f"GloVe LSTM output: {glove_output.shape}")
```

## Performance Comparison

### Computational Complexity

| Model | Training | Prediction | Memory |
|-------|----------|------------|---------|
| Naive Bayes | O(n × d) | O(c × d) | O(c × d) |
| KNN | O(1) | O(n × d) | O(n × d) |
| DNN | O(epochs × batch × d) | O(d) | O(d × h) |
| RNN | O(epochs × batch × seq × h²) | O(seq × h²) | O(h²) |
| LSTM | O(epochs × batch × seq × h²) | O(seq × h²) | O(h²) |

Where:
- n = number of training samples
- d = feature dimension  
- c = number of classes
- h = hidden dimension
- seq = sequence length

### Model Characteristics

| Model | Pros | Cons | Best For |
|-------|------|------|----------|
| **Naive Bayes** | Fast, simple, works with small data | Strong independence assumption | Baseline, sparse features |
| **KNN** | No training needed, simple | Slow prediction, memory intensive | Small datasets, interpretability |
| **DNN** | Fast inference, moderate complexity | Loses sequence information | Medium-sized datasets |
| **RNN** | Captures sequences, moderate params | Slower than DNN, vanishing gradients | Sequential patterns |
| **LSTM** | Best long-term memory, handles sequences | Most complex, slowest training | Complex sequential patterns |

### Memory Usage (Sparse vs Dense)

```python
# Sparse matrix efficiency example
from scipy.sparse import csr_matrix
import numpy as np

# Dense matrix: 10,000 samples × 50,000 features × 4 bytes = ~2GB
dense_matrix = np.random.random((10000, 50000)).astype(np.float32)

# Sparse matrix: only non-zero values stored
# For text data: typically 0.1-1% density = ~20MB
sparse_matrix = csr_matrix(dense_matrix)
sparse_matrix.data = sparse_matrix.data[sparse_matrix.data > 0.99]  # Keep only 1%

print(f"Dense memory: {dense_matrix.nbytes / 1024**2:.1f} MB")
print(f"Sparse memory: {sparse_matrix.data.nbytes / 1024**2:.1f} MB")
```

### Training Tips

1. **Start Simple**: Begin with Naive Bayes for baseline
2. **Feature Engineering**: Quality features matter more than complex models
3. **Regularization**: Use dropout for neural networks
4. **Batch Size**: Larger batches for stable training
5. **Learning Rate**: Start with 1e-3, adjust based on convergence
6. **Early Stopping**: Monitor validation loss to prevent overfitting

### Hyperparameter Guidelines

```python
# Recommended starting parameters
models_config = {
    'naive_bayes': {'alpha': 1.0},
    'knn': {'k': 7, 'distance_metric': 'cosine'},
    'dnn': {'emb_dim': 128, 'hidden_dim': 128, 'dropout': 0.3},
    'rnn': {'hidden_size': 128, 'num_layers': 1, 'dropout': 0.3},
    'lstm': {'hidden_dim': 256, 'num_layers': 2, 'dropout': 0.5}
}
```