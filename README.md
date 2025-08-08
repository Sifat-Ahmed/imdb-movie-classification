## Installation & Usage

### Local Development
```bash
pip install -r requirements.txt
python main.py  # Train models and run evaluation
```

### API Server
```bash
cd src/api
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Docker Deployment
```bash
docker build -t sentiment-api .
docker run -p 8000:8000 sentiment-api
```

## API Endpoints
- `POST /predict` - Single review prediction
- `POST /predict/batch` - Batch predictions
- `GET /health` - Health check
- `GET /models` - Available models info

## Model Performance
The enhanced implementation includes:
- **Naive Bayes**: Improved with proper smoothing and feature selection
- **KNN**: Fixed distance calculations and optimized for text data