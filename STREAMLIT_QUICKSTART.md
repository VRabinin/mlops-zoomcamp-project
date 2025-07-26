# Streamlit App Quick Start Guide

## 🚀 Getting Started

### Prerequisites
1. **MLflow server running**: `make dev-start` 
2. **Trained model registered**: Run the training notebook first
3. **Python environment activated**: `source .venv/bin/activate`

### Launch Application

```bash
# Development mode (recommended)
make streamlit-dev

# Production mode
make streamlit-app

# Direct command
PYTHONPATH=${PYTHONPATH}:$(pwd) streamlit run src/streamlit_app/app.py
```

**🌐 Access URL**: http://localhost:8501

## 📊 Application Features

### 1. Single Prediction Page
- **Input Form**: Enter opportunity details
- **Instant Prediction**: Get win probability in real-time
- **Risk Assessment**: Automatic recommendations
- **Visual Gauge**: Interactive probability display

### 2. Pipeline Overview Page
- **Batch Analysis**: All open opportunities
- **Summary Metrics**: Total value, expected revenue
- **Probability Charts**: Distribution visualizations
- **Top Opportunities**: Ranked by win probability

### 3. Model Information Page
- **Model Details**: Architecture and performance
- **Usage Guidelines**: Best practices for sales teams
- **Limitations**: Important considerations

## 🎯 Key Use Cases

### Sales Team Daily Workflow
1. **Morning Review**: Check Pipeline Overview for priorities
2. **Opportunity Assessment**: Use Single Prediction for new deals
3. **Resource Allocation**: Focus on high-probability opportunities
4. **Risk Management**: Address low-probability deals

### Sales Manager Insights
- **Forecasting**: Probability-weighted revenue predictions
- **Team Performance**: Agent success rate analysis
- **Deal Prioritization**: Focus resources effectively
- **Risk Mitigation**: Early intervention strategies

## 🛠️ Troubleshooting

### Common Issues

**App Won't Start**
```bash
# Check Python environment
source .venv/bin/activate

# Verify MLflow is running
curl http://localhost:5005/health

# Check for port conflicts
lsof -i :8501
```

**Model Loading Errors ("Unable to locate credentials")**
```bash
# Start all services including MinIO
make application-start

# Verify MinIO is accessible
curl http://localhost:9000/minio/health/live

# Check if MinIO credentials are correct in docker-compose.yml
# Default: minioadmin/minioadmin
```

**Model Not Found**
```bash
# Verify model is registered
PYTHONPATH=$(pwd) python -c "
import mlflow
mlflow.set_tracking_uri('http://localhost:5005')
client = mlflow.MlflowClient()
models = client.search_registered_models()
print([m.name for m in models])
"

# If no models found, train one first:
# 1. Open notebooks/03_monthly_win_probability_prediction.ipynb
# 2. Run all cells to train and register model
```

**Data Loading Issues**
```bash
# Check data file exists
ls -la data/features/crm_features.csv

# Verify file format
head -5 data/features/crm_features.csv
```

### Performance Tips
- **Caching**: Model and data are cached for better performance
- **Input Validation**: Enter complete and accurate information
- **Batch Processing**: Use Pipeline Overview for multiple predictions
- **Regular Updates**: Retrain model periodically with new data

### Technical Notes

**MLflow + MinIO Integration**
The app automatically handles MinIO credentials for MLflow artifact access:
- **MinIO Endpoint**: http://localhost:9000
- **Credentials**: minioadmin/minioadmin (development default)
- **S3 Compatibility**: Uses AWS SDK with MinIO endpoint override

**Environment Variables Set Automatically**:
```python
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin  
AWS_DEFAULT_REGION=us-east-1
MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
```

## 🔧 Development

### Project Structure
```
src/streamlit_app/
├── app.py              # Main application
├── README.md           # Detailed documentation
└── __init__.py         # Module initialization
```

### Adding Features
1. **New Input Fields**: Update form sections in `create_single_prediction_input()`
2. **Additional Charts**: Add visualizations in `show_pipeline_overview()`
3. **Custom Logic**: Extend prediction logic in `display_prediction_results()`

### Testing Changes
```bash
# Development mode with auto-reload
make streamlit-dev

# Test specific functions
python -m pytest tests/test_streamlit_app.py
```

## 📈 Integration

### MLOps Platform Integration
- **Data Source**: Uses CRM features from data pipeline
- **Model Registry**: Loads latest model from MLflow
- **Monitoring**: Logs prediction requests and results
- **Deployment**: Part of containerized MLOps stack

### Production Deployment
- **Container**: Dockerfile for production deployment
- **Scaling**: Multiple app instances behind load balancer
- **Security**: Authentication and authorization layers
- **Monitoring**: Application performance monitoring

---

**💡 Pro Tip**: Keep the app running during development and it will auto-reload when you make changes to the code!

**🔗 Quick Links**:
- App: http://localhost:8501
- MLflow: http://localhost:5005
- Documentation: [src/streamlit_app/README.md](README.md)
