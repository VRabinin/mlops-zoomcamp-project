# CRM Win Probability Predictor - Streamlit App

A web application for predicting the probability of winning sales opportunities using machine learning.

## Features

### 🎯 Individual Opportunity Prediction
- Interactive form for entering opportunity details
- Real-time win probability prediction
- Expected revenue calculation
- Risk assessment and recommendations
- Visual probability gauge

### 📊 Sales Pipeline Overview
- Analysis of all open opportunities
- Batch predictions for the entire pipeline
- Expected revenue forecasting
- Probability distribution charts
- Top opportunities ranking

### 🤖 Model Information
- Model details and performance metrics
- Usage guidelines and best practices
- Feature importance insights

## Quick Start

### Prerequisites
- MLflow server running at `http://localhost:5005`
- Trained and registered `monthly_win_probability_model`
- Python environment with required dependencies

### Running the Application

**Option 1: Using Makefile (Recommended)**
```bash
# Development mode with auto-reload
make streamlit-dev

# Production mode
make streamlit-app
```

**Option 2: Direct Command**
```bash
# Activate environment
source .venv/bin/activate

# Set Python path and run
PYTHONPATH=${PYTHONPATH}:$(pwd) streamlit run src/streamlit_app/app.py
```

The application will be available at: **http://localhost:8501**

## Application Pages

### 1. Single Prediction
Enter details for a specific opportunity:
- **Opportunity Details**: ID, deal value, engagement date
- **Sales Context**: Agent, product, account selection
- **Account Information**: Revenue, employees, industry sector
- **Timeline**: Estimated sales cycle duration

Get instant predictions with:
- Win probability percentage
- Expected revenue calculation
- Risk level assessment
- Actionable recommendations

### 2. Pipeline Overview
Analyze your entire sales pipeline:
- **Summary Metrics**: Total opportunities, pipeline value, expected revenue
- **Probability Distribution**: Opportunities categorized by win likelihood
- **Revenue Forecasting**: Expected revenue by probability category
- **Top Opportunities**: Ranked list of highest probability deals

### 3. Model Information
Learn about the prediction model:
- Model architecture and features
- Performance metrics and validation
- Usage guidelines and limitations
- Best practices for sales teams

## Key Features Explained

### Win Probability Calculation
The model considers multiple factors:
- **Historical Performance**: Agent and product success rates
- **Deal Characteristics**: Value, sales cycle, timing
- **Account Context**: Size, industry, relationship history
- **Market Factors**: Seasonality, competition, urgency

### Risk Assessment
Opportunities are categorized as:
- 🟢 **Low Risk** (>70% win probability): High priority, likely to close
- 🟡 **Medium Risk** (40-70%): Needs attention and intervention
- 🔴 **High Risk** (<40%): Review qualification and strategy

### Recommendations Engine
Based on win probability, the app provides:
- **High Priority**: Maintain momentum, prepare proposals
- **Medium Priority**: Address objections, strengthen value proposition
- **Low Priority**: Re-evaluate qualification, consider alternatives

## Technical Architecture

### Data Flow
1. **Input Processing**: Form data → feature engineering
2. **Model Prediction**: MLflow model → probability scores
3. **Result Analysis**: Probability → risk assessment → recommendations
4. **Visualization**: Charts, gauges, and tables

### Model Integration
- **MLflow Integration**: Automatic model loading from registry
- **Feature Engineering**: Real-time calculation of prediction features
- **Caching**: Optimized performance with Streamlit caching
- **Error Handling**: Graceful handling of missing data and model errors

## Configuration

### Environment Variables
The app uses these default settings:
- **MLflow URI**: `http://localhost:5005`
- **Model Name**: `monthly_win_probability_model`
- **Port**: `8501`

### Data Requirements
The app expects:
- CRM features dataset at `data/features/crm_features.csv`
- Trained model registered in MLflow
- Historical data for agent/product performance calculation

## Troubleshooting

### Common Issues

**Model Not Found**
```
Error loading model: ...
```
- Ensure MLflow server is running
- Verify model is trained and registered
- Check model name matches configuration

**Data Loading Errors**
```
Error loading data: ...
```
- Verify `data/features/crm_features.csv` exists
- Check file permissions and format
- Ensure all required columns are present

**Feature Errors**
```
Error making prediction: ...
```
- Check input data completeness
- Verify feature engineering pipeline
- Review model feature requirements

### Performance Tips
- **Caching**: Model and data are cached for better performance
- **Input Validation**: Enter complete and accurate information
- **Batch Processing**: Use Pipeline Overview for multiple predictions
- **Regular Updates**: Retrain model periodically with new data

## Development

### Adding New Features
1. **Model Features**: Update `get_feature_columns()` function
2. **UI Components**: Add new Streamlit components and forms
3. **Visualizations**: Create new Plotly charts and dashboards
4. **Business Logic**: Extend prediction and recommendation logic

### Testing
```bash
# Test the application
python -m pytest tests/test_streamlit_app.py

# Run with coverage
pytest --cov=src/streamlit_app tests/test_streamlit_app.py
```

### Deployment
For production deployment:
1. Configure production MLflow endpoint
2. Set up proper authentication
3. Use container deployment (Docker)
4. Configure load balancing and scaling

## Integration with MLOps Platform

This Streamlit app is part of the larger MLOps platform:
- **Data Pipeline**: Feeds from CRM data processing
- **Model Training**: Uses models from training pipeline
- **Monitoring**: Integrates with model performance monitoring
- **Deployment**: Part of the overall deployment strategy

For more information, see the main project documentation.
