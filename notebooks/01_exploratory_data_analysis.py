"""
Exploratory Data Analysis for CRM Sales Opportunities

This notebook provides initial exploration of the CRM dataset to understand:
- Data structure and quality
- Feature distributions
- Target variable patterns
- Business insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up plotting
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

# Configuration
DATA_PATH = Path("../data/raw")
PROCESSED_PATH = Path("../data/processed")

print("CRM Sales Opportunities - Exploratory Data Analysis")
print("=" * 50)

# Load data
try:
    # Look for CSV files in raw data directory
    csv_files = list(DATA_PATH.glob("*.csv"))
    if csv_files:
        data_file = csv_files[0]
        print(f"Loading data from: {data_file}")
        df = pd.read_csv(data_file)
    else:
        print("No CSV files found in data/raw directory.")
        print("Please run the data ingestion pipeline first:")
        print("python -m src.data.ingestion.crm_ingestion")
        df = None
except Exception as e:
    print(f"Error loading data: {e}")
    df = None

if df is not None:
    # Basic information about the dataset
    print(f"\n📊 Dataset Overview")
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print(f"\n📋 Column Information")
    print(df.dtypes)
    
    print(f"\n🔍 First few rows:")
    print(df.head())
    
    print(f"\n📈 Statistical Summary:")
    print(df.describe(include='all'))
    
    print(f"\n❓ Missing Values:")
    missing_data = df.isnull().sum()
    missing_pct = (missing_data / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing_data,
        'Missing %': missing_pct
    })
    print(missing_df[missing_df['Missing Count'] > 0])
    
    # Data quality checks
    print(f"\n🔍 Data Quality Checks:")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print(f"Completely empty rows: {df.isnull().all(axis=1).sum()}")
    
    # If we have specific columns, do targeted analysis
    if 'close_value' in df.columns:
        print(f"\n💰 Close Value Analysis:")
        print(f"Min: ${df['close_value'].min():,.2f}")
        print(f"Max: ${df['close_value'].max():,.2f}")
        print(f"Mean: ${df['close_value'].mean():,.2f}")
        print(f"Median: ${df['close_value'].median():,.2f}")
    
    if 'deal_stage' in df.columns:
        print(f"\n📊 Deal Stage Distribution:")
        stage_counts = df['deal_stage'].value_counts()
        print(stage_counts)
        print(f"\nDeal Stage Percentages:")
        print((stage_counts / len(df) * 100).round(2))
    
    # Visualizations would go here if this were a real notebook
    print(f"\n✅ EDA Summary completed!")
    print(f"📝 Key findings:")
    print(f"   - Dataset has {df.shape[0]:,} rows and {df.shape[1]} columns")
    print(f"   - Missing data in {missing_df[missing_df['Missing Count'] > 0].shape[0]} columns")
    print(f"   - {df.duplicated().sum()} duplicate rows found")
    
else:
    print("\n❌ Cannot perform EDA without data.")
    print("Please ensure the dataset is available in the data/raw directory.")
