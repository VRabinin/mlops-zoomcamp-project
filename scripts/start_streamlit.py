#!/usr/bin/env python3
"""
Streamlit App Launcher

Simple launcher script for the CRM Win Probability Prediction Streamlit app.
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Launch the Streamlit application"""
    
    # Get project root directory
    project_root = Path(__file__).parent.parent
    app_path = project_root / "src" / "streamlit_app" / "app.py"
    
    if not app_path.exists():
        print(f"Error: App file not found at {app_path}")
        sys.exit(1)
    
    # Launch Streamlit
    cmd = [
        sys.executable, "-m", "streamlit", "run", 
        str(app_path),
        "--server.port", "8501",
        "--server.address", "0.0.0.0",
        "--theme.base", "light"
    ]
    
    print("🚀 Starting CRM Win Probability Predictor...")
    print(f"📊 App will be available at: http://localhost:8501")
    print("🔧 Make sure MLflow server is running at http://localhost:5005")
    print()
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n✋ Application stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
