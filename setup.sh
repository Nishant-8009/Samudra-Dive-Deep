#!/bin/bash

# Create necessary directories
mkdir -p temp
mkdir -p outputs
mkdir -p assets

# Install requirements
pip install -r requirements.txt

echo "Setup completed! Run the app with: streamlit run app.py"