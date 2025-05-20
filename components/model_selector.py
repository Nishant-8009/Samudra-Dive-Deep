import streamlit as st
from config import ENHANCEMENT_MODELS, DETECTION_MODELS

def model_selector():
    """Component for selecting enhancement and detection models."""
    # Enhancement models section
    st.subheader("Step 1: Select Enhancement Models")
    
    # Create checkboxes for each enhancement model
    selected_enhancement_models = {}
    for model_name in ENHANCEMENT_MODELS.keys():
        selected_enhancement_models[model_name] = st.checkbox(f"{model_name}")
    
    # Get list of selected enhancement model names
    selected_enhancement_names = [model_name for model_name, selected in selected_enhancement_models.items() if selected]
    
    # Display selected enhancement models
    if selected_enhancement_names:
        st.write(f"Selected enhancement models: {', '.join(selected_enhancement_names)}")
    else:
        st.info("Please select at least one enhancement model")
    
    # Detection models section
    st.subheader("Step 2: Select Detection Models (applied after enhancement)")
    
    # Create checkboxes for each detection model
    selected_detection_models = {}
    for model_name in DETECTION_MODELS.keys():
        selected_detection_models[model_name] = st.checkbox(f"{model_name}")
    
    # Get list of selected detection model names
    selected_detection_names = [model_name for model_name, selected in selected_detection_models.items() if selected]
    
    # Display selected detection models
    if selected_detection_names:
        st.write(f"Selected detection models: {', '.join(selected_detection_names)}")
    else:
        st.info("Optional: Select detection models to analyze enhanced images")
    
    # Process button - enabled only if at least one enhancement model is selected
    process_button = st.button("Process Image", disabled=len(selected_enhancement_names) == 0)
    
    return selected_enhancement_names, selected_detection_names, process_button