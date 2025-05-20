import streamlit as st
import os
import time
from PIL import Image
from config import ENHANCEMENT_MODELS, DETECTION_MODELS, OUTPUT_DIR, ENHANCED_DIR
from models.inference import process_image, apply_detection

def display_results(input_image, selected_enhancement_models, selected_detection_models=[]):
    """Display and manage the results from sequential model processing."""
    if not selected_enhancement_models:
        st.warning("No enhancement models selected. Please select at least one model.")
        return
    
    # Create directory for enhanced images if it doesn't exist
    os.makedirs(ENHANCED_DIR, exist_ok=True)
    
    st.subheader("Processing Results")
    
    # Show processing message for enhancement models
    with st.spinner(f"Enhancing image with {len(selected_enhancement_models)} model(s)..."):
        # Process the image with each selected enhancement model
        enhancement_results = {}
        
        # Create a progress bar for enhancement
        progress_bar = st.progress(0)
        progress_step = 1.0 / (len(selected_enhancement_models) * 100)
        progress_value = 0
        
        # Process each enhancement model
        for i, model_name in enumerate(selected_enhancement_models):
            st.write(f"Processing with {model_name}...")
            
            # Get the model identifier
            model_id = ENHANCEMENT_MODELS[model_name]
            
            try:
                # Process the image with the current enhancement model
                result_image_path = process_image(input_image, model_id, ENHANCED_DIR)
                enhancement_results[model_name] = result_image_path
                
                # Simulate processing time with progress updates
                for j in range(100):
                    progress_value += progress_step
                    progress_bar.progress(min(progress_value, 1.0))
                    time.sleep(0.01)
                
            except Exception as e:
                st.error(f"Error processing image with {model_name}: {str(e)}")
        
        # Display enhancement results
        if enhancement_results:
            st.success(f"✅ Image enhancement complete! Generated {len(enhancement_results)} enhanced images.")
            
            # Display each enhancement result
            for model_name, result_path in enhancement_results.items():
                with st.expander(f"Enhanced with {model_name}", expanded=True):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Original image
                        st.image(input_image, caption="Original Image", use_column_width=True)
                    
                    with col2:
                        # Enhanced image
                        if os.path.exists(result_path):
                            result_image = Image.open(result_path)
                            st.image(result_image, caption=f"{model_name} Result", use_column_width=True)
                            
                            # Add download button for the enhanced image
                            with open(result_path, "rb") as file:
                                btn = st.download_button(
                                    label=f"Download Enhanced Image",
                                    data=file,
                                    file_name=os.path.basename(result_path),
                                    mime="image/png",
                                    key=f"download_enhanced_{model_name}"
                                )
                        else:
                            st.error(f"Failed to generate enhanced image for {model_name}.")
        else:
            st.error("Failed to generate any enhancement results.")
            return
    
    # Process with detection models if any are selected
    if selected_detection_models and enhancement_results:
        st.subheader("Detection Results")
        
        with st.spinner(f"Applying {len(selected_detection_models)} detection model(s) to enhanced images..."):
            # For each enhanced image, apply all selected detection models
            for enhancement_model, enhanced_image_path in enhancement_results.items():
                st.write(f"## Detection results on images enhanced with {enhancement_model}")
                
                # Load the enhanced image
                enhanced_image = Image.open(enhanced_image_path)
                
                # Process with each detection model
                for detection_model_name in selected_detection_models:
                    st.write(f"Applying {detection_model_name} to image enhanced with {enhancement_model}...")
                    
                    # Get the detection model identifier
                    detection_model_id = DETECTION_MODELS[detection_model_name]
                    
                    try:
                        # Apply detection to the enhanced image
                        detection_result_path = apply_detection(enhanced_image_path, detection_model_id)
                        
                        # Display the detection result
                        if os.path.exists(detection_result_path):
                            detection_result_image = Image.open(detection_result_path)
                            
                            # Show both the enhanced image and the detection result
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.image(enhanced_image, caption=f"Enhanced with {enhancement_model}", use_column_width=True)
                            
                            with col2:
                                st.image(detection_result_image, 
                                         caption=f"{detection_model_name} on {enhancement_model}", 
                                         use_column_width=True)
                                
                                # Add download button for the detection result
                                with open(detection_result_path, "rb") as file:
                                    btn = st.download_button(
                                        label=f"Download {detection_model_name} Result",
                                        data=file,
                                        file_name=os.path.basename(detection_result_path),
                                        mime="image/png",
                                        key=f"download_{enhancement_model}_{detection_model_name}"
                                    )
                        else:
                            st.error(f"Failed to generate detection result for {detection_model_name}.")
                    
                    except Exception as e:
                        st.error(f"Error applying {detection_model_name} to enhanced image: {str(e)}")

