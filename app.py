import streamlit as st
from components.header import render_header
from components.footer import render_footer
from components.image_upload import image_uploader
from components.model_selector import model_selector
from components.results_display import display_results
import os
from config import TITLE, DESCRIPTION, ENHANCED_DIR

# Page configuration
st.set_page_config(
    page_title=TITLE,
    page_icon="🌊",
    layout="wide"
)

def main():
    # Create necessary directories if they don't exist
    os.makedirs("temp", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    os.makedirs(ENHANCED_DIR, exist_ok=True)
    
    # Render header
    render_header()
    
    # Main content
    st.write(DESCRIPTION)
    
    # Sidebar for image upload and model selection
    with st.sidebar:
        st.header("Controls")
        # Image upload component
        uploaded_image = image_uploader()
        
        if uploaded_image is not None:
            # Model selection
            selected_enhancement_models, selected_detection_models, process_button = model_selector()
    
    # Main area for displaying images and results
    if 'uploaded_image' in locals() and uploaded_image is not None:
        # Display the uploaded image
        st.subheader("Uploaded Image")
        st.image(uploaded_image, caption="Original Image", width=800)
        
        # Process image when button is clicked
        if 'process_button' in locals() and process_button and selected_enhancement_models:
            display_results(uploaded_image, selected_enhancement_models, selected_detection_models)
    else:
        # Display placeholder when no image is uploaded
        st.info("👈 Please upload an image using the sidebar to get started")
    
    # Render footer
    render_footer()

if __name__ == "__main__":
    main()