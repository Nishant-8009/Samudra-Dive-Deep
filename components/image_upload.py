import streamlit as st
import os
from PIL import Image
import numpy as np
from utils.image_utils import save_uploaded_image

def image_uploader():
    """Component for uploading and preprocessing images."""
    st.subheader("Upload Image")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
            # Read the image
            image = Image.open(uploaded_file)
            
            # Save the uploaded image to temp directory
            image_path = save_uploaded_image(image, uploaded_file.name)
            
            st.success(f"Image uploaded successfully!")
            
            return image
            
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            return None
    
    return None