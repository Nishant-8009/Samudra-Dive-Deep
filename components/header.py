# /home/ace/Work/CVPRLab/website/components/header.py
import streamlit as st
from config import HEADER_TITLE, HEADER_SUBTITLE
import os
from PIL import Image

# Helper function to display logos consistently and handle errors
def display_logo(container, logo_path, width=80):
    """Checks for logo, loads, and displays it in the given container."""
    if os.path.exists(logo_path):
        try:
            logo_img = Image.open(logo_path)
            # Display directly using the container's image method
            container.image(logo_img, width=width)
        except Exception as e:
            # Show a concise error in the UI
            container.error(f"Err: {os.path.basename(logo_path)}")
            # Log the full error to the console for debugging
            print(f"Error loading/displaying logo {logo_path}: {e}")
    else:
        # Show a concise warning in the UI
        container.warning(f"NF: {os.path.basename(logo_path)}")
        # Log the warning to the console
        print(f"Logo not found at: {logo_path}")

def render_header():
    """Render the application header with specified logo layout."""
    iitrpr_logo_path = "assets/iitrpr_logo.png"
    moes_logo_path = "assets/Ministry_of_Earth_Sciences.jpg"
    noit_logo_path = "assets/NIOT_LOGO_H_E.png"

    col1, col2, col3, col4 = st.columns([1.5, 4, 5, 1.5]) 


    with col1:
        if os.path.exists(iitrpr_logo_path):
            try:
                original_logo = Image.open(iitrpr_logo_path)
                original_logo = original_logo.convert("RGBA")
                white_bg = Image.new("RGBA", original_logo.size, "WHITE")
                white_bg.paste(original_logo, (0, 0), original_logo)
                st.image(white_bg, width=130) 
            except Exception as e:
                st.error(f"Err: {os.path.basename(iitrpr_logo_path)}")
                print(f"Error loading/displaying logo {iitrpr_logo_path}: {e}")
        else:
            st.warning(f"NF: {os.path.basename(iitrpr_logo_path)}")
            print(f"Logo not found at: {iitrpr_logo_path}")


    # Column 2: Title and Subtitle
    with col2:
        st.title(HEADER_TITLE)
        st.write(HEADER_SUBTITLE)

    with col3:

        display_logo(st, moes_logo_path, width=468)

    with col4:

        display_logo(st, noit_logo_path, width=150)

    st.divider()

