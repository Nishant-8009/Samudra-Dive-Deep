import streamlit as st
from config import FOOTER_TEXT

def render_footer():
    """Render the application footer."""
    st.divider()
    st.markdown(f"<p style='text-align: center; color: gray;'>{FOOTER_TEXT}</p>", unsafe_allow_html=True)