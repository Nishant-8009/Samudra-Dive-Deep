# Application settings
TITLE = "IITRPR CVPR Lab - SAMUDRA"
DESCRIPTION = """
Upload an image and apply various machine learning models to transform it.
You can select multiple models to apply simultaneously and compare their results side by side.
"""

# Header settings
HEADER_TITLE = "SAMUDRA"
HEADER_SUBTITLE = "Dive Deeper, See Clearer"

# Footer settings
FOOTER_TEXT = "© 2025 IIT Ropar CVPR Lab. All rights reserved."

# Enhancement Models configuration
ENHANCEMENT_MODELS = {
    "Spectroformer: Underwater Image Enhancement": "spectroformer",
    "Phaseformer: Underwater Image Restoration": "phaseformer",
    "CLUIE-Net : Underwater Image Enhancement": "cluienet",
    "Fish Detector": "fish_detector",
    "Coral Detector": "coral_detector",
}

# Detection Models configuration (applied after enhancement)
DETECTION_MODELS = {
    "Fish Detection": "fish_detection",
    "Coral Detection": "coral_detection",
}

# Paths
TEMP_DIR = "temp"
OUTPUT_DIR = "outputs"
ENHANCED_DIR = "enhanced"  # Directory for storing enhanced images before detection