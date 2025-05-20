import os
import sys # <-- Import sys
import torch
import streamlit as st
from ultralytics import YOLO
from .cluinet import UNetDecoder, UNetEncoder

models_dir = os.path.dirname(os.path.abspath(__file__))

# Cache the loaded models to prevent reloading
@st.cache_resource
def load_model(model_id):
    """
    Load a PyTorch enhancement model based on the model identifier.

    Args:
        model_id (str): Identifier of the model to load

    Returns:
        torch.nn.Module: Loaded PyTorch model
    """
    # --- Add models directory to sys.path ---
    original_sys_path = list(sys.path) # Store original path
    if models_dir not in sys.path:
        print(f"Temporarily adding {models_dir} to sys.path for torch.load")
        sys.path.insert(0, models_dir)
        added_path = True
    else:
        added_path = False
    # ---------------------------------------

    try:
        # Check if CUDA is available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if model_id == "spectroformer":
            print("Attempting torch.load for spectroformer.pth...")
            # Now torch.load should be able to find the 'model_without_CA' module
            spectroformer = torch.load("models/checkpoints/spectroformer.pth", map_location='cpu')
            print("torch.load successful.")
            spectroformer.to(device)
            spectroformer.eval()
            return spectroformer, None
        
        elif model_id == "phaseformer":
            phaseformer = torch.load("models/checkpoints/phaseformer_UIEB.pth", map_location='cpu')
            phaseformer.to(device)
            phaseformer.eval()
            return phaseformer, None
        
        elif model_id == "cluienet":
            fE = UNetEncoder().to(device)
            fl = UNetDecoder().to(device)
            fE.load_state_dict(torch.load("models/checkpoints/cluie_fE_latest.pth", map_location='cpu'))
            fl.load_state_dict(torch.load("models/checkpoints/cluie_fI_latest.pth", map_location='cpu'))
            fE.eval()
            fl.eval()
            return fE, fl
        
        elif model_id == "fish_detector":
            fish_model = YOLO("models/checkpoints/fish_yolov11.pt")
            fish_model.to(device)
            fish_model.eval()
            return fish_model, None
        
        elif model_id == "coral_detector":
            coral_model = YOLO("models/checkpoints/coral_yolov11.pt")
            coral_model.to(device)
            coral_model.eval()
            return coral_model, None

        else:
            raise ValueError(f"Unknown enhancement model identifier: {model_id}")
            
    except Exception as e:
        raise Exception(f"Failed to load enhancement model {model_id}: {str(e)}")
    
@st.cache_resource
def load_detection_model(model_id):
    """
    Load a PyTorch detection model based on the model identifier.

    Args:
        model_id (str): Identifier of the model to load

    Returns:
        torch.nn.Module: Loaded PyTorch model
    """
    original_sys_path = list(sys.path) # Store original path
    if models_dir not in sys.path:
        print(f"Temporarily adding {models_dir} to sys.path for torch.load")
        sys.path.insert(0, models_dir)
        added_path = True
    else:
        added_path = False
    # ---------------------------------------

    try:
        # Check if CUDA is available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if model_id == "fish_detection":
            fish_model = YOLO("models/checkpoints/fish_yolov11.pt")
            fish_model.to(device)
            fish_model.eval()
            return fish_model
        
        elif model_id == "coral_detection":
            coral_model = YOLO("models/checkpoints/coral_yolov11.pt")
            coral_model.to(device)
            coral_model.eval()
            return coral_model

        else:
            raise ValueError(f"Unknown enhancement model identifier: {model_id}")
            
    except Exception as e:
        raise Exception(f"Failed to load enhancement model {model_id}: {str(e)}")
