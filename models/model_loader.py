import os
import sys
import torch
import streamlit as st
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from .cluinet import UNetDecoder, UNetEncoder

HF_TOKEN = os.getenv("HF_TOKEN")  # Set in environment or Streamlit secrets
models_dir = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model(model_id):
    """
    Load a PyTorch enhancement model from Hugging Face private repos.

    Args:
        model_id (str): Identifier of the model to load

    Returns:
        torch.nn.Module or tuple: Loaded PyTorch model(s)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        if model_id == "spectroformer":
            model_path = hf_hub_download(
                repo_id="hahamemo/spectroformer",
                filename="spectroformer.pth",
                token=HF_TOKEN
            )
            model = torch.load(model_path, map_location='cpu')
            model.to(device).eval()
            return model, None

        elif model_id == "phaseformer":
            model_path = hf_hub_download(
                repo_id="hahamemo/phaseformer",
                filename="phaseformer_UIEB.pth",
                token=HF_TOKEN
            )
            model = torch.load(model_path, map_location='cpu')
            model.to(device).eval()
            return model, None

        elif model_id == "cluienet":
            fE_path = hf_hub_download(
                repo_id="hahamemo/cluienet",
                filename="cluie_fE_latest.pth",
                token=HF_TOKEN
            )
            fI_path = hf_hub_download(
                repo_id="hahamemo/cluienet",
                filename="cluie_fI_latest.pth",
                token=HF_TOKEN
            )

            fE = UNetEncoder().to(device)
            fl = UNetDecoder().to(device)
            fE.load_state_dict(torch.load(fE_path, map_location='cpu'))
            fl.load_state_dict(torch.load(fI_path, map_location='cpu'))

            fE.eval()
            fl.eval()
            return fE, fl

        elif model_id == "fish_detector":
            model_path = hf_hub_download(
                repo_id="hahamemo/fish-detection",
                filename="fish_yolov11.pt",
                token=HF_TOKEN
            )
            model = YOLO(model_path)
            model.to(device).eval()
            return model, None

        elif model_id == "coral_detector":
            model_path = hf_hub_download(
                repo_id="hahamemo/coral-detection",
                filename="coral_yolov11.pt",
                token=HF_TOKEN
            )
            model = YOLO(model_path)
            model.to(device).eval()
            return model, None

        else:
            raise ValueError(f"Unknown enhancement model identifier: {model_id}")

    except Exception as e:
        raise Exception(f"Failed to load model '{model_id}': {str(e)}")


@st.cache_resource
def load_detection_model(model_id):
    """
    Load a detection model from Hugging Face private repos.

    Args:
        model_id (str): Identifier of the detection model

    Returns:
        torch.nn.Module: Loaded model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        if model_id == "fish_detection":
            model_path = hf_hub_download(
                repo_id="hahamemo/fish-detection",
                filename="fish_yolov11.pt",
                token=HF_TOKEN
            )
            model = YOLO(model_path)
            model.to(device).eval()
            return model

        elif model_id == "coral_detection":
            model_path = hf_hub_download(
                repo_id="hahamemo/coral-detection",
                filename="coral_yolov11.pt",
                token=HF_TOKEN
            )
            model = YOLO(model_path)
            model.to(device).eval()
            return model

        else:
            raise ValueError(f"Unknown detection model identifier: {model_id}")

    except Exception as e:
        raise Exception(f"Failed to load detection model '{model_id}': {str(e)}")
