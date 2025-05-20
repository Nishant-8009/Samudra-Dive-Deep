import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from models.model_loader import load_model, load_detection_model
from config import OUTPUT_DIR
import uuid
    
def run_spectroformer(input_tensor, model):
    final_l = model(input_tensor)[0]
    final_l = final_l.detach().squeeze(0).cpu()
    
    return final_l

def to_img(x,wide,hig):
    """Convert the tanh (-1 to 1) ranged tensor to image (0 to 1) tensor	"""
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, wide, hig)
    return x

def process_image(input_image, model_id, output_dir=OUTPUT_DIR):
    """
    Process an input image with the specified model.

    Args:
        input_image (PIL.Image): Input image to process
        model_id (str): Model identifier
        output_dir (str): Directory to save the output image

    Returns:
        str: Path to the saved output image
    """
    try:
        # Generate a unique filename for the output
        output_filename = f"{model_id}_{uuid.uuid4().hex}.png"
        output_path = os.path.join(output_dir, output_filename)

        # Load the model
        print(f"Loading model: {model_id}")
        model, model_2 = load_model(model_id)
        model.eval()
        

        # Device configuration
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        model.to(device) # Move model to the correct device

        output_image = None # Initialize output PIL image

        # Perform inference based on the model type
        if model_id == "spectroformer":
            print("Processing with Spectroformer...")

            transform_list = [transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            transform = transforms.Compose(transform_list)
            
            input_image = input_image.convert("RGB").resize((512,512), Image.BICUBIC)

            input_tensor = transform(input_image).unsqueeze(0).to(device)
            print(f"Input tensor shape: {input_tensor.shape}, device: {input_tensor.device}")


            with torch.no_grad():
                print("Running model inference...")
                raw_output = model(input_tensor)


                if isinstance(raw_output, (list, tuple)):
                    output_tensor_gpu = raw_output[0]
                else:
                    output_tensor_gpu = raw_output 

                print(f"Raw output tensor shape (GPU): {output_tensor_gpu.shape}")


                output_tensor_cpu = output_tensor_gpu.detach().squeeze(0).cpu()
                print(f"Processed output tensor shape (CPU): {output_tensor_cpu.shape}")

            output_tensor_cpu = output_tensor_cpu * 0.5 + 0.5

            output_tensor_cpu = torch.clamp(output_tensor_cpu, 0, 1)

            # Convert tensor to PIL Image
            to_pil_transform = transforms.ToPILImage()
            output_image = to_pil_transform(output_tensor_cpu)
            print("Converted output tensor to PIL image.")
        
        elif model_id == "phaseformer":
            print("Processing with Spectroformer...")

            transform_list = [transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            transform = transforms.Compose(transform_list)
            
            input_image = input_image.convert("RGB").resize((512,512), Image.BICUBIC)

            input_tensor = transform(input_image).unsqueeze(0).to(device)
            print(f"Input tensor shape: {input_tensor.shape}, device: {input_tensor.device}")


            with torch.no_grad():
                print("Running model inference...")
                raw_output = model(input_tensor)


                if isinstance(raw_output, (list, tuple)):
                    output_tensor_gpu = raw_output[0]
                else:
                    output_tensor_gpu = raw_output 

                print(f"Raw output tensor shape (GPU): {output_tensor_gpu.shape}")


                output_tensor_cpu = output_tensor_gpu.detach().squeeze(0).cpu()
                print(f"Processed output tensor shape (CPU): {output_tensor_cpu.shape}")

            output_tensor_cpu = output_tensor_cpu * 0.5 + 0.5

            output_tensor_cpu = torch.clamp(output_tensor_cpu, 0, 1)

            # Convert tensor to PIL Image
            to_pil_transform = transforms.ToPILImage()
            output_image = to_pil_transform(output_tensor_cpu)
            print("Converted output tensor to PIL image.")
        
        elif model_id == "cluienet":
            print("Processing with CLUIE-Net...")

            transform_list = [transforms.ToTensor()]
            transform = transforms.Compose(transform_list)
            
            input_tensor = transform(input_image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                print("Running model inference... ")
                
                # model is fE (encoder), model_2 is fl (decoder)
                fE_out, enc_outs = model(input_tensor)
                # Call the decoder (model_2) with the encoder output and skip connections
                decoder_output_tensor = model_2(fE_out, enc_outs)
                # Convert to image range [0, 1] and move to CPU. Shape: [1, C, H, W]
                output_tensor_processed = to_img(decoder_output_tensor.cpu(), input_tensor.shape[2], input_tensor.shape[3])
                
            # Convert tensor to PIL Image
            to_pil_transform = transforms.ToPILImage()
            # Remove the batch dimension (dim 0) before converting. Shape becomes [C, H, W]
            output_image = to_pil_transform(output_tensor_processed.squeeze(0))
            print("Converted output tensor to PIL image.")
            
        elif model_id == "fish_detector":
            with torch.no_grad():
                print("Running fish detection...")
                # input_tensor = Image.open(input_image)
                yolo_outputs = model.predict(input_image)
                output = yolo_outputs[0]
                boxes = output.boxes
                names = output.names
                
                # Print detection results
                for j in range(len(boxes)):
                    label = names[boxes.cls[j].item()]
                    coordinates = boxes.xyxy[j].tolist()
                    confidence = np.round(boxes.conf[j].item(), 2)

                    print(f'Fish {j + 1} is: {label}')
                    print(f'Coordinates are: {coordinates}')
                    print(f'Confidence is: {confidence}')
                    print('-------')

                # Get the annotated image (BGR to RGB)
                annotated_image = output.plot()[:, :, ::-1]
                
                # Convert numpy array to PIL Image
                output_image = Image.fromarray(annotated_image)
                
        elif model_id == "coral_detector":
            with torch.no_grad():
                print("Running coral detection...")
                # input_tensor = Image.open(input_image)
                yolo_outputs = model.predict(input_image)
                output = yolo_outputs[0]
                boxes = output.boxes
                names = output.names
                
                # Print detection results
                for j in range(len(boxes)):
                    label = names[boxes.cls[j].item()]
                    coordinates = boxes.xyxy[j].tolist()
                    confidence = np.round(boxes.conf[j].item(), 2)

                    print(f'Coral {j + 1} is: {label}')
                    print(f'Coordinates are: {coordinates}')
                    print(f'Confidence is: {confidence}')
                    print('-------')

                # Get the annotated image (BGR to RGB)
                annotated_image = output.plot()[:, :, ::-1]
                
                # Convert numpy array to PIL Image
                output_image = Image.fromarray(annotated_image)

        else:
            raise ValueError(f"Inference not implemented for model: {model_id}")

        # Save the final output image if it was generated
        if output_image:
            print(f"Saving output image to: {output_path}")
            output_image.save(output_path)
            return output_path
        else:
             raise RuntimeError(f"Output image was not generated for model {model_id}.")

    except Exception as e:
        print(f"Error during inference for model {model_id}: {str(e)}")

        raise Exception(f"Error during inference for model {model_id}: {str(e)}")


def apply_detection(enhanced_image_path, detection_model_id):
    """
    Apply object detection to an enhanced image.

    Args:
        enhanced_image_path (str): Path to enhanced image
        detection_model_id (str): Detection model identifier
        
    Returns:
        str: Path to the saved output image with detection results
    """
    
    try:
        # Load the enhanced image
        enhanced_image = Image.open(enhanced_image_path) 
        
        output_filename = f"{detection_model_id}_{uuid.uuid4().hex}.png"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        # Load the detection model
        print(f"Loading detection model: {detection_model_id}")
        detection_model = load_detection_model(detection_model_id)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Perform detection
        with torch.no_grad():
            if detection_model_id == "fish_detection":
                print("Running fish detection...")
                yolo_outputs = detection_model.predict(enhanced_image)
                output = yolo_outputs[0]
                boxes = output.boxes
                names = output.names
                
                # Print detection results
                for j in range(len(boxes)):
                    label = names[boxes.cls[j].item()]
                    coordinates = boxes.xyxy[j].tolist()
                    confidence = np.round(boxes.conf[j].item(), 2)

                    print(f'Fish {j + 1} is: {label}')
                    print(f'Coordinates are: {coordinates}')
                    print(f'Confidence is: {confidence}')
                    print('-------')

                # Get the annotated image (BGR to RGB)
                annotated_image = output.plot()[:, :, ::-1]
                
                # Convert numpy array to PIL Image
                output_image = Image.fromarray(annotated_image)
                
            elif detection_model_id == "coral_detection":
                print("Running coral detection...")
                # input_tensor = Image.open(input_image)
                yolo_outputs = detection_model.predict(enhanced_image)
                output = yolo_outputs[0]
                boxes = output.boxes
                names = output.names
                
                # Print detection results
                for j in range(len(boxes)):
                    label = names[boxes.cls[j].item()]
                    coordinates = boxes.xyxy[j].tolist()
                    confidence = np.round(boxes.conf[j].item(), 2)

                    print(f'Coral {j + 1} is: {label}')
                    print(f'Coordinates are: {coordinates}')
                    print(f'Confidence is: {confidence}')
                    print('-------')

                # Get the annotated image (BGR to RGB)
                annotated_image = output.plot()[:, :, ::-1]
                
                # Convert numpy array to PIL Image
                output_image = Image.fromarray(annotated_image)
                
            else:
                raise ValueError(f"Detection not implemented for model: {detection_model_id}")
        
        # Save the final output image
        if output_image:
            print(f"Saving output image with detections to: {output_path}")
            output_image.save(output_path)
            return output_path
        else:
            raise RuntimeError(f"Output image was not generated for detection model {detection_model_id}.")
    
    except Exception as e:
        print(f"Error during detection for model {detection_model_id}: {str(e)}")
        raise Exception(f"Error during detection for model {detection_model_id}: {str(e)}")