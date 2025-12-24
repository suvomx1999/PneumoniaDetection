import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os
from models.transfer import TransferModel
from models.baseline import SimpleCNN

# Page Config
st.set_page_config(page_title="Pneumonia Detection AI", page_icon="🫁", layout="centered")

# Title
st.title("🫁 Pneumonia Detection from Chest X-Rays")
st.markdown("""
This AI model detects Pneumonia from Chest X-Ray images.
**Upload an image** to get a prediction and visualization.
""")

# Sidebar
st.sidebar.header("Model Settings")
model_type = st.sidebar.selectbox("Choose Model Architecture", ["densenet121", "resnet18", "simple_cnn"])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@st.cache_resource
def load_model(model_type):
    if model_type == 'simple_cnn':
        model = SimpleCNN()
    else:
        model = TransferModel(model_name=model_type)
    
    # Load checkpoint if exists, otherwise load fresh (for demo purposes if no checkpoint)
    # In a real app, you'd force a checkpoint load or fail.
    checkpoint_path = f"checkpoints/best_model_{model_type}.pth"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded {checkpoint_path}")
    else:
        st.warning(f"Checkpoint for {model_type} not found. Using untrained weights (predictions will be random).")
    
    model = model.to(device)
    model.eval()
    return model

model = load_model(model_type)

def process_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device)

def get_gradcam(model, input_tensor, model_type):
    # Determine target layer
    if model_type == 'simple_cnn':
        target_layer = model.conv4
    elif model_type == 'densenet121':
        target_layer = model.model.features.denseblock4
    elif model_type == 'resnet18':
        target_layer = model.model.layer4[-1]
    
    gradients = None
    activations = None

    def save_activation(module, input, output):
        nonlocal activations
        activations = output

    def save_gradient(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    # Register hooks
    handle_f = target_layer.register_forward_hook(save_activation)
    handle_b = target_layer.register_full_backward_hook(save_gradient)

    # Forward
    model.zero_grad()
    output = model(input_tensor)
    
    # Backward
    score = output[:, 0]
    score.backward()
    
    # Remove hooks
    handle_f.remove()
    handle_b.remove()

    if gradients is None or activations is None:
        return None

    # Generate Heatmap
    grads = gradients.cpu().data.numpy()[0]
    acts = activations.cpu().data.numpy()[0]
    
    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    
    for i, w in enumerate(weights):
        cam += w * acts[i]
        
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam - np.min(cam)
    cam = cam / (np.max(cam) + 1e-8)
    
    return cam

# File Uploader
uploaded_file = st.file_uploader("Upload Chest X-Ray (JPEG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    image = Image.open(uploaded_file).convert('RGB')
    with col1:
        st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # Prediction
    if st.button("Analyze Image"):
        with st.spinner("Analyzing..."):
            input_tensor = process_image(image)
            
            with torch.no_grad():
                output = model(input_tensor)
                prob = torch.sigmoid(output).item()
                prediction = "Pneumonia" if prob > 0.5 else "Normal"
                confidence = prob if prob > 0.5 else 1 - prob
            
            # Display Result
            if prediction == "Pneumonia":
                st.error(f"**Prediction: {prediction}**")
            else:
                st.success(f"**Prediction: {prediction}**")
            
            st.metric("Confidence Score", f"{confidence:.2%}")
            
            # Grad-CAM Visualization
            st.subheader("Explainability (Grad-CAM)")
            try:
                # We need to enable grad for Grad-CAM
                with torch.set_grad_enabled(True):
                    # Re-forward for gradcam
                    cam_mask = get_gradcam(model, input_tensor, model_type)
                
                if cam_mask is not None:
                    # Prepare heatmap overlay
                    img_np = np.array(image.resize((224, 224)))
                    img_np = np.float32(img_np) / 255
                    
                    heatmap = cv2.applyColorMap(np.uint8(255 * cam_mask), cv2.COLORMAP_JET)
                    heatmap = np.float32(heatmap) / 255
                    
                    # RGB conversion for display
                    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                    
                    cam_result = heatmap * 0.4 + img_np * 0.6
                    cam_result = cam_result / np.max(cam_result)
                    
                    with col2:
                        st.image(cam_result, caption='Grad-CAM Heatmap', use_container_width=True)
                else:
                    st.warning("Could not generate Grad-CAM visualization.")
            except Exception as e:
                st.error(f"Error generating Grad-CAM: {e}")

# Footer
st.markdown("---")
st.markdown("Disclaimer: This tool is for educational purposes only and not for medical diagnosis.")
