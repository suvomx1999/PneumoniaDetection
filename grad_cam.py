import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from models.transfer import TransferModel
from models.baseline import SimpleCNN

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hook for gradients and activations
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def __call__(self, x):
        self.model.eval()
        
        # Forward pass
        output = self.model(x)
        
        # Zero grads
        self.model.zero_grad()
        
        # Backward pass for the predicted class
        # For binary classification with 1 output node (logit), we backprop the logit directly if positive
        # or -logit if negative? Usually we care about the "Pneumonia" class which is 1.
        # So we maximize the output logit.
        
        score = output[:, 0]
        score.backward()
        
        # Generate heatmap
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        weights = np.mean(gradients, axis=(1, 2))
        
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (x.shape[2], x.shape[3]))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        
        return cam

def show_cam_on_image(img_path, mask, alpha=0.5):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = np.float32(img) / 255
    
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap * alpha + img * (1 - alpha) # Blend
    cam = cam / np.max(cam)
    
    cv2.imwrite("gradcam_result.jpg", np.uint8(255 * cam))
    print("Saved Grad-CAM visualization to gradcam_result.jpg")
    
    # Also save the original next to it
    combined = np.hstack((img, cam))
    cv2.imwrite("gradcam_combined.jpg", np.uint8(255 * combined))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Grad-CAM Visualization')
    parser.add_argument('--image_path', type=str, required=True, help='Path to input image')
    parser.add_argument('--model_path', type=str, required=True, help='Path to saved model checkpoint')
    parser.add_argument('--model_type', type=str, default='densenet121', choices=['simple_cnn', 'densenet121', 'resnet18'], help='Model architecture')
    
    args = parser.parse_args()
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
        
    # Load Model
    if args.model_type == 'simple_cnn':
        model = SimpleCNN()
        target_layer = model.conv4 # Last conv layer
    elif args.model_type == 'densenet121':
        model = TransferModel(model_name='densenet121')
        # Target last dense block
        target_layer = model.model.features.denseblock4
    elif args.model_type == 'resnet18':
        model = TransferModel(model_name='resnet18')
        target_layer = model.model.layer4[-1]
        
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    
    # Prepare Image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(args.image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Run Grad-CAM
    grad_cam = GradCAM(model, target_layer)
    mask = grad_cam(input_tensor)
    
    show_cam_on_image(args.image_path, mask)
