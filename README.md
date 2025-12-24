# Pneumonia Detection from Chest X-Ray Images

## Problem Statement
Pneumonia is an infection that inflames the air sacs in one or both lungs. Diagnosing pneumonia from chest X-rays is a challenging task that requires expert radiologists. This project aims to build an end-to-end Deep Learning solution to automatically detect Pneumonia from Chest X-Ray images, assisting medical professionals in rapid screening.

## Dataset
The dataset consists of Chest X-Ray images classified into two categories:
- **Normal**: Healthy lungs
- **Pneumonia**: Lungs showing signs of pneumonia (bacterial or viral)

Images are resized to 224x224 pixels for model compatibility.

## Project Structure
```
.
├── models/
│   ├── baseline.py       # Simple CNN from scratch
│   └── transfer.py       # Transfer Learning (DenseNet121, ResNet18)
├── utils/
│   ├── dataset.py        # Data loading, augmentation, splitting
│   └── callbacks.py      # Early stopping
├── train.py              # Training loop
├── evaluate.py           # Evaluation script
├── grad_cam.py           # Explainability visualization
├── requirements.txt      # Dependencies
└── README.md             # Project documentation
```

## Model Architecture

### 1. Baseline Model (Simple CNN)
A custom Convolutional Neural Network trained from scratch.
- 4 Convolutional Blocks (Conv2D -> BN -> ReLU -> MaxPool)
- Flatten -> Fully Connected Layers -> Output

### 2. Transfer Learning
- **DenseNet121**: Pre-trained on ImageNet. Modified classifier for binary classification. Chosen for its efficiency and feature reuse, which is beneficial for medical imaging.
- **ResNet18**: Option available for lighter weight model.

## Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start (Demo)
If you don't have the dataset yet, you can generate dummy data to verify the code:
```bash
python utils/create_dummy_data.py
python train.py --data_dir ./data --model densenet121 --epochs 1
```

## Usage

### Dataset Preparation
You can structure your data in two ways:
1. **Single Folder**: `data/Normal` and `data/Pneumonia`. The script will automatically split it into train/val/test.
2. **Split Folders**: `data/train/Normal`, `data/train/Pneumonia`, etc.

### Training
To train the model (e.g., DenseNet121):
```bash
python train.py --data_dir /path/to/dataset --model densenet121 --epochs 20 --batch_size 32
```
For the simple CNN:
```bash
python train.py --data_dir /path/to/dataset --model simple_cnn
```

### Evaluation
To evaluate the trained model:
```bash
python evaluate.py --data_dir /path/to/dataset --model_path checkpoints/best_model_densenet121.pth --model_type densenet121
```

### Explainability (Grad-CAM)
To visualize the regions the model is focusing on:
```bash
python grad_cam.py --image_path /path/to/image.jpg --model_path checkpoints/best_model_densenet121.pth --model_type densenet121
```

### Frontend (Web App)
To launch the interactive web application:
```bash
streamlit run app.py
```
This will open a local web server where you can upload images and see predictions in real-time.

## Results
*Note: Run evaluation to populate specific metrics.*
- **Metrics Evaluated**: Accuracy, Precision, Recall, F1-Score, ROC-AUC.
- **Focus**: High Recall is prioritized to minimize false negatives (missing a pneumonia case).

## Ethical Considerations
- **Bias**: The model is trained on a specific dataset and may not generalize to all demographics or equipment types.
- **Clinical Use**: This tool is for **educational and research purposes only**. It is NOT an FDA-approved medical device and should not be used for primary diagnosis without radiologist supervision.
- **Data Privacy**: Ensure patient data is anonymized before use.
