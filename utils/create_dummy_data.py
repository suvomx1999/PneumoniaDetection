import os
import numpy as np
from PIL import Image
from tqdm import tqdm

def create_dummy_data(base_dir, num_images=20):
    classes = ['Normal', 'Pneumonia']
    
    for cls in classes:
        cls_dir = os.path.join(base_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        
        print(f"Generating {num_images} dummy images for {cls}...")
        for i in tqdm(range(num_images)):
            # Generate random noise image
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            
            save_path = os.path.join(cls_dir, f'img_{i}.jpg')
            img.save(save_path)

if __name__ == "__main__":
    data_dir = "./data"
    create_dummy_data(data_dir)
