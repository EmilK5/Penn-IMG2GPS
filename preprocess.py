import os
import torch
import pandas as pd
from PIL import Image, ImageOps
from torchvision import transforms
from typing import Tuple


def preprocess_image(image: Image.Image) -> torch.Tensor:
    inference_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Phone uploads often store rotation in EXIF. Apply it before model inference
    # so the tensor matches the upright image a human sees.
    image = ImageOps.exif_transpose(image).convert("RGB")
    return inference_transform(image)


def prepare_data(path: str) -> Tuple[torch.Tensor, torch.Tensor]:

    # 1. Extract the folder path where the grader's CSV is located
    csv_directory = os.path.dirname(path)
    
    df = pd.read_csv(path)
    X = []
    y = []
    
    # 2. Iterate through the CSV
    for _, row in df.iterrows():
        img_path = os.path.join(csv_directory, row['file_name'])
        
        # Open the image
        image = Image.open(img_path)
        X.append(preprocess_image(image))
        
        y.append([row['Latitude'], row['Longitude']])
        
    # 3. Stack inputs
    X_tensor = torch.stack(X)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    return X_tensor, y_tensor
