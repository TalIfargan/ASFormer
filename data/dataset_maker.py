import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from functools import partial
from torchvision.transforms._presets import ImageClassification, InterpolationMode
from torchvision.models import EfficientNet_V2_S_Weights

transforms = EfficientNet_V2_S_Weights.IMAGENET1K_V1.transforms()




class CustomImageDataset(Dataset):
    def __init__(self, mapping_file, transform=transforms, target_transform=None, n_samples=48000):
        self.mapping = pd.read_csv(mapping_file).sample(n=n_samples, random_state=7)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx):
        img_path = self.mapping.iloc[idx, 0]
        image = read_image(img_path)
        label = self.mapping.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label