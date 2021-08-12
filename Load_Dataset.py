import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import os
from torchvision.transforms import ToTensor, Lambda
import pandas as pd
from torchvision.io import read_image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


class CustomImageDataset(Dataset):
    def __init__(self, data_dir, transform=None, target_transform=None):

        self.data = pd.read_csv(data_dir).to_numpy()
        self.images = self.data[:, 1:]
        self.images = self.images.reshape((42000, 28, 28))
        self.img_labels = self.data[:, 0]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):

        image = self.images[idx, :, :]
        image = image.reshape((28, 28, 1))
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
