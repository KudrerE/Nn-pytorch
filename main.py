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
from Neural_Network import NeuralNetwork
from Load_Dataset import CustomImageDataset
from Neural_Network import train_loop
from Neural_Network import test_loop

learning_rate = 1e-3
batch_size = 64
epochs = 5

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

digit_dataset = CustomImageDataset(
    r"C:\Users\kudre\Desktop\mavhine_try_data\train.csv",
    transform=ToTensor())

train, validate = torch.utils.data.random_split(digit_dataset, [38000, 4000],
                                                generator=torch.Generator().manual_seed(42))



train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(validate, batch_size=batch_size, shuffle=True)

epochs = 100

for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
