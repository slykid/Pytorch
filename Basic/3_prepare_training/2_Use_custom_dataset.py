import torch
import torch.nn
from torch.utils.data import DataLoader
import Custom_Dataset as custom

# 예시 1.
dataset = custom.CustomDataset()
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

for epoch in range(20):
    for idx, samples in enumerate(dataloader):
        print(idx)
        print(samples)


# 예시 2.
import os
import numpy as np
from matplotlib import pyplot as plt

from torch.utils.data import DataLoader
from torchvision import transforms

import DogCatDataset as dogcat

os.environ['KMP_DUPLICATE_LIB_OK']='True'
path = "data\\dogs_vs_cats"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 데이터 셋, 데이터 로더 생성
datasets = dogcat.DogCatDataset(root_path=path, transform=transform)
dataloader = DataLoader(datasets, batch_size=10, shuffle=True)

for image, label in dataloader:
    pass
