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
from matplotlib import pyplot as plt

from torch.utils.data import DataLoader
import DogCatDataset as dogcat

os.environ['KMP_DUPLICATE_LIB_OK']='True'
path = "data\\dogs_vs_cats"

datasets = dogcat.DogCatDataset(root_path=path)
dataloader = DataLoader(datasets, batch_size=10, shuffle=True)

for image, label in dataloader:
    plt.imshow(image)
    print(label)