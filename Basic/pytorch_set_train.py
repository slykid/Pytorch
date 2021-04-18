import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn
from torch.utils.data import DataLoader
import cat_and_dog_dataset as catdog

os.environ['KMP_DUPLICATE_LIB_OK']='True'
path = 'data\\dogs_vs_cats\\org'

dataset = catdog.cat_dog_dataset(path)
dataloader = DataLoader(dataset, batch_size=32, num_workers=2)

for image, labels in dataloader:
    # plt.imshow(image)
    print(labels)