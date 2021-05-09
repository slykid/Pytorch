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

import torch
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

# 각 데이터 별 데이터셋 생성
cat_train = dogcat.DogCatDataset(root_path=path, mode="train", image_kind="cat", transform=transform)
cat_valid = dogcat.DogCatDataset(root_path=path, mode="valid", image_kind="cat", transform=transform)
dog_train = dogcat.DogCatDataset(root_path=path, mode="train", image_kind="dog", transform=transform)
dog_valid = dogcat.DogCatDataset(root_path=path, mode="valid", image_kind="dog", transform=transform)

def imshow(input):
    input = input.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    input = std * input + mean
    input = np.clip(input, 0, 1)
    plt.imshow(input)

# 이미지 확인
imshow(cat_train[0][0])
imshow(cat_valid[0][0])
imshow(dog_train[0][0])
imshow(dog_valid[0][0])

# 학습 데이터 셋, 검증 데이터 셋 생성
train_dataset = torch.utils.data.ConcatDataset([cat_train, dog_train])
valid_dataset = torch.utils.data.ConcatDataset([cat_valid, dog_valid])

print(f"number of train dataset : {len(train_dataset)}\nnumber of valid dataset : {len(valid_dataset)}")

# 데이터 로더 생성
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=True)

# 데이터 로더 동작확인
samples, labels = iter(train_dataloader).next()
fig = plt.figure(figsize=(16,24))
for i in range(24):
    a = fig.add_subplot(4, 6, i + 1)
    a.set_title(labels[i])
    a.axis('off')
    a.imshow(np.transpose(samples[i].numpy(), (1,2,0)))
    plt.subplots_adjust(bottom=0.2, top=0.6, hspace=0)

# 모델링



