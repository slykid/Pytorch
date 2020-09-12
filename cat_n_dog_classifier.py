from glob import glob
import os
import sys
import time
import numpy as np
import shutil

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch import optim

from torch.autograd import Variable

from torch.optim import lr_scheduler

from torchvision import transforms
from torchvision import models
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid

if os.path.isdir('logs') is False:
    os.mkdir('logs')
f = open('logs/20200912.txt', 'w')

path = 'image\\dogs-vs-cats\\dogandcat'  # Windows 이므로 / 대신 \\ 으로 디렉터리 구분함
file = glob(os.path.join(path, "*/*.jpg"))  # glob 함수 : 특정 폴더 이하의 모든 파일을 반환함

num_shape = len(file)
print('Total number of images {len(file)}: ', num_shape)
f.write('Total number of images: {}'.format(num_shape))

if num_shape > 0:
    shuffle = np.random.permutation(num_shape)

    os.mkdir(os.path.join(path, 'valid'))

    for i in ['train', 'valid']:
        for folder in ['dog\\', 'cat\\']:
            os.mkdir(os.path.join(path, i, folder))

    # valid
    for i in shuffle[:2000]:
        folder = file[i].split('\\')[-1].split('.')[0]
        image = file[i].split('\\')[-1]
        os.rename(file[i], os.path.join(path, 'valid', folder, image))

    # train
    for i in shuffle[2000:]:
        folder = file[i].split('\\')[-1].split('.')[0]
        image = file[i].split('\\')[-1]
        os.rename(file[i], os.path.join(path, 'train', folder, image))

# GPU 가동 상태 확인
if torch.cuda.is_available():
    is_cuda = True
    print("\nCuda is available\n")
    f.write("\nCuda is available\n")

# 사진을 Pytorch Tensor 로 로드
# - 사용할 이미지 크기에 대한 설정
# - 정규화 수행
# -
transform = transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
print(transform)
f.write(str(transform) + "\n")

# ImageFolder : 폴더 단위로 데이터가 정렬 되어있고,
#               이미지파일명도 특정 카테고리로 정렬되어 저장되어 있는 경우
#               관련 레이블과 함께 이미지를 메모리에 올린다.
# 아래의 작업을 수행하기 전에 반드시 이미지의 크기는 모두 동일하게 맞춰주어야 한다.
train = ImageFolder(os.path.join(path,'train'),transform)
valid = ImageFolder(os.path.join(path,'valid'),transform)

# ImageFolder 로 데이터를 로드하면, 이미지의 폴더명 별로 클래스가 생성된다.
# - class_to_idx : 데이터 셋에서 사용하는 각 분류 클래스에 대응되는 인덱스 정보를 저장
# - classes : 분류 클래스 정보를 저장
print(train)
print(train.class_to_idx)
print(train.classes)

f.write(str(train.class_to_idx))
f.write(str(train.classes))

# 이미지 출력 함수
def imshow(inp):
    inp = inp.numpy().transpose((1, 2, 0)) # 컬러는 R,G,B 라는 3개의 레이어를 갖기 때문에 3차원으로 텐서를 numpy 배열로 변환
    mean = np.array([[0.405, 0.456, 0.406]])
    std = np.array([0.229, 0.224, 0.225])

    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)

imshow(train[0][0])

# Data Generator 객체 생성
train_gen = torch.utils.data.DataLoader(train, batch_size=64, num_workers=3)
valid_gen = torch.utils.data.DataLoader(valid, batch_size=64, num_workers=3)

dataset_sizes = {'train':len(train_gen.dataset), 'valid':len(valid_gen.dataset)}
dataloaders = {'train':train_gen, 'valid':valid_gen}

# NeuralNet model architecture
model = models.resnet18(pretrained=True)
num_fits = model.fc.in_features
model.fc = nn.Linear(num_fits, 2)

if is_cuda:
    model = model.cuda()

print(model)
f.write(str(model))

# Model Training
learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
# scheduler = optimizer.step()

# train_model 함수 생성
def train_model(model, criterion, optimizer, scheduler, num_epochs=5):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        f.write('Epoch {}/{} \n'.format(epoch, num_epochs - 1))
        f.write('-n' * 10)
        f.write("\n")

        # 각 Epoch은 학습 단계와 검증 단계를 거침
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # 학습 모드 설정
            else:
                model.train(False)  # 검증 모드 설정

            running_loss = 0.0
            running_corrects = 0

            # 데이터 반
            for data in dataloaders[phase]:
                # 입력 데이터 가져오기
                inputs, labels = data

                # 데이터를 Vaariable로 만듦
                if torch.cuda.is_available():
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # 파라미터 기울기 초기화
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # 학습 단계에서만 수행, 역전파 + 옵티마이즈(최적화)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # 통계
                running_loss += loss.data
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss.item() / dataset_sizes[phase]
            epoch_acc = running_corrects.item() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            f.write('{} Loss: {:.4f} Acc: {:.4f}\n'.format(
                phase, epoch_loss, epoch_acc))

            # 모델 복사(Deep Copy)
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    f.write('\nTraining complete in {:.0f}m {:.0f}s\n'.format(
        time_elapsed // 60, time_elapsed % 60))
    f.write('Best val Acc: {:4f}\n\n'.format(best_acc))

    # 최적의 모델 가중치 로딩
    model.load_state_dict(best_model_wts)
    return model

model = train_model(model, criterion, optimizer, scheduler, num_epochs=24)

f.close()
print("Close File")


