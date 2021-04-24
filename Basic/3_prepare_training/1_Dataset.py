import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from matplotlib import pyplot as plt

# 1. iris dataset -> torch dataset 으로 변환
from sklearn.datasets import load_iris

## iris load (numpy 의 ndarray)
iris = load_iris()

x = iris.data[:100]
y = iris.target[:100]

## ndarray -> tensor
x = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

## Dataset 생성
## - 텐서를 데이터셋으로 만들 때는 TensorDataset 생성
dataset = TensorDataset(x, y)

## DataLoader로 데이터 불러오기
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

## 모델 생성
model = nn.Linear(4, 1)
loss_fn = nn.BCEWithLogitsLoss() # 로지스틱회귀 적용
optimizer = optim.SGD(model.parameters(), lr=0.25)

# 학습
losses = []
for epoch in range(100):
    batch_loss = 0.0
    for xx, yy in dataloader:
        optimizer.zero_grad()
        y_pred = model(xx)
        loss = loss_fn(y_pred.view_as(yy), yy)
        loss.backward()
        optimizer.step()
        batch_loss += loss.item()
    losses.append(batch_loss)
    print(f"Epoch {epoch} loss: {batch_loss}")

plt.plot(losses)
plt.show()
