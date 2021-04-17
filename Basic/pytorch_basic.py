# 2. torch 사용법
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from sklearn.datasets import load_boston

# torch 버전 확인
torch.__version__

# 1. Torch Basic

# empty()
# - 초기화되지 않은 행렬을 선언
x = torch.empty(5, 3)   # 초기화 되지 않았으나, 사용전에는 명확한 값을 갖고 있지 않다.
                        # 때문에 행렬이 생성되면, 그 시점에 할당된 메모리에 존재하던 값들이 초기값으로 나타난다.
print(x)
# tensor([[0., 0., 0.],
#         [0., 0., 0.],
#         [0., 0., 0.],
#         [0., 0., 0.],
#         [0., 0., 0.]])

# rand()
# - 무작위로 초기화된 행렬 생성
x = torch.rand(5, 3)
print(x)
# tensor([[0.2630, 0.8361, 0.3703],
#         [0.5465, 0.0426, 0.3301],
#         [0.4426, 0.5271, 0.1465],
#         [0.1873, 0.4050, 0.2614],
#         [0.7108, 0.1801, 0.6386]])

# zeros()
# - dtype 은 long 이고, 0으로 채워진 행렬을 생성
x = torch.zeros(5, 3, dtype=torch.long)
print(x)
# tensor([[0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0]])

# tensor()
# - 데이터로부터 tensor를 직접 생성
x = torch.tensor([5.5, 3])
print(x)  # tensor([5.5000, 3.0000])

# new_ones(), randn_like()
# - new_* 는 메소드는 크기를 받는다.
x = x.new_ones(5, 3, dtype=torch.double)
print(x)
# tensor([[1., 1., 1.],
#         [1., 1., 1.],
#         [1., 1., 1.],
#         [1., 1., 1.],
#         [1., 1., 1.]], dtype=torch.float64)

# Tensor의 종류
## 0차원 텐서
x = torch.rand(10)
x.size()

## 1차원 텐서
y = torch.FloatTensor([21, 23, 25, 30.4, 24.0])
y.size()

## 2차원 텐서
boston = load_boston()
### sklearn import 중 DLL 에러 발생 시 : scipy, numpy 삭제 후 재설치
z = torch.from_numpy(boston.data)
z.size()

## 3차원 텐서
image = np.array(Image.open('image/panda.jpg').resize((224,224)))
image_tensor = torch.from_numpy(image)
image_tensor.size()

## 텐서 슬라이스
sales = torch.FloatTensor([1000.0, 323.2, 333.5, 555.6, 1000.0, 323.2, 333.5, 555.6])
sales[:5]

plt.imshow(image_tensor[25:175, 60:130, 0].numpy())

x = torch.randn_like(torch.empty(5, 3), dtype=torch.float)
print(x)
# tensor([[-1.1830, -0.8495, -0.9411],
#         [ 1.0873,  0.3467, -0.2882],
#         [ 0.0027, -1.0330,  0.6013],
#         [-1.5487,  0.8644,  2.4071],
#         [ 1.7357, -0.5980, -0.9386]])

print(x.size())  # torch.Size([5, 3])

# 2. Operation
x = torch.rand(5, 3)
y = torch.rand(5, 3)
print(x + y)
print(torch.add(x, y))
# tensor([[1.5237, 0.7899, 0.7905],
#         [1.3973, 0.4686, 1.4151],
#         [0.4069, 0.6827, 0.5377],
#         [1.0886, 0.4240, 1.5000],
#         [0.4294, 1.7310, 1.7816]])

res = torch.empty(5, 3)
torch.add(x, y, out=res)
print(res)
# tensor([[1.5237, 0.7899, 0.7905],
#         [1.3973, 0.4686, 1.4151],
#         [0.4069, 0.6827, 0.5377],
#         [1.0886, 0.4240, 1.5000],
#         [0.4294, 1.7310, 1.7816]])

print(res[:, 1])
# tensor([0.7899, 0.4686, 0.6827, 0.4240, 1.7310])

# broadcast
m1 = torch.FloatTensor([[3., 3.]])
m2 = torch.FloatTensor([[2., 2.]])
print(m1 + m2)

m1 = torch.FloatTensor([[3., 3.]])
m2 = torch.FloatTensor([2])
print(m1 + m2)

v1 = torch.FloatTensor([[1, 2]])
v2 = torch.FloatTensor([[3], [4]])
print(v1 + v2)

# squeeze & unsqueeze
## squeeze
ft = torch.FloatTensor([[0], [1], [2]])
print(ft)
print(ft.shape)

print(ft.squeeze())
print(ft.squeeze().shape)

## unsqueeze
ft = torch.FloatTensor([0, 1, 2])
print(ft.shape)

print(ft.unsqueeze(0))  # 0번째에 1인 차원을 추가한다.
print(ft.unsqueeze(0).shape)

# concatenate
x = torch.FloatTensor([[1, 2], [3, 4]])
y = torch.FloatTensor([[5, 6], [7, 8]])
print(torch.cat([x, y], dim=0))
print(torch.cat([x, y], dim=1))


# view()
# - tensor의 크기, 모양을 변경하는 경우에
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)
print(x.size(), y.size(), z.size())
# torch.Size([4, 4]) torch.Size([16]) torch.Size([2, 8])

# item()
# - tensor 에 값이 1개만 있다면 item 을 이용해 확인할 수 있다.
x = torch.randn(1)
print(x)  # tensor([-0.8641])
print(x.item())  # -0.8640572428703308

# 3. Numpy 변환
a = torch.ones(5)
print(a)

b = a.numpy()
print(b)

a.add_(1)
print(a)
print(b)

import numpy as np

a = np.ones(5)
b = torch.from_numpy(a)

np.add(a, 1, out=a)

print(a)
print(b)


# 장치 이동하기
x = torch.randn(1)
print(x)
print(x.item())

## CUDA 장치 객체로 GPU 상에서 직접 tensor를 생성하는 경우 혹은 .to 를 이용해 CUDA 환경으로 옮기는 방법이 있다.
if torch.cuda.is_available():
    device = torch.device("cuda")
    y = torch.ones_like(x, device=device)
    x = x.to(device)
    z = x+y

    print(z)
    print(z.to("cpu", torch.double))
print(device)

# 2. Autograd
import numpy
import torch
from torch.autograd import Variable

x = Variable(torch.ones(2, 2), requires_grad=True)
y = x.mean()

y.backward()

x.grad
x.grad_fn  # 출력 없음
x.data
y.grad_fn


# 3. 저수준 API 신경망
import numpy as np
import pandas as pd

import torch
from torch.autograd import Variable

## 데이터 생성 함수
def get_data():
    data_x = np.asarray([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.18, 7.59, 2.17, 7.04, 10.7, 5.314, 5.75, 9.3, 3.1, 5.6])
    data_y = np.asarray([1.7, 2.76, 2.09, 3.19, 1.69, 1.573, 3.35, 2.6, 2.54, 1.32, 2.94, 3.675, 1.65, 2.904, 2.53, 2.97, 1.6])
    data_type = torch.FloatTensor

    x = Variable(torch.from_numpy(data_x).type(data_type), requires_grad=False).view(17, 1)
    y = Variable(torch.from_numpy(data_y).type(data_type), requires_grad=False).view(17)

    return x, y

## 가중치 생성 함수
def get_weight():
    w = Variable(torch.randn(1), requires_grad=True)
    b = Variable(torch.randn(1), requires_grad=True)

    return w, b


## 신경망 모델
def network_api():
    y_pred = torch.matmul(x, w) + b

    return y_pred

def loss_fn(y, y_pred):
    loss = (y - y_pred).pow(2).sum()
    for param in [w, b]:
        if param.grad is not None:
            param.grad.data.zero_()

    loss.backward()

    return loss.data

## 최적화
def optimize(learning_rate):
    w.data -= learning_rate * w.grad.data
    b.data -= learning_rate * b.grad.data

x, y = get_data()
w, b = get_weight()
learning_rate = 0.01

print(w, b)

y_pred = network_api()
loss_fn(y, y_pred)
optimize(learning_rate)

print(w, b)



