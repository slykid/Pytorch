# 2. torch 사용법
import torch

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
import torch

x = torch.ones(2, 2, requires_grad=True)
print(x)

## function class
y = x + 2
print(y.grad_fn)

z = y * y * 3
out = z.mean()

print(z.out)
print(z.grad_fn)

## requires_grad_
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)  # 별도로 입력값을 설정하지 않았으므로, 기본 값인 False 를 사용

a.requires_grad_(True)  # 입력값을 True로 설정했기 때문에 출력 시 requires_grad 옵션 값이 True로 설정됨
print(a.requires_grad)

b = (a * a).sum()
print(b.grad_fn)

## feedforward
import torch

# torch 버전 확인
torch.__version__

x = torch.FloatTensor(2, 2)
y = torch.FloatTensor(2, 2)
y.requires_grad_(True)

z = (x + y) + torch.FloatTensor(2, 2)

# with torch.no_grad():
#     z = (x + y) + torch.FloatTensor(2, 2)

## simple linear function
def linear(x, W, b):
    y = torch.mm(x, W) + b

    return y

x = torch.FloatTensor(16, 10)
W = torch.FloatTensor(10, 5)
b = torch.FloatTensor(5)

y = linear(x, W, b)
print(y)

## nn.Module
import torch.nn as nn

## proto_type
# class myLinear(nn.Module):
#     def __init__(self, input_size, output_size):
#         super().__init__()
#         self.W = nn.Parameter(torch.FloatTensor(input_size, output_size), requires_grad=True)
#         self.b = nn.Parameter(torch.FloatTensor(output_size), requires_grad=True)
#
#     def forward(self, x):
#         y = torch.mm(x, self.W) + self.b
#         return y

# final version
class myLinear(nn.Module):
    def __init__(self, input_size, output_size):
        super(myLinear, self).__init__()

        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        y = self.linear(x)

        return y


x = torch.FloatTensor(16, 10)
Linear = myLinear(10, 5)
y = Linear(x)

print(y)

## Check param
params = [p.size() for p in Linear.parameters()]
print(params)

## BackPropagation
realValue = 100

x = torch.FloatTensor(16, 10)
linear = myLinear(10, 5)
y = Linear(x)

loss = (realValue - y.sum())**2
loss.backward()

linear.eval()

linear.train()

## Make Model : Linear Regression
import torch
import torch.nn as nn

# 모델 선언
class MyModel(nn.Module):

    # 초기화
    def __init__(self, input_size, output_size):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    # 로직 정의
    def forward(self, x):
        # self.linear.cuda()  # GPU 사용 시 해제 후 사용 가능
        y = self.linear(x)

        return y

# 실제 값 계산 함수 정의
def ground_truth(x):
    return 3 * x[:, 0] + x[:, 1] - 2 * x[:, 2]

# 학습 함수 정의
def train(model, x, y, opt):

    # 경사하강법 파라미터 초기화
    opt.zero_grad()

    # 모델 학습 (FeedForward)
    y_pred = model(x)

    # Loss 계산 (MSE)
    loss = ((y - y_pred) ** 2).sum() / x.size(0)

    # 역전파 (BackPropagation)
    loss.backward()

    # 최적화 수행
    opt.step()

    return loss.data

# 하이퍼파라미터 정의
batch_size = 1   # 배치 크기
n_epochs = 1000  # 학습 횟수
n_iter = 1000    # 학습 1회 시 반복 횟수

model = MyModel(3, 3)
opt = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.1)  # SGD

# 모델 확인
print(model)

# 학습
for epoch in range(n_epochs):
    avg_loss = 0  # MSE

    for i in range(n_iter):
        x = torch.rand(batch_size, 3)
        y = ground_truth(x.data)

        loss = train(model, x, y, opt)

        avg_loss += loss
        avg_loss = avg_loss / n_iter

    x_valid = torch.FloatTensor([[.3, .2, .1]])
    y_valid = ground_truth(x_valid.data)

    model.eval()
    y_pred = model(x_valid)
    model.train

    print('Loss :  %.3f, Real : %.3f, Pred : %.3f' % (avg_loss, y_valid[0], y_pred[0, 0]))

    if avg_loss < 0.001:
        break
