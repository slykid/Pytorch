import torch
from torch.autograd import Variable
from torch.nn import Linear

# 단일 레이어
data = Variable(torch.randn(1, 10))
Layer = Linear(in_features=10, out_features=5, bias=True)
Layer(data)

Layer.weight
Layer.bias

# 다중 레이어
data = Variable(torch.randn(1, 10))
Layer1 = Linear(10, 5)
Layer2 = Linear(5, 2)
Layer2(Layer1(data))

# ReLU
from torch.nn import ReLU
import torch

data = torch.Tensor([[1, 2, -1, -2]])
relu = ReLU()
relu(data)

# Feedforward
import torch
from torch.autograd import Variable
from torch import nn


class Network_ex(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Network_ex).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self,input):
        out = self.layer1(input)
        out = nn.ReLU(out)
        out = self.layer2(out)

        return out

# 오차 함수
# - MSE
loss = nn.MSELoss()
data = Variable(torch.randn(3, 5), requires_grad=True)
target = Variable(torch.randn(3, 5))
output = loss(data, target)
output.backward()

def cross_entropy(true_label, prediction):
    if true_label == 1:
        return -log(prediction)
    else:
        return -log(1 - prediction)

loss = nn.CrossEntropyLoss()
input = Variable(torch.randn(3, 5), requires_grad=True)
target = Variable(torch.LongTensor(3).random_(5))
output = loss(input, target)
output.backward()

import torch.optim as optim
optimizer = optim.SGD(model.parameters(), lr = 0.01)

for input, target in dataset:
  optimizer.zero_grad()
  output = model(input)
  loss = loss_fn(output, target)
  loss.backward()
  optimizer.step()