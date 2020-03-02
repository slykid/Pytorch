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
