import torch

device = torch.device("cpu")

# 创建一个指定size的随机矩阵
x = torch.rand((5, 3), device=device)
print(x)

# 创建一个指定数据的矩阵
y = torch.tensor([[5.2, 3], [3.2, 5.4]], dtype=torch.float, device=device)
print(y)

# 获取矩阵维度
print(y.size())
