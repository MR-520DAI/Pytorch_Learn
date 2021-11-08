import torch

device = torch.device("cpu")

x = torch.tensor([[1, 1], [2, 2]], dtype=torch.int8, device=device)

y = torch.tensor([[2, 3], [3, 2]], dtype=torch.int8, device=device)

# 加法操作
print(x + y)
print(torch.add(x, y))

# 乘法操作
print(x * y)

# 矩阵乘法操作
print(torch.matmul(x, y))
