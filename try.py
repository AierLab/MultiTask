import torch

# 假设的梯度张量
gradA = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
gradB = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
gradC = torch.tensor([[0.2, 0.3], [0.1, 0.4]])

# 权重
weight_A = 0.8
weight_B = 0.1
weight_C = 0.1

# 假设的掩码
maskA = torch.tensor([[True, False], [False, True]])
maskB = torch.tensor([[False, True], [True, False]])
maskC = torch.tensor([[True, True], [False, False]])

# 初始化 grad_total
grad_total = torch.zeros_like(gradA)

# 计算替换
for i in range(grad_total.size(0)):  # 遍历第一个维度
    # 计算结果
    result = (
        weight_A * gradA[i] * maskA[i].float() +
        weight_B * gradB[i] * maskB[i].float() +
        weight_C * gradC[i] * maskC[i].float()
    )

    # 替换 grad_total[i] 中 maskA[i] 为 True 的位置
    grad_total[i][maskA[i]] = result[maskA[i]]

print("Updated grad_total:")
print(grad_total)
