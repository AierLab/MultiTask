import torch
import matplotlib.pyplot as plt
import os

# 读取 .pth 文件
file_path = '/home/4paradigm/Weather/masks_ori/ori_60/maskA_epoch0.pth'
data = torch.load(file_path)

# 假设文件中的数据是一个字典或列表，你可以打印 data 来查看结构
# print(data)

# 假设数据是一个字典，并且 maskA 存储在字典的 'maskA' 键下
# 如果是列表，直接按索引获取一个 tensor，假设是列表的第一个元素
# 根据实际数据结构来调整以下代码
if isinstance(data, dict) and 'maskA' in data:
    maskA = data['maskA']
elif isinstance(data, list):
    maskA = data[0]  # 假设数据是一个列表，取第一个 tensor
else:
    raise ValueError("数据结构不符合预期")

# 选择其中一个 tensor，假设是第一个 tensor
# import pdb;pdb.set_trace()
tensor = maskA[6]  # 如果 maskA 是一个列表，则选择第一个元素

# 如果 tensor 是多维的，需要选取一个通道或者将其转换为二维图像
# 假设是 3 通道图像或者单通道图像，将其转换为 NumPy 数组
tensor = tensor[0][2].cpu().detach().numpy() # 转换为 H x W x C

# 可视化 tensor 并保存为图像
output_path = '/home/4paradigm/WGWS-Net/maskA_image.png'
plt.imshow(tensor)
plt.axis('off')  # 不显示坐标轴
plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
plt.show()

print(f"Image saved at {output_path}")
