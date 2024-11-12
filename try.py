import torch

def ensure_consistent_mask(mask_list):
    """确保 mask 列表中的每个 tensor 元素与前 3 个 tensor 的相应位置保持一致（如果前三个相同）。"""
    for i in range(3, len(mask_list)):
        # 获取前 3 个 mask 的相应位置的 tensor（列表中的元素为list，每个list元素为tensor）
        prev1, prev2, prev3 = mask_list[i - 3], mask_list[i - 2], mask_list[i - 1]
        current = mask_list[i]
        print(prev1[0].shape)  # Just a sample print for debugging, remove in production

        # 找到前 3 个 tensor 中相同的位置，逐元素比较
        consistent_mask_list = []
        for t1, t2, t3 in zip(prev1, prev2, prev3):
            # 对每对 tensor 使用 torch.eq() 进行逐元素比较
            consistent_mask = (t1 == t2) & (t2 == t3)
            consistent_mask_list.append(consistent_mask)

        # 更新 mask_list 的当前元素
        updated_mask_list = []
        for idx, (consistent_mask, t1,c1) in enumerate(zip(consistent_mask_list, prev1,current)):
            # 在 consistent_mask 对应位置为 True 时，比较 current[idx] 和 prev1[idx]
            updated_mask = torch.where(
                consistent_mask, 
                torch.where(c1 == t1, current[idx], ~current[idx]),  # 如果一致则保留 current，否则取反
                current[idx]  # 如果不一致，保留 consistent_mask 的值
            )
            updated_mask_list.append(updated_mask)

        # 更新 mask_list 的当前元素
        mask_list[i] = updated_mask_list

    return mask_list


def process_masks(maskAs, maskBs, maskCs):
    """处理 mask 列表并确保它们满足一致性条件。"""
    if len(maskAs) < 3:
        # 如果长度小于 4，将 maskAs 的所有元素复制到 maskB_change 和 maskC_change
        maskA_change = maskAs
        maskB_change = maskAs
        maskC_change = maskAs
    else:
        # 对 maskAs、maskBs 和 maskCs 应用一致性检查
        maskA_change = ensure_consistent_mask(maskAs)
        maskB_change = ensure_consistent_mask(maskBs)
        maskC_change = ensure_consistent_mask(maskCs)

    return maskA_change, maskB_change, maskC_change

# 示例数据 (lists of lists of tensors)
maskAs = [
    [torch.tensor([0, 0, 0], dtype=torch.bool), torch.tensor([0, 0, 0], dtype=torch.bool)],
    [torch.tensor([1, 0, 0], dtype=torch.bool), torch.tensor([0, 1, 0], dtype=torch.bool)],
    [torch.tensor([0, 0, 0], dtype=torch.bool), torch.tensor([0, 0, 0], dtype=torch.bool)],
    [torch.tensor([1, 1, 0], dtype=torch.bool), torch.tensor([1, 1, 1], dtype=torch.bool)]
]

maskBs = [
    [torch.tensor([1, 1, 1], dtype=torch.bool), torch.tensor([1, 1, 1], dtype=torch.bool)],
    [torch.tensor([1, 1, 1], dtype=torch.bool), torch.tensor([1, 0, 1], dtype=torch.bool)],
    [torch.tensor([0, 1, 1], dtype=torch.bool), torch.tensor([0, 0, 1], dtype=torch.bool)],
    [torch.tensor([0, 0, 1], dtype=torch.bool), torch.tensor([0, 1, 1], dtype=torch.bool)]
]

maskCs = [
    [torch.tensor([0, 0, 0], dtype=torch.bool), torch.tensor([0, 0, 0], dtype=torch.bool)],
    [torch.tensor([0, 0, 0], dtype=torch.bool), torch.tensor([0, 1, 0], dtype=torch.bool)],
    [torch.tensor([0, 0, 0], dtype=torch.bool), torch.tensor([0, 1, 1], dtype=torch.bool)],
    [torch.tensor([1, 1, 1], dtype=torch.bool), torch.tensor([0, 1, 0], dtype=torch.bool)]
]

# 运行处理函数
maskA_change, maskB_change, maskC_change = process_masks(maskAs, maskBs, maskCs)

# 打印结果
print("Updated Masks:")
print("maskA_change:", maskA_change)
print("maskB_change:", maskB_change)
print("maskC_change:", maskC_change)
