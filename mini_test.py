import torch
from torch.utils.data import random_split
import torch.utils.data as Data
from joblib import load

def test_dataset():
    # 假设已有的数据
    train_set = load(f'data_ed/OYLPm24_test_720_4xdata')
    train_label = load(f'data_ed/OYLPm24_test_720_4ylabel')
    dataset_train=Data.TensorDataset(train_set, train_label)

    # 划分子集大小
    subset_size = len(dataset_train) // 10
    lengths = [subset_size] * 10
    remainder = len(dataset_train) - sum(lengths)
    if remainder > 0:
        lengths[-1] += remainder

    # 固定随机生成器种子
    g = torch.Generator().manual_seed(42)  # 42 可以换成你喜欢的数字

    # 使用 random_split，并传入生成器
    subsets = random_split(dataset_train, lengths, generator=g)

    # DataLoader 列表
    batch_size = 64
    dataloaders = [
        Data.DataLoader(subset, batch_size=batch_size, shuffle=True)
        for subset in subsets
    ]

    for task_id, loader in enumerate(dataloaders):
        print(f"\n=== Training on Task {task_id} with {len(loader.dataset)} samples ===")
        for batch_x, batch_y in loader:
            # 在这里写训练逻辑
            # print(batch_x.shape, batch_y.shape)
            pass

if __name__ == '__main__':
    test_dataset()