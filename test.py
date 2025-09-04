import torch.utils.data as Data
import torch
from joblib import load

if __name__ == '__main__':
    data_name = 'OYLPm24'
    val_set = load(f'data_ed/{data_name}_val_720_4xdata')
    val_label = load(f'data_ed/{data_name}_val_720_4ylabel')

    train_set = load(f'data_ed/{data_name}_train_720_4xdata')
    train_label = load(f'data_ed/{data_name}_train_720_4ylabel')
    # 测试集
    test_set = load(f'data_ed/{data_name}_test_720_4xdata')
    test_label = load(f'data_ed/{data_name}_test_720_4ylabel')


    train_loader = Data.DataLoader(dataset=Data.TensorDataset(train_set, train_label),
                                  batch_size=1, num_workers=1, drop_last=True)
    val_loader = Data.DataLoader(dataset=Data.TensorDataset(val_set, val_label),
                                  batch_size=1, num_workers=1, drop_last=True)
    test_loader = Data.DataLoader(dataset=Data.TensorDataset(test_set, test_label),
                                  batch_size=1, num_workers=1, drop_last=True)

    
    print(f"test_loader is {len(test_loader)}")
    for step, (data, label) in enumerate(test_loader):
        print(data.shape, label.shape)