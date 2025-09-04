import torch
from torch.utils.data import random_split
import torch.utils.data as Data
from joblib import load
import os
import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import torch.nn as nn
import datetime
import logging
import argparse
from torch.utils.tensorboard import SummaryWriter
import random
import time
import copy
from csv import writer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from model_cl import CEEMDANCNNBiLSTMModel
from prun import *


def set_deterministic(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # 多GPU时也需要
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Set deterministic mode with seed {seed}.")

# 看下这个网络结构总共有多少个参数
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Total Parameters: {total_params:,}")
    table = []
    total = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            table.append([name, param.numel()])
            total+=param.numel()
    logging.info(f"Total Trainable Parameters: {total:,}")
    logging.info(table)

# 加载数据集
def dataloader(batch_size, workers=4):
    # 训练集
    train_set = load(f'data_ed/OYLPm24_train_720_4xdata')
    train_label = load(f'data_ed/OYLPm24_train_720_4ylabel')
    # 验证集
    val_set = load(f'data_ed/OYLPm24_val_720_4xdata')
    val_label = load(f'data_ed/OYLPm24_val_720_4ylabel')
    # 测试集
    test_set = load(f'data_ed/OYLPm24_test_720_4xdata')
    test_label = load(f'data_ed/OYLPm24_test_720_4ylabel')


    dataset_train=Data.TensorDataset(train_set, train_label)
    dataset_val = Data.TensorDataset(val_set, val_label)
    dataset_test = Data.TensorDataset(test_set, test_label)

    # 划分子集大小--train
    subset_size = len(dataset_train) // 10
    lengths = [subset_size] * 10
    remainder = len(dataset_train) - sum(lengths)
    if remainder > 0:
        lengths[-1] += remainder


    # 固定随机生成器种子
    g = torch.Generator().manual_seed(42)  # 42 可以换成你喜欢的数字

    # 使用 random_split，并传入生成器
    subsets = random_split(dataset_train, lengths, generator=g)

    # Train DataLoader 
    batch_size = 64
    train_loader = [
        Data.DataLoader(subset, batch_size=batch_size, shuffle=True)
        for subset in subsets
    ]

    # 划分子集大小--val
    subset_size = len(dataset_val) // 10
    lengths_val = [subset_size] * 10
    remainder = len(dataset_val) - sum(lengths_val)
    if remainder > 0:
        lengths_val[-1] += remainder

    subsets_val = random_split(dataset_val, lengths_val, generator=g)

    val_loader = [
        Data.DataLoader(subset, batch_size=batch_size, shuffle=True)
        for subset in subsets_val
    ]

    # 划分子集大小--test
    subset_size = len(dataset_test) // 10
    lengths_test = [subset_size] * 10
    remainder = len(dataset_test) - sum(lengths_test)
    if remainder > 0:
        lengths_test[-1] += remainder
        
    subsets_test = random_split(dataset_test, lengths_test, generator=g)

    test_loader = [
        Data.DataLoader(subset, batch_size=batch_size, shuffle=True)
        for subset in subsets_test
    ]

    # for task_id, loader in enumerate(train_loader):
    #     print(f"\n=== Training on Task {task_id} with {len(loader.dataset)} samples ===")
    #     for batch_x, batch_y in loader:
    #         # 在这里写训练逻辑
    #         # print(batch_x.shape, batch_y.shape)
    #         pass

    

    # # 加载数据
    # train_loader = Data.DataLoader(dataset=Data.TensorDataset(train_set, train_label),
    #                                batch_size=batch_size, num_workers=workers, drop_last=True)
    # val_loader = Data.DataLoader(dataset=Data.TensorDataset(val_set, val_label),
    #                               batch_size=batch_size, num_workers=workers, drop_last=True)
    # test_loader = Data.DataLoader(dataset=Data.TensorDataset(test_set, test_label),
    #                               batch_size=batch_size, num_workers=workers, drop_last=True)
    return train_loader, val_loader, test_loader

def model_train(batch_size, epochs, model, optimizer, loss_function, train_loader, val_loader, test_loader):
    
    # tensorboard
    try:
        tb_writer = SummaryWriter(log_dir)
    except Exception as e:
        logging.error(f"TensorBoard初始化失败：{e}")
        tb_writer = None

    # model = model.to(device)
    # 样本长度
    train_size = len(train_loader) * batch_size
    val_size = len(val_loader) * batch_size

    # 最低MSE
    minimum_mse = 1000.
    # 最佳模型
    best_model = model
    best_epoch = 0
    
    # 剪枝控制标志初始化
    prun_true = False

    train_mse = []  # 记录在训练集上每个epoch的 MSE 指标的变化情况   平均值
    val_mse = []  # 记录在验证集上每个epoch的 MSE 指标的变化情况   平均值

    # 创建loss保存文件
    loss_csv_path = os.path.join(log_dir, 'loss_log.csv')
    with open(loss_csv_path, 'w', newline='') as f:
        csv_writer = writer(f)
        csv_writer.writerow(['Epoch', 'Train Loss', 'Val Loss'])

    # print(model.feature[0])

    start_time = time.time()
    for epoch in range(epochs):
        # 训练
        model.train()

        train_mse_loss = 0.  # 保存当前epoch的MSE loss和
        for seq, labels in train_loader:
            seq, labels = seq.to(device), labels.to(device)
            optimizer.zero_grad()
            y_pred = model(seq)
            loss = loss_function(y_pred, labels)
            train_mse_loss += loss.item()
            # 反向传播和参数更新
            loss.backward()
            optimizer.step()
            # 在优化器步骤后始终重新应用掩码，确保被剪枝的权重保持为零
            pruner.apply_mask()
        if prun_true and epoch>prun_start and epoch<prun_end:
            pruner.prune_and_maybe_regrow()
            logging.info(f"Epoch {epoch+1}: density={pruner.density():.3f} "
                f"nonzero={pruner.count_nonzero()}/{pruner.total_params()}")
            prun_true = False

        train_av_mseloss = train_mse_loss / train_size
        train_mse.append(train_av_mseloss)
        logging.info(f'Epoch: {epoch + 1:2} train_MSE-Loss: {train_av_mseloss:10.8f}')

        with torch.no_grad():
            model.eval()
            val_mse_loss = 0. 
            for data, label in val_loader:
                data, label = data.to(device), label.to(device)
                pre = model(data)
                val_loss = loss_function(pre, label)
                val_mse_loss += val_loss.item()
            val_av_mseloss = val_mse_loss / val_size
            val_mse.append(val_av_mseloss)

            with open(loss_csv_path, 'a', newline='') as f:
                csv_writer = writer(f)
                csv_writer.writerow([epoch+1, train_av_mseloss, val_av_mseloss])

            if tb_writer:
                tb_writer.add_scalars('Loss', {'Train': train_av_mseloss, 'Val': val_av_mseloss}, epoch)
                
            logging.info(f'Epoch: {epoch + 1:2} val_MSE_Loss:{val_av_mseloss:10.8f}')
            # 保存当前最优模型参数
            if val_av_mseloss < minimum_mse:
                prun_true = True # 继续剪枝
                minimum_mse = val_av_mseloss
                best_model = copy.deepcopy(model)
                best_epoch = epoch
                logging.info(f'{epoch} epoch min_MSE: {minimum_mse}\n')
    if tb_writer:
        tb_writer.close()  
    try:
        torch.save(best_model, os.path.join(log_dir, f'best_{output_size}_model.pt'))   
        logging.info(f'\nDuration: {time.time() - start_time:.0f} seconds')
    except Exception as e:
        logging.error(f'Failed to save best model: {e}')

    logging.info(f'{output_size}-step prediction min_MSE: {minimum_mse}, best_epoch: {best_epoch}')

    # 模型评估,测试集
    scaler = load(f'data_ed/OYLPm24_720_4_scaler')
    model_pre(best_model, test_loader, scaler)

    # 可视化
    plt.plot(range(epochs), train_mse, color='b', label='train_MSE-loss')
    plt.plot(range(epochs), val_mse, color='y', label='val_MSE-loss')
    plt.legend()
    save_path = os.path.join(log_dir,f"{output_size}-step prediction loss curve.png")
    try:
        plt.savefig(save_path)
        plt.clf()  # 清除当前figure
        plt.close()  # 关闭figure释放内存
        logging.info(f'Loss curve saved successfully at {save_path}')
    except Exception as e:
        logging.error(f'Failed to save loss curve: {e}')
    
    # plt.show()  # 显示 lable
    # logging.info(f'min_MSE: {minimum_mse}')
    # 记录日志
    # 定义模型参数
    logging.info(f"seed: {seed}")
    logging.info(f"batch_size: {batch_size}")
    logging.info(f"input_dim: {input_dim}")
    logging.info(f"conv_archs: {conv_archs}")
    logging.info(f"hidden_layer_sizes: {hidden_layer_sizes}")
    logging.info(f"output_dim: {output_dim}")
    logging.info(f"output_size: {output_size}")
    logging.info(f"learn_rate: {learn_rate}")
    logging.info(f"drop_ratio: {n}")
    logging.info(f"min_density: {m}")
    logging.info(f"prun_start: {prun_start}")
    logging.info(f"prun_end: {prun_end}")



    count_parameters(model)

def model_pre(model, test_loader, scaler):
    # 模型加载
    model = model.to(device)

    # 预测数据
    original_data = []
    pre_data = []
    with torch.no_grad():
            for data, label in test_loader:
                origin_lable = label.tolist()
                original_data += origin_lable
                model.eval()
                data, label = data.to(device), label.to(device)
                test_pred = model(data)
                test_pred = test_pred.tolist()
                pre_data += test_pred
    # 将列表转换为 NumPy 数组
    original_data = np.array(original_data)
    pre_data = np.array(pre_data)
    logging.info('data shape:')
    logging.info('Original data shape: %s, Predicted data shape: %s', original_data.shape, pre_data.shape)

    # 保存归一化后的数据
    save_path = os.path.join(log_dir, 'normalized_data.csv')
    pd.DataFrame({
        'original': original_data[:, output_size-1],
        'predicted': pre_data[:, output_size-1]
    }).to_csv(save_path, index=False)
    logging.info(f'Normalized data saved at {save_path}')

    # 创建一个包含三张子图的画布
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 10), dpi=300)
    # 循环绘制三张子图
    # 随机选择3个起始位置
    total_rows = len(original_data)
    for i, ax in enumerate(axes):
        start_idx = np.random.randint(0, total_rows - output_size)
        ax.plot(original_data[start_idx, : ],  color = 'red',label = 'Original Value', linestyle='-', linewidth=2, marker='o', markersize=3)
        ax.plot(pre_data[start_idx, : ], color = 'green', label = 'Predicted Value', linestyle=':', marker='x', markersize=3) 
        ax.set_title(f'start_{start_idx},{output_size}_step')
        # 调整图例
        ax.legend(fontsize=12)
    plt.grid(True)  # 网格线
    # 调整子图之间的间距
    plt.tight_layout()
    # 保存图片
    plt.savefig(os.path.join(log_dir, f'Normalized_{output_size} step prediction_sample.png'), dpi=300)
    plt.close()

    # 模型分数
    score = r2_score(original_data, pre_data)
    logging.info('*'*50)
    logging.info('Normalized model--R^2: %s', score)

    logging.info('*'*50)
    # 测试集上的预测误差
    test_mse = mean_squared_error(original_data, pre_data)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(original_data, pre_data)
    # calculate accuracy
    threshold = 0.5
    relative_error = np.abs(original_data - pre_data) / (np.abs(original_data) + 1e-8)  # 避免除零
    correct = (relative_error < threshold).astype(int)
    accuracy = correct.mean()

    logging.info('Normalized test_dataset--MSE: %s', test_mse)
    logging.info('Normalized test_dataset--ACC: %s', accuracy)
    logging.info('Normalized test_dataset--RMSE: %s', test_rmse)
    logging.info('Normalized test_dataset--MAE: %s', test_mae)

    # 可视化结果
    plt.figure(figsize=(12, 6), dpi=100)
    plt.clf()  # 确保新建干净figure
    plt.plot(original_data[:, output_size-1], label='original data', color='orange')  # 真实值
    plt.plot(pre_data[:, output_size-1], label=f'Normalized + {output_size } step prediction', color='green')  # 预测值
    plt.legend()
    try:
        plt.savefig(os.path.join(log_dir, f'Normalized_{output_size} step prediction.png'))
        print(f'Normalized_{output_size} step prediction.png saved successfully')
        plt.clf()  # 清除当前figure
        plt.close()  # 关闭figure释放内存
    except Exception as e:
        logging.error(f'Failed to save prediction image: {e}')

    plt.clf()
    # 反归一化处理
    # 使用相同的均值和标准差对预测结果进行反归一化处理
    # 反标准化
    original_data = scaler.inverse_transform(original_data)
    pre_data = scaler.inverse_transform(pre_data)
    
    # 保存反归一化后的数据
    save_path = os.path.join(log_dir, 'denormalized_data.csv')
    pd.DataFrame({
        'original': original_data[:, output_size-1],
        'predicted': pre_data[:, output_size-1]
    }).to_csv(save_path, index=False)
    logging.info(f'Denormalized data saved at {save_path}')

    # 创建一个包含三张子图的画布
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 10), dpi=300)
    # 循环绘制三张子图
    # 随机选择3个起始位置
    total_rows = len(original_data)
    for i, ax in enumerate(axes):
        start_idx = np.random.randint(0, total_rows - output_size)
        ax.plot(original_data[start_idx, : ],  color = 'red',label = 'Original Value', linestyle='-', linewidth=2, marker='o', markersize=3)
        ax.plot(pre_data[start_idx, : ], color = 'green', label = 'Predicted Value', linestyle=':', marker='x', markersize=3) 
        ax.set_title(f'start_{start_idx},{output_size}_step')
        # 调整图例
        ax.legend(fontsize=12)
    plt.grid(True)  # 网格线
    # 调整子图之间的间距
    plt.tight_layout()
    # 保存图片
    plt.savefig(os.path.join(log_dir, f'Denormalized_{output_size} step prediction_sample.png'), dpi=300)
    plt.close()

    # # 模型分数
    # score = r2_score(original_data, pre_data)
    # logging.info('*'*50)
    # logging.info('Denormalized model--R^2: %s', score)

    # logging.info('*'*50)
    # # 测试集上的预测误差
    # test_mse = mean_squared_error(original_data, pre_data)
    # test_rmse = np.sqrt(test_mse)
    # test_mae = mean_absolute_error(original_data, pre_data)
    # logging.info('Denormalized test_dataset--MSE: %s', test_mse)
    # logging.info('Denormalized test_dataset--ACC: %s', 1-test_mse)
    # logging.info('Denormalized test_dataset--RMSE: %s', test_rmse)
    # logging.info('Denormalized test_dataset--MAE: %s', test_mae)

    # 可视化结果
    plt.figure(figsize=(12, 6), dpi=100)
    plt.clf()  # 确保新建干净figure
    plt.plot(original_data[:, output_size-1], label='original data', color='orange')  # 真实值
    plt.plot(pre_data[:, output_size-1], label=f'Denormalized + {output_size } step prediction', color='green')  # 预测值
    plt.legend()
    try:
        plt.savefig(os.path.join(log_dir, f'Denormalized_{output_size} step prediction.png'))
        print(f'Denormalized_{output_size} step prediction.png saved successfully')
        plt.clf()  # 清除当前figure
        plt.close()  # 关闭figure释放内存
    except Exception as e:
        logging.error(f'Failed to save prediction image: {e}')

    plt.clf()
    # plt.show()

if __name__ == '__main__':
    # 定义模型参数
    batch_size = 64
    input_dim = 15  # 输入维度为15
    conv_archs = ((1, 16),)
    hidden_layer_sizes = [32, 16, 32]
    output_dim = 1 # 输出维度
    output_size = 4 # 多步预测输出
    learn_rate = 0.0003
    epochs = 350  # 50 
    n = 0.10  # 0.10
    m = 0.80  # 0.80
    prun_start = 40 # 开始剪枝的轮数
    prun_end = 250 # 45 

    # 参数与配置
    seed = 42  # 设置随机种子，以使实验结果具有可重复性

    set_deterministic(seed)
    current_time = datetime.datetime.now().strftime("%m%d_%H%M")
    log_dir = "logs/" + current_time +  f'_OYLPm24' + f'_{n}_{m}_{prun_start}_{prun_end}'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # 构建日志文件的完整路径
    log_filename = f"{current_time}.txt"
    log_filepath = os.path.join(log_dir, log_filename)
    # 配置日志记录器
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filepath), # 输出到文件
            logging.StreamHandler() # 输出到控制台
        ]
    )
    # 指定使用第0号GPU
    gpu_id_1 = "0,1"
    gpu_id_2 = "2,3"
    gpu_id_all = "0,1,2,3"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"hidden_layer_sizes is {len(hidden_layer_sizes)}")
    model = CEEMDANCNNBiLSTMModel(batch_size, input_dim, conv_archs, hidden_layer_sizes, output_dim, output_size).to(device)
    
    pretrained_dict = torch.load('best_4_model.pt', weights_only=False).state_dict()
    model_dict = model.state_dict()
    load_key, no_load_key, temp_dict = [], [], {}
    
    for k, v in pretrained_dict.items():
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
            load_key.append(k)
        else:
            no_load_key.append(k)
    model_dict.update(temp_dict)
    model.load_state_dict(model_dict)
    print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
    print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
    print("\n\033[1;33;44m温馨提示，部分参数没有载入是正常现象。\033[0m")

    for name, param in model.named_parameters():
        if 'adapters' not in name:
            param.requires_grad = False

    loss_function = nn.MSELoss(reduction='sum')  # loss

    # optimizer = torch.optim.Adam(model.parameters(), learn_rate)  # 优化器
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learn_rate)

    pruner = StochasticMagnitudePruner(
        model,
        drop_ratio=n,            # init 0.1
        min_density=m,           # init 0.8
        exclude_biases=False,
        normalize="global",
        device=device,
        seed=seed,
    )
    logging.info(model)
    # 加载数据
    train_loader, val_loader, test_loader = dataloader(batch_size)

    # Incremental Learning
    # for task_id in range(5):
    #     model_train(batch_size, epochs, model, optimizer, loss_function, train_loader, val_loader, test_loader)
    model_train(batch_size, epochs, model, optimizer, loss_function, train_loader[6], val_loader[6], test_loader[6])
