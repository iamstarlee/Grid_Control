import os
import torch
from joblib import dump, load
import torch.utils.data as Data
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import torch.nn as nn
import datetime
import logging
import argparse
import time
from csv import writer
import matplotlib
matplotlib.use('Agg')  # 强制使用非交互式后端
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from model import CEEMDANCNNBiLSTMModel, device

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_name = 'OYLPm24'
output_size = 4 # 多步预测输出

# 添加命令行参数解析
parser = argparse.ArgumentParser(description='CEEMDAN-CNN-LSTM模型推理')
def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser.add_argument('--s_true', type=str_to_bool, default=False, help='是否加入结构可塑性')
args = parser.parse_args()

scaler = load('data_ed/OYLPm24_720_4_scaler')

if args.s_true:
    model_path = 's_best_4_model.pt'
    # 构建日志文件的完整路径
    log_filename = f"s_log/log.txt"
else:
    # model_path = 'best_4_model.pt'
    model_path = 'logs/0904_1657_OYLPm24_0.1_0.8_40_150/best_4_model.pt'
    # 构建日志文件的完整路径
    log_filename = f"log/log.txt"


# 配置日志记录器
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, mode='w'), # 输出到文件（覆盖模式）
        logging.StreamHandler() # 输出到控制台
    ]
)
def dataloader(batch_size=64, workers=0):

    # 测试集
    test_set = load(f'data_ed/{data_name}_test_720_4xdata')
    test_label = load(f'data_ed/{data_name}_test_720_4ylabel')

    # 加载数据
    test_loader = Data.DataLoader(dataset=Data.TensorDataset(test_set, test_label),
                                  batch_size=batch_size, num_workers=workers, drop_last=True)
    return test_loader

# 加载数据
test_loader = dataloader()
def model_pre(model, test_loader, scaler):
    infer_start_time = time.time()
    # 模型加载
    model = model.to(device)

    # 预测数据
    original_data = []
    pre_data = []
    
    with torch.no_grad():
            for data, label in test_loader:
                origin_lable = label.tolist()
                original_data += origin_lable

                model.eval()  # 将模型设置为评估模式
                data, label = data.to(device), label.to(device)
                # 预测
                test_pred = model(data)  # 对测试集进行预测
                test_pred = test_pred.tolist()
                pre_data += test_pred
    # 将列表转换为 NumPy 数组
    original_data = np.array(original_data)
    pre_data = np.array(pre_data)

    infer_end_time = time.time()
    infer_time = infer_end_time - infer_start_time
    logging.info(f'Inference time: {infer_time:.6f} seconds')

    # 保存归一化后的数据
    if args.s_true:
        save_path = os.path.join("s_log", 'normalized_data.csv')
    else:
        save_path = os.path.join("log", 'normalized_data.csv')
    pd.DataFrame({
        'original': original_data[:, output_size-1],
        'predicted': pre_data[:, output_size-1]
    }).to_csv(save_path, index=False)
    logging.info(f'Normalized data saved at {save_path}')

    # 模型分数
    score = r2_score(original_data, pre_data)
    logging.info('*'*50)
    logging.info('Normalized model--R^2: %s', score)

    logging.info('*'*50)
    # 测试集上的预测误差
    test_mse = mean_squared_error(original_data, pre_data)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(original_data, pre_data)
    logging.info('test_dataset--MSE: %s', test_mse)
    logging.info('test_dataset--RMSE: %s', test_rmse)
    logging.info('test_dataset--MAE: %s', test_mae)

    # 计算准确率
    threshold = 0.5
    relative_error = np.abs(original_data - pre_data) / (np.abs(original_data) + 1e-8)  # 避免除零
    correct = (relative_error < threshold).astype(int)
    accuracy = correct.mean()
    logging.info('test_dataset--ACC: %s', accuracy)


    # 可视化结果
    plt.figure(figsize=(12, 6), dpi=100)
    plt.clf()  # 确保新建干净figure
    plt.plot(original_data[:, output_size-1], label='original data', color='orange')  # 真实值
    plt.plot(pre_data[:, output_size-1], label=f'Normalized + {output_size } step prediction', color='green')  # 预测值
    plt.legend()
    try:
        if args.s_true:
            plt.savefig(os.path.join("s_log", f'Normalized_{output_size} step prediction.png'))
        else:
            plt.savefig(os.path.join("log", f'Normalized_{output_size} step prediction.png'))
        logging.info(f'Normalized_{output_size} step prediction.png saved successfully')
        plt.clf()  # 清除当前figure
        plt.close()  # 关闭figure释放内存
    except Exception as e:
        logging.error(f'Failed to save prediction image: {e}')

    plt.clf()
if __name__ == '__main__':
    # 模型推理
    infer_model = torch.load(model_path, weights_only=False)
    model_pre(infer_model, test_loader, scaler)
