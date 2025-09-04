import torch.nn as nn
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义低秩Adapter
class Adapter(nn.Module):
    def __init__(self, hidden_size, bottleneck_size=16):
        super(Adapter, self).__init__()
        self.down = nn.Linear(hidden_size, bottleneck_size, bias=True)
        self.up = nn.Linear(bottleneck_size, hidden_size, bias=True)

    def forward(self, x):
        return x + self.up(self.down(x))


class ReGhos_Block(nn.Module):
    def __init__(self, hidden_size, bottleneck_size=16, kernel_size=3):
        super(ReGhos_Block, self).__init__()
        
        self.out_channels = bottleneck_size
        self.in_channels = 360
        self.kernel_size = kernel_size
        self.stride = 1
        self.padding = kernel_size // 2
        self.dilation = 1

        self.Ghos_Mod = nn.Sequential(
        nn.Conv1d(self.in_channels, self.in_channels, kernel_size, 1, kernel_size//2, groups=36, bias=True),
        )
        

    def forward(self, x):
        output = x + self.Ghos_Mod(x)
        return output

class CEEMDANCNNBiLSTMModel(nn.Module):
    def __init__(self, batch_size, input_dim, conv_archs, hidden_layer_sizes, output_dim, output_size):
        """
        预测任务  params:
        batch_size       : 批次量大小
        input_dim        : 输入数据的维度
        conv_archs       : CNN 网络结构，层数和每层通道数
        hidden_layer_size: BiLSTM 隐层的数目和维度
        output_dim       : 输出维度
        output_size      : 输出序列长度，对应多步预测步长
        """
        super().__init__()
        # 批次量大小
        self.batch_size = batch_size
        # CNN参数
        self.conv_arch = conv_archs  # 网络结构
        self.input_channels = input_dim  # 输入通道数
        self.feature = self.make_layers()
        self.convlayer = [0,3,99]    # 卷积层索引，其它层没有权重参数，不加索引,99用来占位
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = 'rclstm'

        # bilstm层数
        self.num_layers = len(hidden_layer_sizes)
        self.bilstm_layers = nn.ModuleList()  # 用于保存LSTM层的列表
        self.adapters = nn.ModuleList([
            Adapter(hidden_size=32) for _ in range(2)
            ])  # 用于保存Adapter层的列表
        
        # self.ghosts = nn.ModuleList([
        #     ReGhos_Block(hidden_size=32) for _ in range(2)
        # ])  # 用于保存Ghost模块的列表
        
        # 定义第一层BiLSTM
        # 处理单层或多层配置
        conv_output_size = conv_archs[-1] if isinstance(conv_archs[ 0], int) else conv_archs[-1][-1]
        self.bilstm_layers.append(
            nn.LSTM(conv_output_size, hidden_layer_sizes[0], batch_first=True, bidirectional=False)
           
        )
        
        # 定义后续的BiLSTM层
        for i in range(1, self.num_layers):
            self.bilstm_layers.append(
                nn.LSTM(hidden_layer_sizes[i - 1] * 1, hidden_layer_sizes[i], batch_first=True, bidirectional=False)
                )

        # 定义线性层
        self.linear = nn.Linear(hidden_layer_sizes[-1] * 1, output_dim * output_size)

    # CNN卷积池化结构
    def make_layers(self):
        layers = []
        # 处理单层或多层配置
        # conv_arch = self.conv_arch if isinstance(self.conv_arch, list) else [self.conv_arch]
        for (num_convs, out_channels) in self.conv_arch:
            for _ in range(num_convs):
                layers.append(nn.Conv1d(self.input_channels, out_channels, kernel_size=3, padding=1))
                layers.append(nn.ReLU(inplace=True))
                self.input_channels = out_channels
            layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def forward(self, input_seq):
        # 改变输入形状，适应网络输入[batch , dim, seq_length]
        input_seq = input_seq.permute(0, 2, 1)
        # 送入 CNN 网络
        cnn_feature = self.feature(input_seq)  # torch.Size([64, 64, 6])

        # 送入 BiLSTM 模型
        # 改变输入形状，适应网络输入[batch, seq_length, dim]
        # 使用 permute 方法进行维度变换， 实现了维度的变换，而不改变数据的顺序
        bilstm_out = cnn_feature.permute(0, 2, 1)
        for bilstm in self.bilstm_layers:
            bilstm_out, _ = bilstm(bilstm_out)  ## 进行一次LSTM层的前向传播
        # print(lstm_out.size())  # torch.Size([64, 6, 256])
        for adapter in self.adapters:
            bilstm_out = adapter(bilstm_out)
        # for ghost in self.ghosts:
        #     bilstm_out = ghost(bilstm_out)

        predict = self.linear(bilstm_out[:, -1, :])  # torch.Size([64, 3]  # 仅使用最后一个时间步的输出
        return predict