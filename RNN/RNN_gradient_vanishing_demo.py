import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
from torch import nn
from torch.nn import Embedding, RNNCell
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 一个简单的RNN结构示例
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SimpleRNN, self).__init__()
        self.rnn  = nn.RNN(input_size, hidden_size, batch_first=True)
    
    def forward(self, x):
        out, _ = self.rnn(x)
        return out

class SentimentRNN:
    def __init__(self, vocab_size, embed_size, hidden_size, output_size):
        # 词嵌入层
        self.embed = Embedding(vocab_size, embed_size)
        
        # RNN层
        self.rnn = RNNCell(embed_size, hidden_size)
        
        # 输出层（情感分类：正面/负面）
        self.W_ho = nn.Parameter(torch.randn(output_size, hidden_size))
        self.b_o = nn.Parameter(torch.randn(output_size))
    
    def forward(self, text):
        # text: [batch_size, seq_len]
        
        # 1. 词嵌入
        embedded = self.embed(text)  # [batch_size, seq_len, embed_size]
        
        # 2. RNN处理序列
        batch_size, seq_len, _ = embedded.size()
        h_t = torch.zeros(batch_size, hidden_size)  # 初始隐藏状态
        for t in range(seq_len):
            x_t = embedded[:, t, :]  # 当前时间步的输入
            h_t = self.rnn(x_t, h_t)  # 更新隐藏状态
        
        # 3. 最后时间步的输出（整个句子的表示）
        # y = W_ho·h_last + b_o
        logits = torch.matmul(h_t, self.W_ho.T) + self.b_o
        
        # 4. 二分类：sigmoid得到概率
        probs = torch.sigmoid(logits)
        
        return probs

# 时间序列预测RNN模型
class TimeSeriesRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, output_size=1, num_layers=1):
        """
        初始化时间序列预测专用的RNN模型
        Args:
            input_size (int): 输入特征维度，即每个时间步的输入数据维度，默认1（单变量时间序列）
            hidden_size (int): RNN隐藏层的维度/神经元数量，决定模型的记忆容量，默认32
            output_size (int): 输出结果维度，即模型预测结果的维度，默认1（单变量时间序列预测）
            num_layers (int): RNN的层数，多层RNN可提升模型表达能力，默认1（单层RNN）
        """
        # 调用父类nn.Module的初始化方法，这是PyTorch自定义模型的必备步骤
        super(TimeSeriesRNN, self).__init__()
        
        # 将隐藏层维度保存为实例属性，方便后续前向传播等方法调用
        self.hidden_size = hidden_size
        
        # 将RNN层数保存为实例属性，方便后续初始化隐藏状态等操作使用
        self.num_layers = num_layers
        
        # 定义核心RNN层，构建循环神经网络结构
        # batch_first=True：指定输入输出数据格式为 [batch_size, seq_length, feature_dim]
        # 该格式更符合日常数据处理习惯，默认格式为 [seq_length, batch_size, feature_dim]
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        
        # 定义全连接层（线性层），将RNN隐藏层的输出映射到最终的预测输出维度
        # 输入维度为RNN隐藏层维度hidden_size，输出维度为预测目标维度output_size
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x: [batch_size, seq_len, input_size]
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, h0)  # out: [batch_size, seq_len, hidden_size]
        out = self.fc(out[:, -1, :])  # 只取最后一个时间步的输出
        return out

def load_and_preprocess_data(file_path, seq_length=5):
    """
    加载CSV数据并创建时间序列样本
    file_path: CSV文件路径
    seq_length: 用于预测的历史序列长度
    """
    # 加载数据
    df = pd.read_csv(file_path)
    data = df['agricultural_income'].values
    
    # 数据归一化
    data_mean = data.mean()
    data_std = data.std()
    data_normalized = (data - data_mean) / data_std
    
    # 创建滑动窗口样本
    X, y = [], []
    for i in range(len(data_normalized) - seq_length):
        X.append(data_normalized[i:i+seq_length])
        y.append(data_normalized[i+seq_length])
    
    # 例如，如果你有一个单变量时间序列（每个时间步只有一个值）
    # 但模型期望输入形状为 (batch_size, seq_len, input_size)
    # 那么 input_size = 1，就需要通过 unsqueeze(-1) 把 (batch, seq_len) 变成 (batch, seq_len, 1)
    # 类似于给单通道信号“加一个通道维度”。
    X = torch.FloatTensor(X).unsqueeze(-1)  # [samples, seq_length, 1]
    y = torch.FloatTensor(y).unsqueeze(-1)  # [samples, 1]
    
    return X, y, data_mean, data_std

def train_model(model, X_train, y_train, epochs=200, lr=0.01):
    """
    model: 待训练的RNN模型
    X_train: 训练输入数据
    y_train: 训练目标数据
    epochs: 训练轮数
    lr: 学习率
    """
    criterion = nn.MSELoss() # 损失函数为均方误差
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) # 优化器为Adam
    
    losses = [] # 记录每次迭代的损失值
    print("=== 开始训练 ===")
    for epoch in range(epochs):
        model.train()
        
        # 前向传播
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # 反向传播
        optimizer.zero_grad() # 清空梯度
        loss.backward() # 反向传播
        optimizer.step() # 更新参数
        
        losses.append(loss.item())
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')
    
    return losses


# ============== 旧主程序（已注释） ==============
""" # 模拟数据
batch_size = 4
seq_len = 10
vocab_size = 10000
embed_size = 300
hidden_size = 128

# 创建模拟输入（4个句子，每个10个词）
text = torch.randint(0, vocab_size, (batch_size, seq_len))
print("输入文本形状:", text.shape)  # [4, 10]
print("输入示例（第一个句子）:", text[0])
print()

# 创建模型组件
embedding_layer = nn.Embedding(vocab_size, embed_size)
rnn_cell = nn.RNNCell(embed_size, hidden_size)

# 步骤1：词嵌入
embedded = embedding_layer(text)  # [4, 10, 300]
print("嵌入后形状:", embedded.shape)
print("第一个句子的第一个词的向量:", embedded[0, 0, :5], "...")  # 显示前5个维度
print()

# 步骤2：初始化隐藏状态
h_t = torch.zeros(batch_size, hidden_size)  # [4, 128]
print("初始隐藏状态形状:", h_t.shape)
print("初始隐藏状态值（全零）:", h_t[0, :5], "...")
print()

# 步骤3：循环处理序列
print("=== 循环处理序列 ===")
for t in range(seq_len):
    print(f"时间步 {t}:")
    
    # 获取当前时间步的输入
    x_t = embedded[:, t, :]  # [4, 300]
    print(f"  输入 x_t 形状: {x_t.shape}")
    
    # RNN更新
    h_t = rnn_cell(x_t, h_t)  # [4, 128]
    print(f"  更新后 h_t 形状: {h_t.shape}")
    
    # 查看第一个句子的隐藏状态变化
    print(f"  第一个句子的隐藏状态前5维: {h_t[0, :5]}")
    print() """

# ============== RNN时间序列预测训练 ==============

# 加载和预处理数据
seq_length = 5  # 使用5个历史数据点预测下一个
file_path = '1952-1988年中国农业实际国民收入指数序列.csv'

print("=== 加载数据 ===")
X, y, data_mean, data_std = load_and_preprocess_data(file_path, seq_length)
print(f"数据集大小: X={X.shape}, y={y.shape}")
print(f"数据均值: {data_mean:.2f}, 标准差: {data_std:.2f}")
print()

# 划分训练集和测试集（80%训练，20%测试）
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
print()

# 创建模型
model = TimeSeriesRNN(input_size=1, hidden_size=32, output_size=1, num_layers=2)
print("=== 模型结构 ===")
print(model)
print()

# 训练模型
losses = train_model(model, X_train, y_train, epochs=200, lr=0.01)

# 评估模型
model.eval()
with torch.no_grad():
    train_pred = model(X_train)
    test_pred = model(X_test)
    
    train_loss = nn.MSELoss()(train_pred, y_train)
    test_loss = nn.MSELoss()(test_pred, y_test)
    
    print()
    print("=== 评估结果 ===")
    print(f"训练集MSE: {train_loss.item():.6f}")
    print(f"测试集MSE: {test_loss.item():.6f}")

# 可视化结果
plt.figure(figsize=(12, 5))

# 子图1：训练损失曲线
plt.subplot(1, 2, 1) # 1行2列，第1个子图
plt.plot(losses, color='#2E86AB', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training Loss Curve', fontsize=14)
plt.grid(True, alpha=0.3)

# 子图2：预测结果对比
plt.subplot(1, 2, 2)
# 反归一化处理

train_pred_np = train_pred.numpy() * data_std + data_mean 
y_train_np = y_train.numpy() * data_std + data_mean
test_pred_np = test_pred.numpy() * data_std + data_mean
y_test_np = y_test.numpy() * data_std + data_mean

plt.plot(range(len(y_train_np)), y_train_np, 'o-', label='Train Actual', color='#2E86AB', alpha=0.6)
plt.plot(range(len(y_train_np)), train_pred_np, 's-', label='Train Predicted', color='#A23B72', alpha=0.6)
plt.plot(range(len(y_train_np), len(y_train_np) + len(y_test_np)), y_test_np, 'o-', label='Test Actual', color='#F18F01', alpha=0.6)
plt.plot(range(len(y_train_np), len(y_train_np) + len(y_test_np)), test_pred_np, 's-', label='Test Predicted', color='#C73E1D', alpha=0.6)
plt.xlabel('Sample Index', fontsize=12)
plt.ylabel('Agricultural Income Index', fontsize=12)
plt.title('RNN Predictions vs Actual Values', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rnn_training_results.png', dpi=300, bbox_inches='tight')
print("\n结果图表已保存为: rnn_training_results.png")