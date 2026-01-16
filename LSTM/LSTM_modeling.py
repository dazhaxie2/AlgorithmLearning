import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置环境变量以避免OpenMP库冲突
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# 数据预处理
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
    
    X = torch.FloatTensor(X).unsqueeze(-1)  # [samples, seq_length, 1]
    y = torch.FloatTensor(y).unsqueeze(-1)  # [samples, 1]
    
    return X, y, data_mean, data_std

# 时间序列预测LSTM模型
class TimeSeriesLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, output_size=1, num_layers=1):
        super(TimeSeriesLSTM, self).__init__()
        
        # 保存参数作为实例变量
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        # 使用 LSTM 替换 LSTM 以解决梯度消失问题
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # 定义全连接层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: [batch_size, seq_len, input_size]
        # LSTM 自动处理初始状态，如果不传则默认为 0
        out, _ = self.lstm(x)  # out: [batch_size, seq_len, hidden_size]
        out = self.fc(out[:, -1, :])  # 只取最后一个时间步的输出
        return out

# 训练模型
def train_model(model, X_train, y_train, epochs=200, lr=0.01):
    criterion = nn.SmoothL1Loss()  # 使用平滑L1损失
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # 优化器为Adam
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    
    losses = []  # 记录每次迭代的损失值
    for epoch in range(epochs):
        model.train()
        
        for batch_x, batch_y in train_loader:
            # 前向传播
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        losses.append(loss.item())
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')
    
    return losses

# 可视化训练过程和预测结果
def plot_results(losses, train_pred, y_train, test_pred, y_test, data_mean, data_std):
    plt.figure(figsize=(12, 5))

    # 子图1：训练损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(losses, color='#2E86AB', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss Curve', fontsize=14)
    plt.grid(True, alpha=0.3)

    # 子图2：预测结果对比
    plt.subplot(1, 2, 2)
    
    # 反归一化
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
    plt.title('LSTM Predictions vs Actual Values', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('lstm_results_v0.0.png', dpi=300, bbox_inches='tight')
    print("\n结果图表已保存为: lstm_results_v0.0.png")

# 主程序
def main():
    # 加载数据
    seq_length = 5  # 使用5个历史数据点预测下一个
    file_path = '1952-1988年中国农业实际国民收入指数序列.csv'

    print("=== 加载数据 ===")
    X, y, data_mean, data_std = load_and_preprocess_data(file_path, seq_length)
    print(f"数据集大小: X={X.shape}, y={y.shape}")
    print(f"数据均值: {data_mean:.2f}, 标准差: {data_std:.2f}")
    
    # 划分训练集和测试集（80%训练，20%测试）
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
    
    # 创建模型
    model = TimeSeriesLSTM(input_size=1, hidden_size=32, output_size=1, num_layers=2)
    print("=== 模型结构 ===")
    print(model)
    
    # 训练模型
    losses = train_model(model, X_train, y_train, epochs=300, lr=0.001)
    
    # 评估模型
    model.eval()
    with torch.no_grad():
        train_pred = model(X_train)
        test_pred = model(X_test)
        
        train_loss = nn.MSELoss()(train_pred, y_train)
        test_loss = nn.MSELoss()(test_pred, y_test)
        
        print("\n=== 评估结果 ===")
        print(f"训练集MSE: {train_loss.item():.6f}")
        print(f"测试集MSE: {test_loss.item():.6f}")
    
    # 可视化结果
    plot_results(losses, train_pred, y_train, test_pred, y_test, data_mean, data_std)

if __name__ == "__main__":
    main()