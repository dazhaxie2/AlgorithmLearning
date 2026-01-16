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

# 时间序列预测RNN模型
class TimeSeriesRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, output_size=1, num_layers=1):
        super(TimeSeriesRNN, self).__init__()
        
        # 保存参数作为实例变量
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        # 定义RNN层
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        
        # 定义全连接层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: [batch_size, seq_len, input_size]
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, h0)  # out: [batch_size, seq_len, hidden_size]
        out = self.fc(out[:, -1, :])  # 只取最后一个时间步的输出
        return out

# 反向RNN
class ReverseRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(ReverseRNN, self).__init__()
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            # dropout=0.2 if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        # 反转输入序列
        x_reversed = torch.flip(x, dims=[1])
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        output, hn = self.rnn(x_reversed, h0)
        out = self.fc(output[:, -1, :])  # 取最后一个时间步的输出
        return out
        self.rnn = nn.RNN(
            input_size = input_dim,
            hidden_size = hidden_dim,
            num_layers = num_layers,
            
        )

# 双向RNN
class BidirectionalRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers = 1):
        super(BidirectionalRNN, self).__init__()
        self.rnn = nn.RNN(
            input_size = input_dim,
            hidden_size = hidden_dim,
            num_layers = num_layers,
            batch_first = True,
            bidirectional = True
        )
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.rnn.num_layers * 2, batch_size, self.rnn.hidden_size).to(x.device)
        output, hn = self.rnn(x, h0)
        
        # 合并正向和反向的最后一个隐藏状态
        # hn形状：[num_layers * 2, batch_size, hidden_dim]
        forward_h = hn[-2] # 正向最后一个隐藏状态
        backward_h = hn[-1] # 反向最后一个隐藏状态
        combined_h = torch.cat((forward_h, backward_h), dim=1) # [batch_size, hidden_dim * 2]
        
        out = self.fc(combined_h)
        return out


# 多层RNN
class MultiLayerRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers = 2):
        super(MultiLayerRNN, self).__init__()
        self.rnn = nn.RNN(
            input_size = input_dim,
            hidden_size = hidden_dim,
            num_layers = num_layers,
            batch_first = True,
            # dropout = 0.2
        )

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        batch_size = x.size(0)
        h0 = torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size)
        output, hn = self.rnn(x, h0) # output: [batch_size, seq_len, hidden_dim]
        out = self.fc(hn[-1])  # 只取最后一个时间步的输出
        return out

# 训练模型
def train_model(model, X_train, y_train, epochs=300, lr=0.001):
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

# 可视化四种模型对比结果
def plot_comparison_results(results, y_test, data_mean, data_std):
    """
    对比四种RNN模型的预测结果
    results: dict, 包含每种模型的预测结果和损失
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 反归一化真实值
    y_test_np = y_test.numpy() * data_std + data_mean
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    model_names = list(results.keys())
    
    # 子图1：四种模型的训练损失曲线对比
    ax1 = axes[0, 0]
    for i, (name, data) in enumerate(results.items()):
        ax1.plot(data['losses'], label=name, color=colors[i], linewidth=2, alpha=0.8)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss Comparison', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 子图2：四种模型的测试集预测对比
    ax2 = axes[0, 1]
    x_indices = range(len(y_test_np))
    ax2.plot(x_indices, y_test_np, 'ko-', label='Actual', linewidth=2, markersize=6)
    for i, (name, data) in enumerate(results.items()):
        pred_np = data['test_pred'].numpy() * data_std + data_mean
        ax2.plot(x_indices, pred_np, 's--', label=name, color=colors[i], alpha=0.7, markersize=5)
    ax2.set_xlabel('Sample Index', fontsize=12)
    ax2.set_ylabel('Agricultural Income Index', fontsize=12)
    ax2.set_title('Test Set Predictions Comparison', fontsize=14)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 子图3：四种模型的MSE对比（柱状图）
    ax3 = axes[1, 0]
    train_mses = [results[name]['train_mse'] for name in model_names]
    test_mses = [results[name]['test_mse'] for name in model_names]
    x_pos = np.arange(len(model_names))
    width = 0.35
    bars1 = ax3.bar(x_pos - width/2, train_mses, width, label='Train MSE', color='#2E86AB', alpha=0.8)
    bars2 = ax3.bar(x_pos + width/2, test_mses, width, label='Test MSE', color='#C73E1D', alpha=0.8)
    ax3.set_xlabel('Model', fontsize=12)
    ax3.set_ylabel('MSE', fontsize=12)
    ax3.set_title('MSE Comparison', fontsize=14)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(model_names, rotation=15, ha='right')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    # 在柱状图上添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax3.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax3.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    
    # 子图4：预测误差分布对比（箱线图）
    ax4 = axes[1, 1]
    errors = []
    for name in model_names:
        pred_np = results[name]['test_pred'].numpy() * data_std + data_mean
        error = pred_np.flatten() - y_test_np.flatten()
        errors.append(error)
    bp = ax4.boxplot(errors, labels=model_names, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax4.set_xlabel('Model', fontsize=12)
    ax4.set_ylabel('Prediction Error', fontsize=12)
    ax4.set_title('Prediction Error Distribution', fontsize=14)
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_xticklabels(model_names, rotation=15, ha='right')
    
    plt.tight_layout()
    plt.savefig('rnn_models_comparison.png', dpi=300, bbox_inches='tight')
    print("\n四种模型对比图表已保存为: rnn_models_comparison.png")

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
    
    # 定义四种模型
    models = {
        'TimeSeriesRNN': TimeSeriesRNN(input_size=1, hidden_size=32, output_size=1, num_layers=2),
        'ReverseRNN': ReverseRNN(input_dim=1, hidden_dim=32, output_dim=1, num_layers=2),
        'BidirectionalRNN': BidirectionalRNN(input_dim=1, hidden_dim=32, output_dim=1, num_layers=2),
        'MultiLayerRNN': MultiLayerRNN(input_dim=1, hidden_dim=32, output_dim=1, num_layers=3)
    }
    
    # 存储所有模型的结果
    results = {}
    
    # 训练和评估每种模型
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"训练模型: {name}")
        print(f"{'='*50}")
        print(model)
        
        # 训练模型
        losses = train_model(model, X_train, y_train, epochs=300, lr=0.001)
        
        # 评估模型
        model.eval()
        with torch.no_grad():
            train_pred = model(X_train)
            test_pred = model(X_test)
            
            train_mse = nn.MSELoss()(train_pred, y_train).item()
            test_mse = nn.MSELoss()(test_pred, y_test).item()
            
            print(f"\n{name} 评估结果:")
            print(f"  训练集MSE: {train_mse:.6f}")
            print(f"  测试集MSE: {test_mse:.6f}")
        
        # 存储结果
        results[name] = {
            'losses': losses,
            'train_pred': train_pred,
            'test_pred': test_pred,
            'train_mse': train_mse,
            'test_mse': test_mse
        }
    
    # 打印总结
    print(f"\n{'='*50}")
    print("四种模型性能总结")
    print(f"{'='*50}")
    print(f"{'模型名称':<20} {'训练集MSE':<15} {'测试集MSE':<15}")
    print("-" * 50)
    for name, data in results.items():
        print(f"{name:<20} {data['train_mse']:<15.6f} {data['test_mse']:<15.6f}")
    
    # 找出最优模型
    best_model = min(results.items(), key=lambda x: x[1]['test_mse'])
    print(f"\n最优模型（测试集MSE最低）: {best_model[0]}，MSE = {best_model[1]['test_mse']:.6f}")
    
    # 可视化对比结果
    plot_comparison_results(results, y_test, data_mean, data_std)

if __name__ == "__main__":
    main()