# GRU 建模
# 环境配置与数据准备：导入必要的库（PyTorch, Pandas, Numpy）并设置环境变量解决可能的库冲突。
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置环境变量以避免OpenMP库冲突
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# 数据预处理：加载 CSV 数据，进行归一化处理，并使用滑动窗口法构建训练样本。
data = pd.read_csv('1952-1988年中国农业实际国民收入指数序列.csv')

# GRU 模型定义：这是学习的核心。你需要理解 GRU 如何通过重置门 (Reset Gate) 和 更新门 (Update Gate) 来精简 LSTM 的结构。
class GRUModel(torch.nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super(GRUModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.gru = torch.nn.GRU(input_size, hidden_layer_size)
        self.linear = torch.nn.Linear(hidden_layer_size, output_size)

    def forward(self, input, hidden=None):
        # input 形状: (seq_len, batch, input_size)
        # 根据项目规范，显式传入 hidden 状态
        if hidden is None:
            hidden = torch.zeros(1, input.size(1), self.hidden_layer_size).to(input.device)
        
        out, hidden = self.gru(input, hidden)
        
        # 我们只取序列的最后一个时间步的输出进行预测
        # out 形状: (seq_len, batch, hidden_size)
        # 取 [-1] 得到 (batch, hidden_size)
        prediction = self.linear(out[-1])
        return prediction, hidden

# 模型训练逻辑
def train_model(model, train_data, train_labels, epochs=100, lr=0.01):
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []  # 记录损失值
    
    for i in range(epochs):
        # 注意：这里假设 train_data 已经是 (seq_len, 1, 1) 的形状
        for seq, labels in zip(train_data, train_labels):
            optimizer.zero_grad()
            
            # 增加 batch 维度和 feature 维度: (seq_len) -> (seq_len, 1, 1)
            seq_input = seq.view(len(seq), 1, -1)
            
            y_pred, _ = model(seq_input)
            
            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()
            
        if i % 25 == 0:
            print(f'Epoch: {i:3} Loss: {single_loss.item():10.8f}')
        losses.append(single_loss.item())
    return model, optimizer, losses

# 结果可视化与评估：通过对比预测曲线和实际曲线来评估模型。
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
    train_pred_np = np.array(train_pred) * data_std + data_mean 
    y_train_np = np.array([y.item() for y in y_train]) * data_std + data_mean
    test_pred_np = np.array(test_pred) * data_std + data_mean
    y_test_np = np.array([y.item() for y in y_test]) * data_std + data_mean

    plt.plot(range(len(y_train_np)), y_train_np, 'o-', label='Train Actual', color='#2E86AB', alpha=0.6)
    plt.plot(range(len(y_train_np)), train_pred_np, 's-', label='Train Predicted', color='#A23B72', alpha=0.6)
    plt.plot(range(len(y_train_np), len(y_train_np) + len(y_test_np)), y_test_np, 'o-', label='Test Actual', color='#F18F01', alpha=0.6)
    plt.plot(range(len(y_train_np), len(y_train_np) + len(y_test_np)), test_pred_np, 's-', label='Test Predicted', color='#C73E1D', alpha=0.6)
    
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('Agricultural Income Index', fontsize=12)
    plt.title('GRU Predictions vs Actual Values', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('gru_results_v0.1.png', dpi=300, bbox_inches='tight')
    print("\n结果图表已保存为: gru_results_v0.1.png")

# 主程序
def main():
    print("=== GRU 模型训练示例 ===")
    
    # 1. 加载数据
    # 请确保路径正确，如果文件在上一级目录请使用 '../RNN/...'
    file_path = '1952-1988年中国农业实际国民收入指数序列.csv'
    df = pd.read_csv(file_path)
    raw_data = df['agricultural_income'].values.astype('float32')
    
    # 2. 归一化
    data_mean = raw_data.mean()
    data_std = raw_data.std()
    data_norm = (raw_data - data_mean) / data_std
    
    # 3. 构建滑动窗口 (用过去 5 个月预测第 6 个月)
    window_size = 5
    X, y = [], []
    for i in range(len(data_norm) - window_size):
        X.append(data_norm[i : i + window_size])
        y.append(data_norm[i + window_size])
    
    # 转换为 PyTorch 张量
    X = [torch.FloatTensor(x) for x in X]
    y = [torch.FloatTensor([val]).view(1, 1) for val in y] # 形状对齐为 (1, 1) 防止广播警告

    # 4. 划分数据集
    train_size = int(len(X) * 0.8)
    train_data, test_data = X[:train_size], X[train_size:]
    train_labels, test_labels = y[:train_size], y[train_size:]
    
    # 5. 初始化模型
    model = GRUModel(input_size=1, hidden_layer_size=64, output_size=1)
    print(f"模型结构:\n{model}")
    
    # 6. 训练
    print("=== 开始训练 ===")
    model, _, losses = train_model(model, train_data, train_labels, epochs=200, lr=0.001)
    
    # 7. 预测与评估
    model.eval()
    train_predictions = []
    test_predictions = []
    with torch.no_grad():
        # 获取训练集预测结果
        for seq in train_data:
            seq_input = seq.view(len(seq), 1, -1)
            pred, _ = model(seq_input)
            train_predictions.append(pred.item())
            
        # 获取测试集预测结果
        for seq in test_data:
            seq_input = seq.view(len(seq), 1, -1)
            pred, _ = model(seq_input)
            test_predictions.append(pred.item())
            
    # 8. 结果可视化
    plot_results(losses, train_predictions, train_labels, test_predictions, test_labels, data_mean, data_std)

if __name__ == "__main__":
    main()
