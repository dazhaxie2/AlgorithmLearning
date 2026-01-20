import os
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler

# 强制使用 Agg 后端（不显示 GUI 窗口）
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# --- 全局参数 ---
window_size = 60    # 增加窗口大小，发挥 Transformer 全局注意力优势
d_model = 64        # 特征维度
num_heads = 4       # 多头注意力的头数
ff_dim = 128        # 前馈网络维度
epochs = 200
FUTURE_STEPS = 500  # 滚动预测 500 步
ROLL_START_IDX = 2500

def get_dataset(csv_path='./上证指数历史数据.csv'):
    df = pd.read_csv(csv_path)
    df['开盘'] = df['开盘'].astype(str).str.replace(',', '').astype(float)
    df['开盘'] = df['开盘'].ffill()
    
    scaler = MinMaxScaler()
    data = scaler.fit_transform(df['开盘'].values.reshape(-1, 1)).flatten()
    
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i : i + window_size])
        y.append(data[i + window_size])
    
    X, y = np.array(X), np.array(y)
    split = int(0.8 * len(X))
    return X[:split, ..., np.newaxis], y[:split], X[split:, ..., np.newaxis], y[split:], scaler, X[..., np.newaxis]

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    """构建单层 Transformer Encoder"""
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def build_transformer_model():
    """构建基于 Transformer 的回归模型"""
    inputs = keras.Input(shape=(window_size, 1))
    
    # 编码层：堆叠 2 层 Encoder
    x = transformer_encoder(inputs, d_model, num_heads, ff_dim, dropout=0.1)
    x = transformer_encoder(x, d_model, num_heads, ff_dim, dropout=0.1)
    
    # 全局池化 + 全连接输出
    x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(1, activation="linear")(x)
    
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")
    return model

def rolling_predict(model, start_window, steps):
    """执行滚动预测逻辑"""
    curr_input = start_window.reshape(1, window_size, 1).astype(np.float32)
    preds = []
    print(f"正在执行 Transformer {steps} 步滚动预测...")
    for _ in range(steps):
        p = model.predict(curr_input, verbose=0)
        preds.append(p[0, 0])
        new_val = p.reshape(1, 1, 1)
        curr_input = np.concatenate([curr_input[:, 1:, :], new_val], axis=1)
    return np.array(preds)

def main():
    # 1. 数据准备
    train_X, train_y, test_X, test_y, scaler, X_all = get_dataset()
    
    # 2. 模型训练/加载
    model = build_transformer_model()
    # 热启动：修复 Rank 问题
    model.predict(np.zeros((1, window_size, 1)), verbose=0)
    
    weights_path = 'transformer_stock.weights.h5'
    if not os.path.exists(weights_path):
        print("开始训练 Transformer 模型...")
        model.fit(train_X, train_y, validation_split=0.1, epochs=epochs, batch_size=32)
        model.save_weights(weights_path)
        joblib.dump(scaler, 'tr_scaler.pkl')
    else:
        print("加载已保存的权重...")
        model.load_weights(weights_path)
        scaler = joblib.load('tr_scaler.pkl')

    # 3. 预测计算
    # 逐步预测 (One-step)
    step_preds = model.predict(X_all, verbose=0).flatten()
    
    # 滚动预测 (Rolling) - 从索引 2500 开始
    start_window = X_all[ROLL_START_IDX]
    roll_preds_scaled = rolling_predict(model, start_window, FUTURE_STEPS)

    # 4. 反归一化
    def denorm(d): return scaler.inverse_transform(d.reshape(-1, 1)).flatten()
    y_true_full = denorm(np.concatenate([train_y, test_y]))
    y_step = denorm(step_preds)
    y_roll = denorm(roll_preds_scaled)

    # 5. 计算滚动段 RMSE
    target_true = y_true_full[ROLL_START_IDX : ROLL_START_IDX + FUTURE_STEPS]
    rmse = np.sqrt(np.mean(np.square(target_true - y_roll)))

    # 6. 绘图保存
    plt.figure(figsize=(15, 8))
    plt.plot(y_true_full, label='Actual Price', color='black', alpha=0.3)
    plt.plot(y_step, label='One-step (Step-by-step)', color='blue', alpha=0.5, linestyle='--')
    
    roll_range = range(ROLL_START_IDX, ROLL_START_IDX + FUTURE_STEPS)
    plt.plot(roll_range, y_roll, label=f'Transformer Rolling (RMSE:{rmse:.2f})', color='red', linewidth=2)
    
    plt.axvline(x=ROLL_START_IDX, color='green', linestyle=':', label='Forecast Start')
    plt.title(f"Transformer Stock Prediction: Rolling 500 Steps (Window={window_size})")
    plt.legend()
    plt.grid(True, alpha=0.2)
    
    save_path = 'transformer_rolling_results.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[任务完成] 图表已保存至: {save_path}, RMSE 为: {rmse:.2f}")

if __name__ == '__main__':
    main()