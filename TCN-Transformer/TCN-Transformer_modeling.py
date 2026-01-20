import os
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tcn.tcn import TCN
from sklearn.preprocessing import MinMaxScaler

# 强制使用 Agg 后端（无窗口模式）
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# --- 全局超参数 ---
window_size = 64
filter_nums = 64
num_heads = 4
ff_dim = 128
epochs = 150
FUTURE_STEPS = 500
ROLL_START_IDX = 2500

def get_dataset(csv_path='./上证指数历史数据.csv'):
    """复用数据集读取逻辑"""
    df = pd.read_csv(csv_path)
    df['开盘'] = df['开盘'].astype(str).str.replace(',', '').astype(float)
    df['开盘'] = df['开盘'].ffill()
    
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df['开盘'].values.reshape(-1, 1)).flatten()
    
    X, y = [], []
    for i in range(len(data_scaled) - window_size):
        X.append(data_scaled[i : i + window_size])
        y.append(data_scaled[i + window_size])
        
    return np.array(X)[..., np.newaxis], np.array(y), scaler, data_scaled

def build_hybrid_model():
    """核心：TCN + Transformer 混合架构"""
    inputs = keras.Input(shape=(window_size, 1))
    
    # 1. TCN 层：提取局部时序形态（类似 K 线组合特征）
    # 设置 return_sequences=True 以便对接 Transformer 层
    x = TCN(nb_filters=filter_nums, 
            kernel_size=3, 
            dilations=[1, 2, 4, 8], 
            return_sequences=True,
            name="TCN_Extractor")(inputs)
    
    # 2. Transformer 层：捕捉全局长程依赖
    # Self-Attention
    attn_output = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=filter_nums
    )(x, x)
    x = layers.Add()([x, attn_output]) # 残差连接
    x = layers.LayerNormalization()(x)
    
    # Feed Forward
    ff_output = layers.Dense(ff_dim, activation="relu")(x)
    ff_output = layers.Dense(filter_nums)(ff_output)
    x = layers.Add()([x, ff_output])
    x = layers.LayerNormalization()(x)
    
    # 3. 输出层
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(1)(x)
    
    model = keras.Model(inputs, outputs, name="TCN_Transformer_Hybrid")
    model.compile(optimizer="adam", loss="mse")
    return model

def rolling_predict(model, start_window, steps):
    """递归滚动预测"""
    curr_input = start_window.reshape(1, window_size, 1).astype(np.float32)
    preds = []
    for _ in range(steps):
        p = model.predict(curr_input, verbose=0)
        preds.append(p[0, 0])
        new_val = p.reshape(1, 1, 1)
        curr_input = np.concatenate([curr_input[:, 1:, :], new_val], axis=1)
    return np.array(preds)

def main():
    # 1. 准备数据
    X_all, y_true_scaled, scaler, raw_data_scaled = get_dataset()
    
    # 2. 初始化与热启动（解决 Rank 问题）
    model = build_hybrid_model()
    model.predict(np.zeros((1, window_size, 1)), verbose=0)
    
    weights_path = 'hybrid_model.weights.h5'
    if not os.path.exists(weights_path):
        print("开始训练 TCN-Transformer 混合模型...")
        split = int(0.8 * len(X_all))
        model.fit(X_all[:split], y_true_scaled[:split], 
                  validation_split=0.1, epochs=epochs, batch_size=64)
        model.save_weights(weights_path)
        joblib.dump(scaler, 'hybrid_scaler.pkl')
    else:
        print("加载混合模型权重...")
        model.load_weights(weights_path)
        scaler = joblib.load('hybrid_scaler.pkl')

    # 3. 逐步预测与滚动预测
    print("正在执行 500 步滚动预测...")
    step_preds_scaled = model.predict(X_all, verbose=0).flatten()
    roll_preds_scaled = rolling_predict(model, X_all[ROLL_START_IDX], FUTURE_STEPS)

    # 4. 反归一化
    def denorm(d): return scaler.inverse_transform(d.reshape(-1, 1)).flatten()
    y_true = denorm(y_true_scaled)
    y_step = denorm(step_preds_scaled)
    y_roll = denorm(roll_preds_scaled)

    # 5. 计算 RMSE（仅针对滚动预测段）
    target_true = y_true[ROLL_START_IDX : ROLL_START_IDX + FUTURE_STEPS]
    rmse = np.sqrt(np.mean(np.square(target_true - y_roll)))

    # 6. 绘图对比
    plt.figure(figsize=(16, 8))
    
    # 真实价格
    plt.plot(y_true, label='Actual Price', color='black', alpha=0.25, linewidth=1)
    
    # 逐步预测 (基于真实窗口)
    plt.plot(y_step, label='Hybrid One-step (Benchmark)', color='blue', alpha=0.4, linestyle='--')
    
    # 滚动预测 (500步长线)
    roll_range = range(ROLL_START_IDX, ROLL_START_IDX + FUTURE_STEPS)
    plt.plot(roll_range, y_roll, label=f'Hybrid Rolling Forecast (RMSE: {rmse:.2f})', color='red', linewidth=2)
    
    plt.axvline(x=ROLL_START_IDX, color='green', linestyle=':', label='Forecast Start Index 2500')
    
    plt.title(f"TCN-Transformer Hybrid: 500-Step Rolling Forecast Analysis")
    plt.xlabel("Sample Index")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, alpha=0.2)
    
    save_path = 'hybrid_tcn_transformer_results.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n[任务完成] 对比图已保存为: {save_path}")
    print(f"滚动预测段 RMSE: {rmse:.2f}")

if __name__ == '__main__':
    main()