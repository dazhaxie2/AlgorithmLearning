import os
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler

# 强制使用 Agg 后端
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# --- 全局参数 ---
window_size = 96     # Informer/PatchTST 通常需要更长的窗口（如96）
FUTURE_STEPS = 500
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
    return np.array(X)[..., np.newaxis], np.array(y), scaler

# --- 1. Vanilla Transformer 核心层 ---
def transformer_block(inputs, head_size, num_heads, ff_dim, name):
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads)(x, x)
    res = x + inputs
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

# --- 2. PatchTST 模拟：引入分块 (Patching) 概念 ---
def build_patch_tst(input_shape):
    inputs = keras.Input(shape=input_shape)
    # Patching: 将窗口序列切分为更小的块 (例如 patch_len=16)
    patch_len = 16
    x = layers.Reshape((window_size // patch_len, patch_len))(inputs) 
    x = layers.Dense(64)(x) # 嵌入层
    x = transformer_block(x, 64, 4, 128, "patch_1")
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(1)(x)
    return keras.Model(inputs, outputs, name="PatchTST_Sim")

# --- 3. Informer 模拟：引入蒸馏 (Distillation) 概念 ---
def build_informer(input_shape):
    inputs = keras.Input(shape=input_shape)
    x = transformer_block(inputs, 64, 4, 128, "inf_1")
    # 蒸馏层：通过 MaxPool 减半时序长度，模拟 Informer 的降采样处理
    x = layers.MaxPool1D(pool_size=2)(x) 
    x = transformer_block(x, 64, 4, 128, "inf_2")
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(1)(x)
    return keras.Model(inputs, outputs, name="Informer_Sim")

def rolling_predict(model, start_window, steps):
    curr_input = start_window.reshape(1, window_size, 1).astype(np.float32)
    preds = []
    for _ in range(steps):
        p = model.predict(curr_input, verbose=0)
        preds.append(p[0, 0])
        new_val = p.reshape(1, 1, 1)
        curr_input = np.concatenate([curr_input[:, 1:, :], new_val], axis=1)
    return np.array(preds)

def main():
    X_all, y_all_scaled, scaler = get_dataset()
    
    # 定义实验模型
    models = {
        "Vanilla Transformer": None, # 占位
        "Informer (Sim)": build_informer((window_size, 1)),
        "PatchTST (Sim)": build_patch_tst((window_size, 1))
    }
    
    # 基础 Transformer 构建
    inputs_v = keras.Input(shape=(window_size, 1))
    x_v = transformer_block(inputs_v, 64, 4, 128, "v1")
    x_v = layers.GlobalAveragePooling1D()(x_v)
    out_v = layers.Dense(1)(x_v)
    models["Vanilla Transformer"] = keras.Model(inputs_v, out_v, name="Vanilla")

    # 存储结果
    results = {}
    
    # 运行对比
    plt.figure(figsize=(16, 9))
    
    # 绘制真实值
    def denorm(d): return scaler.inverse_transform(d.reshape(-1, 1)).flatten()
    y_true = denorm(y_all_scaled[ROLL_START_IDX : ROLL_START_IDX + FUTURE_STEPS])
    plt.plot(y_true, label='Actual Price', color='black', linewidth=2, alpha=0.8)

    for name, model in models.items():
        print(f"\n正在处理模型: {name}...")
        model.compile(optimizer="adam", loss="mse")
        
        # 预热并简易训练 (为了演示对比，epochs设为较小值)
        split = int(0.8 * len(X_all))
        model.fit(X_all[:split], y_all_scaled[:split], epochs=20, batch_size=64, verbose=0)
        
        # 滚动预测
        roll_preds = rolling_predict(model, X_all[ROLL_START_IDX], FUTURE_STEPS)
        y_roll = denorm(roll_preds)
        
        # 计算 RMSE
        rmse = np.sqrt(np.mean(np.square(y_true - y_roll)))
        results[name] = rmse
        
        plt.plot(y_roll, label=f'{name} (RMSE: {rmse:.2f})')

    plt.title(f"Transformer Family Comparison: 500-Step Rolling Forecast")
    plt.xlabel("Steps from Start Index (2500)")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = 'transformer_variants_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n" + "="*30)
    for name, rmse in results.items():
        print(f"{name:20} | RMSE: {rmse:.2f}")
    print("="*30)
    print(f"对比图已保存至: {save_path}")

if __name__ == '__main__':
    main()