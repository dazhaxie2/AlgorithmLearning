import os
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from tcn.tcn import TCN

# 强制使用 Agg 后端
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# --- 实验配置 ---
ROLL_START_IDX = 2500  # 滚动预测起始点
FUTURE_STEPS = 500     # 滚动预测长度

def get_data(csv_path='./上证指数历史数据.csv', window_size=30):
    """根据不同的 window_size 准备数据"""
    df = pd.read_csv(csv_path)
    df['开盘'] = df['开盘'].astype(str).str.replace(',', '').astype(float)
    df['开盘'] = df['开盘'].ffill()
    
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df['开盘'].values.reshape(-1, 1)).flatten()
    
    X, y = [], []
    for i in range(len(data_scaled) - window_size):
        X.append(data_scaled[i : i + window_size])
        y.append(data_scaled[i + window_size])
        
    return np.array(X)[..., np.newaxis], np.array(y), scaler

def build_and_train(X, y, window_size, filter_nums, model_name):
    """构建并训练指定参数的模型"""
    model = keras.models.Sequential([
        keras.layers.Input(shape=(window_size, 1)),
        TCN(nb_filters=filter_nums, kernel_size=3, dilations=[1, 2, 4, 8]),
        keras.layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    
    print(f"\n>>> 正在训练模型: {model_name} (Window={window_size}, Filters={filter_nums})")
    # 训练集划分
    split = int(0.8 * len(X))
    model.fit(X[:split], y[:split], epochs=50, batch_size=32, verbose=0) # 实验演示使用50轮以节省时间
    return model

def rolling_predict(model, start_window, steps):
    """执行滚动预测"""
    curr_input = start_window.reshape(1, -1, 1).astype(np.float32)
    window_len = start_window.shape[0]
    preds = []
    for _ in range(steps):
        p = model.predict(curr_input, verbose=0)
        preds.append(p[0, 0])
        new_val = p.reshape(1, 1, 1)
        curr_input = np.concatenate([curr_input[:, 1:, :], new_val], axis=1)
    return np.array(preds)

def calculate_rmse(true, pred):
    """计算 RMSE 误差"""
    return np.sqrt(np.mean(np.square(true - pred)))

def run_experiment():
    # --- 实验 1: 基准参数 ---
    w1, f1 = 30, 32
    X1, y1, scaler1 = get_data(window_size=w1)
    model1 = build_and_train(X1, y1, w1, f1, "Exp_Baseline")
    
    # --- 实验 2: 增强参数 ---
    w2, f2 = 60, 128  # 增加窗口长度到60，滤波器到128
    X2, y2, scaler2 = get_data(window_size=w2)
    model2 = build_and_train(X2, y2, w2, f2, "Exp_Enhanced")
    
    # --- 提取真实值用于对比 ---
    # 获取滚动预测覆盖范围内的真实数据 (反归一化)
    # 注意：y1 的索引是从 w1 开始的，需要对齐
    true_segment_scaled = y1[ROLL_START_IDX : ROLL_START_IDX + FUTURE_STEPS]
    y_true = scaler1.inverse_transform(true_segment_scaled.reshape(-1, 1)).flatten()
    
    # --- 执行滚动预测 ---
    print("\n>>> 正在执行滚动预测对比...")
    roll1_scaled = rolling_predict(model1, X1[ROLL_START_IDX], FUTURE_STEPS)
    roll2_scaled = rolling_predict(model2, X2[ROLL_START_IDX], FUTURE_STEPS)
    
    y_roll1 = scaler1.inverse_transform(roll1_scaled.reshape(-1, 1)).flatten()
    y_roll2 = scaler2.inverse_transform(roll2_scaled.reshape(-1, 1)).flatten()
    
    # --- 计算 RMSE ---
    rmse1 = calculate_rmse(y_true, y_roll1)
    rmse2 = calculate_rmse(y_true, y_roll2)
    
    print("-" * 30)
    print(f"实验 1 (W={w1}, F={f1}) RMSE: {rmse1:.2f}")
    print(f"实验 2 (W={w2}, F={f2}) RMSE: {rmse2:.2f}")
    print("-" * 30)
    
    # --- 绘图 ---
    plt.figure(figsize=(14, 7))
    plt.plot(y_true, label='Actual Price (Ground Truth)', color='black', linewidth=2)
    plt.plot(y_roll1, label=f'Exp 1 (W={w1}, F={f1}) - RMSE: {rmse1:.2f}', linestyle='--')
    plt.plot(y_roll2, label=f'Exp 2 (W={w2}, F={f2}) - RMSE: {rmse2:.2f}', linestyle='-')
    
    plt.title(f"Impact of Window Size & Filters on 500-Step Rolling Forecast")
    plt.xlabel("Steps from Start Index (2500)")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = 'tcn_parameter_experiment.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n对比实验完成！结果图表已保存至: {save_path}")

if __name__ == '__main__':
    run_experiment()