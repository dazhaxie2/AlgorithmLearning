import os
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from tcn.tcn import TCN

# 强制 Matplotlib 使用 Agg 后端（不弹出窗口）
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# --- 全局超参数 ---
window_size = 30
filter_nums = 64
kernel_size = 3
epochs = 300
ROLL_START_IDX = 2500  # 滚动预测的起始索引
FUTURE_STEPS = 500     # 滚动预测步数

def get_full_processed_data(csv_path='./上证指数历史数据.csv'):
    """获取全量处理后的数据、特征矩阵和归一化器"""
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

def build_model_fn():
    """构建 TCN 顺序模型"""
    model = keras.models.Sequential([
        keras.layers.Input(shape=(window_size, 1)),
        TCN(nb_filters=filter_nums, kernel_size=kernel_size, dilations=[1, 2, 4, 8]),
        keras.layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def rolling_predict(model, start_window, steps):
    """从指定窗口开始执行长时间步滚动预测"""
    curr_input = start_window.reshape(1, window_size, 1).astype(np.float32)
    preds = []
    for _ in range(steps):
        p = model.predict(curr_input, verbose=0)
        preds.append(p[0, 0])
        new_val = p.reshape(1, 1, 1)
        # 将预测值拼接至窗口末尾，剔除首位
        curr_input = np.concatenate([curr_input[:, 1:, :], new_val], axis=1)
    return np.array(preds)

def train_model():
    """执行训练（若权重不存在）"""
    weights_path = 'tcn_model.weights.h5'
    if os.path.exists(weights_path):
        return
    
    X, y, scaler, _ = get_full_processed_data()
    split = int(0.8 * len(X))
    model = build_model_fn()
    model.fit(X[:split], y[:split], validation_split=0.1, epochs=epochs, batch_size=32)
    model.save_weights(weights_path)
    joblib.dump(scaler, 'scaler.pkl')

def predict_and_compare():
    """主逻辑：逐步预测 vs 滚动预测"""
    # 1. 加载模型与预热
    model = build_model_fn()
    model.predict(np.zeros((1, window_size, 1)), verbose=0) 
    model.load_weights('tcn_model.weights.h5')
    scaler = joblib.load('scaler.pkl')
    
    # 2. 获取数据（全量数据用于对比展示）
    X_all, y_true_scaled, _, _ = get_full_processed_data()
    
    # 3. 逐步预测 (One-step Prediction) - 基于真实历史
    print("正在执行逐步预测...")
    step_by_step_preds = model.predict(X_all, verbose=1).flatten()
    
    # 4. 滚动预测 (Rolling Prediction) - 从 2500 步开始
    # 注意：X_all[i] 是预测 y_true_scaled[i] 的输入窗口
    start_window = X_all[ROLL_START_IDX]
    roll_preds_scaled = rolling_predict(model, start_window, FUTURE_STEPS)
    
    # 5. 反归一化
    def denorm(d): return scaler.inverse_transform(d.reshape(-1, 1)).flatten()
    y_true = denorm(y_true_scaled)
    y_step = denorm(step_by_step_preds)
    y_roll = denorm(roll_preds_scaled)
    
    # 6. 绘图保存
    plt.figure(figsize=(16, 8))
    
    # 绘制真实曲线
    plt.plot(y_true, label='Actual Price', color='black', alpha=0.3, linewidth=1)
    
    # 绘制全量逐步预测（模型在已知数据上的表现）
    plt.plot(y_step, label='One-step Prediction', color='blue', alpha=0.6, linestyle='--')
    
    # 绘制从 2500 开始的滚动预测（由于是从索引2500开始向后预测，其起点对应真实数据的 2500 处）
    roll_range = range(ROLL_START_IDX, ROLL_START_IDX + FUTURE_STEPS)
    plt.plot(roll_range, y_roll, label=f'Rolling Forecast from Index {ROLL_START_IDX}', color='red', linewidth=2)
    
    # 标记起点
    plt.axvline(x=ROLL_START_IDX, color='green', linestyle=':', label='Rolling Start Point')
    
    plt.title(f"TCN Comparison: One-step vs Rolling Forecast (Steps={FUTURE_STEPS})")
    plt.xlabel("Sample Index")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, alpha=0.2)
    
    save_path = 'tcn_comparison_result.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[成功] 滚动预测完成。对比图已保存至: {save_path}")

if __name__ == '__main__':
    # train_model()
    predict_and_compare()