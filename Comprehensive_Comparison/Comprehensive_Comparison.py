import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 深度学习与机器学习库
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tcn.tcn import TCN
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from statsmodels.tsa.arima.model import ARIMA

# --- 基础配置 ---
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import matplotlib
matplotlib.use('Agg') 

WINDOW_SIZE = 64
FUTURE_STEPS = 72    
ROLL_START_IDX = 2500 

# --- 1. 数据加载与对齐 ---
def load_data(path='./上证指数历史数据.csv'):
    df = pd.read_csv(path)
    df['开盘'] = df['开盘'].astype(str).str.replace(',', '').astype(float)
    # 重要：如果是倒序存的数据（最新在顶端），必须反转
    if df.iloc[0,0] > df.iloc[-1,0]: 
        df = df.iloc[::-1].reset_index(drop=True)
    df['开盘'] = df['开盘'].ffill()
    
    scaler = MinMaxScaler()
    data_raw = df['开盘'].values.reshape(-1, 1)
    data_scaled = scaler.fit_transform(data_raw).flatten()
    
    X, y = [], []
    for i in range(len(data_scaled) - WINDOW_SIZE):
        X.append(data_scaled[i : i + WINDOW_SIZE])
        y.append(data_scaled[i + WINDOW_SIZE])
    return np.array(X), np.array(y), scaler, data_scaled

# --- 2. 模型构建工厂 ---
def build_dl_model(m_type):
    inputs = layers.Input(shape=(WINDOW_SIZE, 1))
    if m_type == "LSTM":
        x = layers.LSTM(64)(inputs)
    elif m_type == "GRU":
        x = layers.GRU(64)(inputs)
    elif m_type == "TCN":
        x = TCN(nb_filters=32, kernel_size=3, dilations=[1, 2, 4, 8])(inputs)
    elif m_type == "Transformer":
        x = layers.MultiHeadAttention(num_heads=4, key_dim=32)(inputs, inputs)
        x = layers.GlobalAveragePooling1D()(x)
    elif m_type == "Informer": 
        x = layers.Conv1D(32, 3, padding='same')(inputs)
        x = layers.MaxPooling1D(2)(x)
        x = layers.MultiHeadAttention(num_heads=2, key_dim=32)(x, x)
        x = layers.GlobalAveragePooling1D()(x)
    elif m_type == "PatchTST": 
        x = layers.Reshape((8, 8))(inputs) 
        x = layers.MultiHeadAttention(num_heads=2, key_dim=32)(x, x)
        x = layers.GlobalAveragePooling1D()(x)
    elif m_type == "TCN-Transformer":
        x = TCN(nb_filters=32, kernel_size=3, dilations=[1, 2], return_sequences=True)(inputs)
        x = layers.MultiHeadAttention(num_heads=2, key_dim=32)(x, x)
        x = layers.GlobalAveragePooling1D()(x)
    
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    model(np.zeros((1, WINDOW_SIZE, 1))) 
    return model

# --- 3. 排名评估逻辑 ---
def evaluate_rankings(roll_results, y_true_scaled, scaler):
    def denorm(d): return scaler.inverse_transform(d.reshape(-1, 1)).flatten()
    y_true = denorm(y_true_scaled)
    metrics = []
    for m_name, p_scaled in roll_results.items():
        length = min(len(y_true), len(p_scaled))
        p_real, t_real = denorm(p_scaled[:length]), y_true[:length]
        rmse = np.sqrt(mean_squared_error(t_real, p_real))
        mae = mean_absolute_error(t_real, p_real)
        metrics.append({"Model": m_name, "RMSE": rmse, "MAE": mae, "Score": rmse + mae})
    
    df = pd.DataFrame(metrics).sort_values(by="Score")
    print("\n" + "="*50 + "\n模型预测误差排名 (前 72 步)\n" + "-"*50)
    print(df.to_string(index=False, formatters={'RMSE': '{:,.2f}'.format, 'MAE': '{:,.2f}'.format}))
    print("="*50)
    return df

# --- 4. 主实验流程 ---
def run_experiment():
    X, y, scaler, data_scaled = load_data()
    split = int(0.8 * len(X))
    X_dl = X[..., np.newaxis]
    
    roll_results = {}
    models = ["LSTM", "GRU", "TCN", "Transformer", "Informer", "PatchTST", "TCN-Transformer", "XGBoost", "LightGBM", "ARIMA"]

    for m_name in models:
        print(f"正在运行: {m_name}...")
        if m_name in ["XGBoost", "LightGBM"]:
            reg = XGBRegressor() if m_name == "XGBoost" else LGBMRegressor(verbosity=-1)
            reg.fit(X[:split], y[:split])
            curr = X[ROLL_START_IDX].reshape(1, -1)
            preds = [reg.predict(curr)[0]]
            for _ in range(FUTURE_STEPS - 1):
                curr = np.append(curr[:, 1:], preds[-1]).reshape(1, -1)
                preds.append(reg.predict(curr)[0])
            roll_results[m_name] = np.array(preds)
        elif m_name == "ARIMA":
            history = list(data_scaled[:ROLL_START_IDX + WINDOW_SIZE])
            preds = []
            for _ in range(min(FUTURE_STEPS, 40)):
                m_stat = ARIMA(history, order=(2,1,0)).fit()
                p = m_stat.forecast()[0]
                preds.append(p); history.append(p)
            roll_results[m_name] = np.array(preds)
        else:
            model = build_dl_model(m_name)
            model.fit(X_dl[:split], y[:split], epochs=10, batch_size=64, verbose=0)
            curr = X_dl[ROLL_START_IDX].reshape(1, WINDOW_SIZE, 1)
            preds = []
            for _ in range(FUTURE_STEPS):
                p = model.predict(curr, verbose=0)[0, 0]
                preds.append(p)
                curr = np.concatenate([curr[:, 1:, :], np.array([[[p]]])], axis=1)
            roll_results[m_name] = np.array(preds)

    # 绘制结果图
    def denorm(d): return scaler.inverse_transform(d.reshape(-1, 1)).flatten()
    plt.figure(figsize=(12, 6))
    real_y = denorm(y[ROLL_START_IDX : ROLL_START_IDX + FUTURE_STEPS])
    plt.plot(real_y, 'k', label='Actual', linewidth=2)
    for m, p in roll_results.items():
        plt.plot(denorm(p), label=m, alpha=0.8)
    plt.title(f"Rolling Forecast Comparison ({FUTURE_STEPS} steps)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig('rolling_comparison.png', bbox_inches='tight')
    
    # 在函数内部执行评估并返回
    df_rank = evaluate_rankings(roll_results, y[ROLL_START_IDX : ROLL_START_IDX + FUTURE_STEPS], scaler)
    return df_rank

if __name__ == '__main__':
    df_final_rank = run_experiment()