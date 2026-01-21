import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor

# 深度学习
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tcn.tcn import TCN

# 机器学习
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# 统计模型（ARIMA 无法原生实现一次性长输出，仍采用内置多步 forecast）
from statsmodels.tsa.arima.model import ARIMA

# --- 基础配置 ---
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import matplotlib
matplotlib.use('Agg')

# 参数设置
WINDOW_SIZE = 64
FUTURE_STEPS = 72    # 一次性输出 72 个预测值
ROLL_START_IDX = 2500 
DL_EPOCHS = 100
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

# --- 1. 数据加载（生成多输出标签） ---
def load_data_direct(path='./上证指数历史数据.csv'):
    df = pd.read_csv(path)
    df['开盘'] = df['开盘'].astype(str).str.replace(',', '').astype(float)
    if df.iloc[0, 0] > df.iloc[-1, 0]: 
        df = df.iloc[::-1].reset_index(drop=True)
    df['开盘'] = df['开盘'].ffill()
    
    scaler = MinMaxScaler()
    data_raw = df['开盘'].values.reshape(-1, 1)
    data_scaled = scaler.fit_transform(data_raw).flatten()
    
    X, y = [], []
    # y 现在包含未来连续的 FUTURE_STEPS 个点
    for i in range(len(data_scaled) - WINDOW_SIZE - FUTURE_STEPS + 1):
        X.append(data_scaled[i : i + WINDOW_SIZE])
        y.append(data_scaled[i + WINDOW_SIZE : i + WINDOW_SIZE + FUTURE_STEPS])
    
    return np.array(X), np.array(y), scaler

# --- 2. 深度学习模型工厂（Direct Output 架构） ---
def build_dl_model_direct(m_type):
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
    
    # 所有深度学习模型统一输出 FUTURE_STEPS 维向量
    outputs = layers.Dense(FUTURE_STEPS)(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

# --- 3. 实验主循环 ---
def run_experiment():
    X, y, scaler = load_data_direct()
    split = int(0.8 * len(X))
    X_dl = X[..., np.newaxis]
    
    results = {}
    dl_models = ["LSTM", "GRU", "TCN", "Transformer", "Informer", "PatchTST", "TCN-Transformer"]
    ml_models = ["XGBoost", "LightGBM"]
    
    # A. 运行深度学习模型
    for m_name in dl_models:
        print(f"正在训练深度学习一次性预测模型: {m_name}")
        model = build_dl_model_direct(m_name)
        model.fit(X_dl[:split], y[:split], epochs=DL_EPOCHS, batch_size=64, verbose=0)
        
        test_input = X_dl[ROLL_START_IDX].reshape(1, WINDOW_SIZE, 1)
        results[m_name] = model.predict(test_input, verbose=0).flatten()

    # B. 运行机器学习模型 (使用 MultiOutputRegressor 包装)
    for m_name in ml_models:
        print(f"正在训练机器学习多输出模型: {m_name}")
        base_reg = XGBRegressor(n_estimators=100) if m_name == "XGBoost" else LGBMRegressor(n_estimators=100, verbosity=-1)
        # 将单输出模型转为多输出模型
        wrapper = MultiOutputRegressor(base_reg)
        wrapper.fit(X[:split], y[:split])
        
        test_input = X[ROLL_START_IDX].reshape(1, -1)
        results[m_name] = wrapper.predict(test_input).flatten()

    # C. 统计模型 (ARIMA 的长序列预测本质上是内置滚动)
    print("正在运行统计模型: ARIMA")
    history = list(X[ROLL_START_IDX])
    m_stat = ARIMA(history, order=(2, 1, 0)).fit()
    results["ARIMA"] = m_stat.forecast(FUTURE_STEPS)

    # --- 4. 绘图与排名 ---
    def denorm(d): return scaler.inverse_transform(d.reshape(-1, 1)).flatten()
    
    plt.figure(figsize=(15, 8))
    real_future = denorm(y[ROLL_START_IDX])
    plt.plot(real_future, 'k', label='真实走势 (Ground Truth)', linewidth=3)
    
    metrics = []
    for m_name, pred_scaled in results.items():
        pred_real = denorm(pred_scaled)
        plt.plot(pred_real, label=m_name, alpha=0.8)
        
        # 计算评分
        rmse = np.sqrt(mean_squared_error(real_future, pred_real))
        mae = mean_absolute_error(real_future, pred_real)
        metrics.append({"Model": m_name, "RMSE": rmse, "MAE": mae, "Score": rmse + mae})

    plt.title(f"11款模型一次性长序列预测对比 ({FUTURE_STEPS}步直接输出)", fontsize=16)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig('direct_11_models_comparison.png', bbox_inches='tight', dpi=300)
    
    # 打印排名
    df_rank = pd.DataFrame(metrics).sort_values(by="Score")
    print("\n" + "="*50 + "\n一次性长序列预测误差排名\n" + "-"*50)
    print(df_rank.to_string(index=False))
    print("="*50)

if __name__ == '__main__':
    run_experiment()