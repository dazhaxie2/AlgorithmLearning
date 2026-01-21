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
from tensorflow.keras import layers,losses
from tcn.tcn import TCN

# 机器学习
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from statsmodels.tsa.arima.model import ARIMA

# --- 基础配置 ---
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import matplotlib
matplotlib.use('Agg')

# 参数设置
WINDOW_SIZE = 64
FUTURE_STEPS = 72
ROLL_START_IDX = 2500 
DL_EPOCHS = 150  # 增加迭代次数，配合早停
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

# --- 1. 特征工程与数据加载 ---
def create_indicators(df):
    # 移动平均线
    df['MA5'] = df['开盘'].rolling(5).mean()
    df['MA20'] = df['开盘'].rolling(20).mean()
    # 相对强弱指标 (RSI)
    delta = df['开盘'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    # 填充缺失值
    df = df.fillna(method='bfill')
    return df

def load_data_optimized(path='./上证指数历史数据.csv'):
    df = pd.read_csv(path)
    df['开盘'] = df['开盘'].astype(str).str.replace(',', '').astype(float)
    if df.iloc[0, 0] > df.iloc[-1, 0]: 
        df = df.iloc[::-1].reset_index(drop=True)
    
    # 执行特征工程
    df = create_indicators(df)
    
    # 选取的特征列
    feature_cols = ['开盘', 'MA5', 'MA20', 'RSI']
    n_features = len(feature_cols)
    
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    data_x_raw = scaler_x.fit_transform(df[feature_cols].values)
    data_y_raw = scaler_y.fit_transform(df[['开盘']].values).flatten()
    
    X, y = [], []
    for i in range(len(data_y_raw) - WINDOW_SIZE - FUTURE_STEPS + 1):
        X.append(data_x_raw[i : i + WINDOW_SIZE])
        y.append(data_y_raw[i + WINDOW_SIZE : i + WINDOW_SIZE + FUTURE_STEPS])
    
    return np.array(X), np.array(y), scaler_y, n_features

# --- 2. 深度学习模型工厂 (增强型架构) ---
def build_dl_model_optimized(m_type, n_features):
    inputs = layers.Input(shape=(WINDOW_SIZE, n_features))
    
    if m_type == "LSTM":
        x = layers.LSTM(128, return_sequences=True)(inputs)
        x = layers.Dropout(0.2)(x)
        x = layers.LSTM(64)(x)
        
    elif m_type == "GRU":
        x = layers.GRU(128, return_sequences=True)(inputs)
        x = layers.GRU(64)(x)
        
    elif m_type == "TCN":
        x = TCN(nb_filters=64, kernel_size=3, dilations=[1, 2, 4, 8, 16], 
                use_skip_connections=True)(inputs)
        
    elif m_type == "Transformer":
        # 简单位置编码
        pos_indices = tf.range(start=0, limit=WINDOW_SIZE, delta=1)
        pos_enc = layers.Embedding(input_dim=WINDOW_SIZE, output_dim=n_features)(pos_indices)
        x = inputs + pos_enc
        # Multi-Head Attention
        attn = layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
        x = layers.LayerNormalization()(attn + x)
        x = layers.Flatten()(x)
        
    elif m_type == "Informer":
        # 简化版 Informer 思想：ProbSparse Attention 的平替使用 Conv1D 提取特征
        x = layers.Conv1D(64, 3, activation='relu', padding='same')(inputs)
        x = layers.MaxPooling1D(2)(x)
        attn = layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
        x = layers.LayerNormalization()(attn + x)
        x = layers.GlobalAveragePooling1D()(x)
        
    elif m_type == "PatchTST":
        # Patching: 将 64 步分为 8 个 patch
        x = layers.Reshape((8, 8 * n_features))(inputs)
        attn = layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
        x = layers.LayerNormalization()(attn + x)
        x = layers.Flatten()(x)
        
    elif m_type == "TCN-Transformer":
        x = TCN(nb_filters=32, kernel_size=3, dilations=[1, 2, 4], return_sequences=True)(inputs)
        attn = layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
        x = layers.LayerNormalization()(attn + x)
        x = layers.Flatten()(x)

    # 统一输出层
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(FUTURE_STEPS)(x)
    
    model = keras.Model(inputs, outputs)
    # 使用 Huber Loss 应对金融数据的离群波动
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=losses.Huber())
    return model

# --- 3. 实验主循环 ---
def run_experiment():
    X, y, scaler, n_features = load_data_optimized()
    split = int(0.8 * len(X))
    
    results = {}
    dl_models = ["LSTM", "GRU", "TCN", "Transformer", "Informer", "PatchTST", "TCN-Transformer"]
    ml_models = ["XGBoost", "LightGBM"]
    
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    # A. 运行深度学习模型
    for m_name in dl_models:
        print(f"训练优化版深度学习模型: {m_name}")
        model = build_dl_model_optimized(m_name, n_features)
        model.fit(X[:split], y[:split], 
                  validation_split=0.1,
                  epochs=DL_EPOCHS, 
                  batch_size=32, 
                  callbacks=[early_stop], 
                  verbose=0)
        
        test_input = X[ROLL_START_IDX].reshape(1, WINDOW_SIZE, n_features)
        results[m_name] = model.predict(test_input, verbose=0).flatten()

    # B. 运行机器学习模型 (Flatten X 为 2D)
    X_ml = X.reshape(len(X), -1)
    for m_name in ml_models:
        print(f"训练优化版机器学习模型: {m_name}")
        if m_name == "XGBoost":
            base_reg = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, n_jobs=-1)
        else:
            base_reg = LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, verbosity=-1)
            
        wrapper = MultiOutputRegressor(base_reg)
        wrapper.fit(X_ml[:split], y[:split])
        
        test_input = X_ml[ROLL_START_IDX].reshape(1, -1)
        results[m_name] = wrapper.predict(test_input).flatten()

    # C. 统计模型
    print("运行 ARIMA (基于原始收盘价趋势)")
    history = list(y[ROLL_START_IDX-1]) # 取前一个窗口的标准化数据
    m_stat = ARIMA(history, order=(5, 1, 0)).fit()
    results["ARIMA"] = m_stat.forecast(FUTURE_STEPS)

    # --- 4. 绘图与排名 ---
    def denorm(d): return scaler.inverse_transform(d.reshape(-1, 1)).flatten()
    
    plt.figure(figsize=(15, 8))
    real_future = denorm(y[ROLL_START_IDX])
    plt.plot(real_future, 'k', label='真实走势', linewidth=3)
    
    metrics = []
    for m_name, pred_scaled in results.items():
        pred_real = denorm(pred_scaled)
        plt.plot(pred_real, label=m_name, alpha=0.7)
        rmse = np.sqrt(mean_squared_error(real_future, pred_real))
        mae = mean_absolute_error(real_future, pred_real)
        metrics.append({"Model": m_name, "RMSE": rmse, "MAE": mae, "Score": rmse + mae})

    plt.title(f"优化后 11 款模型 {FUTURE_STEPS} 步长序列预测对比", fontsize=16)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig('optimized_comparison.png', bbox_inches='tight', dpi=300)
    
    df_rank = pd.DataFrame(metrics).sort_values(by="Score")
    print("\n" + "="*50 + "\n优化后预测误差排名\n" + "-"*50)
    print(df_rank.to_string(index=False))
    print("="*50)

if __name__ == '__main__':
    run_experiment()