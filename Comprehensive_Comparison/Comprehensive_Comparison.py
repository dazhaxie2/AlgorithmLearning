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

# 训练配置（提高 epoch + EarlyStopping）
DL_EPOCHS = 200
DL_BATCH_SIZE = 64
DL_PATIENCE = 15
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


# --- 1. 数据加载与对齐 ---
def load_data(path='./上证指数历史数据.csv'):
    df = pd.read_csv(path)

    # 这里假设第一列是日期/时间列（你原先用 df.iloc[0,0] 判断倒序）
    # 若第一列不是日期而是数值，这个判断也可能成立；保留你的逻辑
    df['开盘'] = df['开盘'].astype(str).str.replace(',', '').astype(float)

    # 如果是倒序存的数据（最新在顶端），必须反转
    if df.iloc[0, 0] > df.iloc[-1, 0]:
        df = df.iloc[::-1].reset_index(drop=True)

    df['开盘'] = df['开盘'].ffill()

    scaler = MinMaxScaler()
    data_raw = df['开盘'].values.reshape(-1, 1)
    data_scaled = scaler.fit_transform(data_raw).flatten()

    X, y = [], []
    for i in range(len(data_scaled) - WINDOW_SIZE):
        X.append(data_scaled[i: i + WINDOW_SIZE])
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
        # (64,1) -> reshape 到 (8,8) 会丢失“通道意义”，但保留你的结构
        x = layers.Reshape((8, 8))(inputs)
        x = layers.MultiHeadAttention(num_heads=2, key_dim=32)(x, x)
        x = layers.GlobalAveragePooling1D()(x)

    elif m_type == "TCN-Transformer":
        x = TCN(nb_filters=32, kernel_size=3, dilations=[1, 2], return_sequences=True)(inputs)
        x = layers.MultiHeadAttention(num_heads=2, key_dim=32)(x, x)
        x = layers.GlobalAveragePooling1D()(x)

    else:
        raise ValueError(f"Unknown model type: {m_type}")

    # 关键：y 被 MinMax 到 [0,1]，输出建议限制在 [0,1]
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    model(np.zeros((1, WINDOW_SIZE, 1), dtype=np.float32))  # build
    return model


# --- 3. 排名评估逻辑 ---
def evaluate_rankings(roll_results, y_true_scaled, scaler):
    def denorm(d):
        return scaler.inverse_transform(d.reshape(-1, 1)).flatten()

    y_true = denorm(y_true_scaled)

    metrics = []
    for m_name, p_scaled in roll_results.items():
        length = min(len(y_true), len(p_scaled))
        p_real, t_real = denorm(p_scaled[:length]), y_true[:length]

        rmse = np.sqrt(mean_squared_error(t_real, p_real))
        mae = mean_absolute_error(t_real, p_real)
        metrics.append({"Model": m_name, "RMSE": rmse, "MAE": mae, "Score": rmse + mae})

    df = pd.DataFrame(metrics).sort_values(by="Score")
    print("\n" + "=" * 50 + "\n模型预测误差排名 (滚动预测前若干步)\n" + "-" * 50)
    print(df.to_string(index=False, formatters={'RMSE': '{:,.2f}'.format, 'MAE': '{:,.2f}'.format}))
    print("=" * 50)
    return df


# --- 4. 主实验流程 ---
def run_experiment():
    X, y, scaler, data_scaled = load_data()
    split = int(0.8 * len(X))

    # 深度学习输入需要 (N, T, C)
    X_dl = X[..., np.newaxis]

    # 建议：先做个检查，避免 FUTURE_STEPS 越界导致实际真实值不足
    if ROLL_START_IDX + FUTURE_STEPS > len(y):
        raise ValueError(
            f"ROLL_START_IDX({ROLL_START_IDX}) + FUTURE_STEPS({FUTURE_STEPS}) 超出数据范围 len(y)={len(y)}"
        )

    roll_results = {}
    models = ["LSTM", "GRU", "TCN", "Transformer", "Informer", "PatchTST",
              "TCN-Transformer", "XGBoost", "LightGBM", "ARIMA"]

    for m_name in models:
        print(f"正在运行: {m_name}...")

        if m_name in ["XGBoost", "LightGBM"]:
            reg = XGBRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=SEED
            ) if m_name == "XGBoost" else LGBMRegressor(
                n_estimators=3000,
                learning_rate=0.01,
                num_leaves=63,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=SEED,
                verbosity=-1
            )

            reg.fit(X[:split], y[:split])

            curr = X[ROLL_START_IDX].reshape(1, -1)
            preds = [reg.predict(curr)[0]]
            for _ in range(FUTURE_STEPS - 1):
                curr = np.append(curr[:, 1:], preds[-1]).reshape(1, -1)
                preds.append(reg.predict(curr)[0])

            roll_results[m_name] = np.array(preds, dtype=np.float32)

        elif m_name == "ARIMA":
            # ARIMA 用原始序列（scaled）历史滚动
            history = list(data_scaled[:ROLL_START_IDX + WINDOW_SIZE])
            preds = []
            for _ in range(FUTURE_STEPS):  # 不再限制 40 步
                m_stat = ARIMA(history, order=(2, 1, 0)).fit()
                p = float(m_stat.forecast()[0])
                preds.append(p)
                history.append(p)

            roll_results[m_name] = np.array(preds, dtype=np.float32)

        else:
            model = build_dl_model(m_name)

            cb = [
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=DL_PATIENCE,
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=max(3, DL_PATIENCE // 3),
                    min_lr=1e-6,
                    verbose=0
                )
            ]

            # 用训练集末尾切一部分做验证（避免只看 train loss）
            model.fit(
                X_dl[:split],
                y[:split],
                epochs=DL_EPOCHS,
                batch_size=DL_BATCH_SIZE,
                verbose=0,
                validation_split=0.1,
                callbacks=cb
            )

            curr = X_dl[ROLL_START_IDX].reshape(1, WINDOW_SIZE, 1)
            preds = []
            for _ in range(FUTURE_STEPS):
                p = float(model.predict(curr, verbose=0)[0, 0])
                preds.append(p)
                curr = np.concatenate([curr[:, 1:, :], np.array([[[p]]], dtype=np.float32)], axis=1)

            roll_results[m_name] = np.array(preds, dtype=np.float32)

    # 绘制结果图（时间轴对齐到 y 的全局索引）
    def denorm(d):
        return scaler.inverse_transform(d.reshape(-1, 1)).flatten()

    start = ROLL_START_IDX
    end = ROLL_START_IDX + FUTURE_STEPS
    t = np.arange(start, end)

    plt.figure(figsize=(12, 6))
    real_y = denorm(y[start:end])
    plt.plot(t, real_y, 'k', label='Actual', linewidth=2)

    for m, p in roll_results.items():
        tt = np.arange(start, start + len(p))
        plt.plot(tt, denorm(p), label=m, alpha=0.85)

    plt.title(f"Rolling Forecast Comparison ({FUTURE_STEPS} steps)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig('rolling_comparison.png', bbox_inches='tight', dpi=160)
    plt.close()

    # 评估
    df_rank = evaluate_rankings(roll_results, y[start:end], scaler)
    return df_rank


if __name__ == '__main__':
    df_final_rank = run_experiment()
    # 可选：保存排名
    df_final_rank.to_csv("model_rankings.csv", index=False, encoding="utf-8-sig")