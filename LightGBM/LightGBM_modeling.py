"""
LightGBM 回归入门示例
任务：预测房价
"""
import lightgbm as lgb
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. 加载数据
print("加载加州房价数据集...")
housing = fetch_california_housing()
X, y = housing.data, housing.target

print(f"特征: {housing.feature_names}")
print(f"数据形状: {X.shape}")
print(f"目标值范围: {y.min():.2f} - {y.max():.2f} (单位：10万美元)")

# 2. 划分数据
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. 创建数据集
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# 4. 参数配置
params = {
    'objective': 'regression',     # 回归任务
    'metric': 'rmse',              # 均方根误差
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,       # 行采样
    'bagging_freq': 5,             # 每 5 轮做一次采样
    'verbose': -1
}

# 5. 训练
print("\n开始训练...")
model = lgb.train(
    params,
    train_data,
    num_boost_round=500,
    valid_sets=[train_data, test_data],
    valid_names=['train', 'valid'],
    callbacks=[
        lgb.early_stopping(50),
        lgb.log_evaluation(100)
    ]
)

# 6. 评估
y_pred = model.predict(X_test, num_iteration=model.best_iteration)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\n模型评估:")
print(f"  RMSE: {rmse:.4f} (约 {rmse*10:.0f}00 美元误差)")
print(f"  R² Score: {r2:.4f}")
