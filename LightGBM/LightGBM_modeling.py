import lightgbm as lgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings

# --- 基础配置 ---
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题
plt.rcParams['axes.unicode_minus'] = False 

# 1. 数据准备
housing = fetch_california_housing()
X, y = housing.data, housing.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. 训练模型
train_data = lgb.Dataset(X_train, label=y_train)
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'verbosity': -1,
    'learning_rate': 0.05,
    'num_leaves': 31
}

model = lgb.train(params, train_data, num_boost_round=300)

# 3. 预测
y_pred = model.predict(X_test)

# --- 4. 波动曲线对比可视化 ---

# 创建一个画布包含两个子图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
plt.subplots_adjust(hspace=0.3)

# --- 子图 1: 局部样本波动对比 (前 100 个样本) ---
# 这种方式可以清晰看到模型对每一个波峰和波谷的追踪能力
sample_range = 100 
ax1.plot(range(sample_range), y_test[:sample_range], label='真实值', color='#2ecc71', linewidth=2, marker='o', markersize=4)
ax1.plot(range(sample_range), y_pred[:sample_range], label='预测值', color='#e74c3c', linewidth=1.5, linestyle='--', marker='x', markersize=4)

ax1.set_title(f'局部波动追踪对比 (前 {sample_range} 个样本)', fontsize=14)
ax1.set_xlabel('样本序号')
ax1.set_ylabel('房价 (10万美元)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# --- 子图 2: 整体趋势对比 (排序后) ---
# 将真实值从小到大排序，观察预测值是否能跟上整体趋势，以及在高端房产上的表现
sort_idx = np.argsort(y_test)
y_test_sorted = y_test[sort_idx]
y_pred_sorted = y_pred[sort_idx]

# 为了不让曲线太拥挤，进行降采样显示（每 10 个点取一个）
ax2.plot(range(0, len(y_test), 10), y_test_sorted[::10], label='真实值(排序后)', color='black', alpha=0.6)
ax2.scatter(range(0, len(y_test), 10), y_pred_sorted[::10], label='预测值', color='#3498db', s=5, alpha=0.5)

ax2.set_title('整体趋势拟合对比 (按房价升序排列)', fontsize=14)
ax2.set_xlabel('排序后的样本序号')
ax2.set_ylabel('房价 (10万美元)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 5. 保存结果
save_name = 'lgb_curve_comparison.png'
plt.savefig(save_name, dpi=300, bbox_inches='tight')
print(f"波动对比图已保存为: {save_name}")

# 计算误差并输出
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"测试集 RMSE: {rmse:.4f}")