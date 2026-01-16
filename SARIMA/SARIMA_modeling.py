import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import itertools
import warnings
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae

# --- 配置项 ---
warnings.filterwarnings("ignore") 
# 设置中文字体，如果报错请尝试更换为 'Microsoft YaHei' 或 'SimHei'
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False 

# ==========================================
# 1. 数据准备
# ==========================================
print("正在读取数据...")
try:
    df = pd.read_csv("MER_T02_06.csv", index_col="YYYYMM")
except FileNotFoundError:
    print("错误：未找到文件 MER_T02_06.csv，请确认文件路径。")
    exit()

# 数据筛选
key_list = ['Coal Consumed by the Electric Power Sector']
df = df[df['Description'].isin(key_list)]

# 索引处理
df.index = df.index.astype(str)
# 过滤掉季度数据（索引最后两位大于12的）
df = df[df.index.str[-2:].astype(int) <= 12]

# 数据清洗
df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
df = df.dropna(subset=['Value'])

# 转换为Datetime索引并设置频率为月初('MS')
# 这是时间序列模型非常关键的一步
df.index = pd.to_datetime(df.index, format='%Y%m')
df = df.asfreq('MS') 

# 提取目标序列
CCE = df['Value']
print(f"数据加载完成，时间范围: {CCE.index[0].date()} 到 {CCE.index[-1].date()}")

# 可视化原始数据
fig, ax = plt.subplots(figsize=(15, 8))
CCE.plot(ax=ax, fontsize=15)
ax.set_title('电力行业碳排放 (原始数据)', fontsize=20)
ax.set_ylabel('碳排放量', fontsize=15)
ax.grid()
plt.savefig('1_电力行业碳排放_原始.png', dpi=300, bbox_inches='tight')
plt.close()

# ==========================================
# 2. 平稳性检验 (辅助判断)
# ==========================================
# 注意：这里我们只做检验来确定 d 的值，但后续模型训练依然喂原始数据 CCE
def test_stationarity(timeseries, label=""):
    dftest = ADF(timeseries.dropna())
    print(f"[{label}] ADF检验 p值: {dftest[1]:.6f}")
    return dftest[1] < 0.05

print("\n--- 平稳性检验 ---")
test_stationarity(CCE, "原始数据")
test_stationarity(CCE.diff(1), "一阶差分")
test_stationarity(CCE.diff(1).diff(12), "一阶+季节差分")
# 结论通常是：原始不平稳，差分后平稳。所以模型参数 d=1, D=1 是合理的。

# ==========================================
# 3. 模型定阶 (基于原始数据搜索)
# ==========================================
def SARIMA_search(data):
    # 定义p, d, q的范围
    # d=1, D=1 通常是固定的（基于上面的ADF分析）
    # s=12 是固定的（月度数据）
    p = q = range(0, 3) # 范围可以根据算力适当调整，如 range(0, 3)
    P = Q = range(0, 3)
    
    pdq = list(itertools.product(p, [1], q))        # d 固定为 1
    seasonal_pdq = list(itertools.product(P, [1], Q, [12])) # D 固定为 1, s 固定为 12
    
    best_aic = float('inf')
    best_param = None
    best_seasonal_param = None
    
    print("\n开始网格搜索最佳参数 (AIC准则)...")
    for param in pdq:
        for seasonal_param in seasonal_pdq:
            try:
                # 关键修正：这里传入的是原始数据 data (CCE)，而不是差分后的数据
                mod = sm.tsa.SARIMAX(data,
                                     order=param,
                                     seasonal_order=seasonal_param,
                                     enforce_stationarity=False,
                                     enforce_invertibility=False)
                results = mod.fit(disp=False, maxiter=50) # disp=False 关闭详细输出
                
                print(f'ARIMA{param}x{seasonal_param} - AIC:{results.aic:.2f}')
                
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_param = param
                    best_seasonal_param = seasonal_param
            except:
                continue
                
    print(f"\n最佳模型参数: ARIMA{best_param}x{best_seasonal_param} - AIC:{best_aic:.2f}")
    return best_param, best_seasonal_param

# 执行搜索 (如果你想节省时间，可以注释掉下面这行，直接使用推荐参数)
# best_order, best_seasonal_order = SARIMA_search(CCE)

# 最佳模型参数: ARIMA(2, 1, 2)x(2, 1, 2, 12) - AIC:6428.43
# 如果不想每次都搜索，可以手动指定（假设这是搜索出来的结果）：
best_order = (2, 1, 2)
best_seasonal_order = (2, 1, 2, 12)

# ==========================================
# 4. 模型建立与训练
# ==========================================
print("\n正在训练最佳模型...")
# 关键修正：传入 CCE (原始数据)
model = sm.tsa.SARIMAX(CCE, 
                       order=best_order, 
                       seasonal_order=best_seasonal_order,
                       enforce_stationarity=False,
                       enforce_invertibility=False)
SARIMA_m = model.fit()
print(SARIMA_m.summary())

# 残差诊断
fig = SARIMA_m.plot_diagnostics(figsize=(15, 12))
plt.savefig('2_SARIMA模型诊断.png', dpi=300, bbox_inches='tight')
plt.close()

# 白噪声检验
lb_result = acorr_ljungbox(SARIMA_m.resid, lags=[12], return_df=True)
print("\n残差白噪声检验 (Ljung-Box):")
print(lb_result)

# ==========================================
# 5. 模型预测与评估
# ==========================================

def PredictionAnalysis(data, model_res, start_date, dynamic=False, label_suffix=""):
    # 获取预测结果
    pred_res = model_res.get_prediction(start=pd.to_datetime(start_date), dynamic=dynamic)
    pred_mean = pred_res.predicted_mean
    pred_ci = pred_res.conf_int()
    
    # 获取对应时间段的真实值
    truth = data[pred_mean.index]
    
    # 计算误差
    mse_val = mse(truth, pred_mean)
    rmse_val = np.sqrt(mse_val)
    mae_val = mae(truth, pred_mean)
    
    print(f"\n[{label_suffix}] 预测评估:")
    print(f"  MSE:  {mse_val:.4f}")
    print(f"  RMSE: {rmse_val:.4f}")
    print(f"  MAE:  {mae_val:.4f}")
    
    # 绘图
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 画全量历史数据
    ax.plot(data.index, data, label='历史真实值', color='#1f77b4')
    
    # 画预测数据
    ax.plot(pred_mean.index, pred_mean, label='预测值', color='#ff7f0e', linewidth=2)
    
    # 画置信区间
    ax.fill_between(pred_ci.index, 
                    pred_ci.iloc[:, 0], 
                    pred_ci.iloc[:, 1], color='gray', alpha=0.2, label='95%置信区间')
    
    # 局部放大显示（只显示预测开始前后的数据）
    zoom_start = pd.to_datetime(start_date) - pd.DateOffset(months=24)
    ax.set_xlim(left=zoom_start)
    
    ax.set_title(f'SARIMA 预测结果 ({label_suffix})', fontsize=18)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    return fig

# 设定预测切分点 (建议设在数据末尾的前2-3年，用于验证)
# 假设数据到2023年，我们从2021年开始预测
split_date = '2021-01-01'

# 1. 静态预测 (Static Forecast) - 这里的每一步都用了上一步的真实值
fig_static = PredictionAnalysis(CCE, SARIMA_m, split_date, dynamic=False, label_suffix="静态预测")
fig_static.savefig('3_SARIMA模型静态预测.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. 动态预测 (Dynamic Forecast) - 从split_date开始，完全依赖模型之前的预测值
fig_dynamic = PredictionAnalysis(CCE, SARIMA_m, split_date, dynamic=True, label_suffix="动态预测")
fig_dynamic.savefig('4_SARIMA模型动态预测.png', dpi=300, bbox_inches='tight')
plt.close()

# ==========================================
# 6. 未来预测 (Out-of-sample Forecast)
# ==========================================
print("\n生成未来24个月的预测...")
# 获取未来步数
steps = 24
pred_future = SARIMA_m.get_forecast(steps=steps)
pred_mean = pred_future.predicted_mean
pred_ci = pred_future.conf_int()

fig, ax = plt.subplots(figsize=(15, 8))
# 画过去5年的数据
ax.plot(CCE.index[-60:], CCE[-60:], label='历史数据 (近5年)')
# 画未来预测
ax.plot(pred_mean.index, pred_mean, label='未来预测', color='red')
ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='pink', alpha=0.3)
ax.set_title('未来24个月电力行业碳排放预测', fontsize=20)
ax.legend()
ax.grid()
plt.savefig('5_未来预测趋势图.png', dpi=300, bbox_inches='tight')
plt.close()

print("所有步骤已完成，图片已保存。")