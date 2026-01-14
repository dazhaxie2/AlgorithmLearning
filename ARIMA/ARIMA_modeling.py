import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
warnings.filterwarnings("ignore")

# 设置中文字体和负号正常显示
rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为SimHei(黑体)
rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 确保使用非GUI后端（用于服务器环境）
matplotlib.use('Agg')

# 1. 创建示例数据（如果CSV文件不存在）
def create_sample_data():
    """创建示例数据，用于演示（实际使用时请替换为真实数据）"""
    years = list(range(1952, 1989))
    # 1952年为基准100，模拟1952-1988年农业收入指数
    agricultural_income = [100 + 2.5 * (y - 1952) + 5 * np.sin((y - 1952) * 0.2) for y in years]
    # 添加一些随机波动
    np.random.seed(42)
    agricultural_income = [x + np.random.normal(0, 5) for x in agricultural_income]
    
    # 创建DataFrame
    df = pd.DataFrame({
        'year': years,
        'agricultural_income': agricultural_income
    })
    df.to_csv("1952-1988年中国农业实际国民收入指数序列.csv", index=False)
    print("示例数据已创建，保存为 '1952-1988年中国农业实际国民收入指数序列.csv'")
    return df

# 尝试读取数据，如果不存在则创建示例数据
try:
    data = pd.read_csv("1952-1988年中国农业实际国民收入指数序列.csv")
    print("成功加载数据文件")
except FileNotFoundError:
    print("数据文件未找到，正在创建示例数据...")
    data = create_sample_data()

# 清理列名
data.columns = data.columns.str.strip()
# 设置年份为索引
data.set_index('year', inplace=True)

# 确保数据是数值类型
data['agricultural_income'] = pd.to_numeric(data['agricultural_income'], errors='coerce')

# 检查数据是否包含缺失值
if data['agricultural_income'].isnull().any():
    print("警告：数据包含缺失值，将进行前向填充")
    data['agricultural_income'] = data['agricultural_income'].ffill()

# 1. 平稳性检验与差分处理
print("="*50)
print("平稳性检验与差分处理")
print("="*50)

# 原始序列平稳性检验
adf_test = adfuller(data['agricultural_income'])
print("ADF 统计量 (原始序列): {:.2f}".format(adf_test[0]))
print("p 值 (原始序列): {:.5f}".format(adf_test[1]))
print("临界值:")
for key, value in adf_test[4].items():
    print(f"\t{key}: {value:.2f}")

# 动态差分直到序列平稳
diff_data = data['agricultural_income'].copy()
d = 0
adf_results = []
adf_results.append(("原始序列", adf_test[0], adf_test[1]))

while adf_test[1] > 0.05:
    diff_data = diff_data.diff().dropna()
    d += 1
    adf_test = adfuller(diff_data)
    adf_results.append((f"{d}阶差分", adf_test[0], adf_test[1]))
    print(f"{d}阶差分后ADF检验: p值={adf_test[1]:.5f}")

# 确认最终差分阶数
print("\n最终差分阶数 d = {}".format(d))
print(f"最终序列ADF检验: p值={adf_results[-1][2]:.5f}")

# 2. 绘制原始序列和差分后序列
plt.figure(figsize=(14, 10))

# 原始序列
plt.subplot(2, 1, 1)
plt.plot(data.index, data['agricultural_income'], 'b-')
plt.title('原始序列 (1952-1988)')
plt.xlabel('年份')
plt.ylabel('农业实际国民收入指数')
plt.grid(True)

# 差分后序列
plt.subplot(2, 1, 2)
plt.plot(diff_data.index, diff_data, 'r-')
plt.title(f'差分后序列 (d={d})')
plt.xlabel('年份')
plt.ylabel('差分值')
plt.grid(True)

plt.tight_layout()
plt.savefig('time_series_diff.png', dpi=300, bbox_inches='tight')
print("\n序列图已保存为 time_series_diff.png")

# 3. ACF和PACF分析（仅对平稳序列）
plt.figure(figsize=(12, 8))
plt.suptitle('ACF和PACF图 (平稳序列)', fontsize=16)

# ACF图
plt.subplot(2, 1, 1)
plot_acf(diff_data, lags=15, ax=plt.gca())
plt.title('自相关函数 (ACF)')

# PACF图
plt.subplot(2, 1, 2)
plot_pacf(diff_data, lags=15, ax=plt.gca())
plt.title('偏自相关函数 (PACF)')

plt.tight_layout()
plt.savefig('acf_pacf_plots.png', dpi=300, bbox_inches='tight')
print("\nACF和PACF图已保存为 acf_pacf_plots.png")

# 4. 参数选择指南
print("\n--- 参数选择指南 ---")
print("根据PACF图确定p值（自回归阶数）：")
print("- 如果PACF在第p个滞后后截尾（即落在置信区间内），则p为该滞后数")
print("- 如果PACF逐渐衰减，则可能需要尝试不同的p值")

print("\n根据ACF图确定q值（移动平均阶数）：")
print("- 如果ACF在第q个滞后后截尾，则q为该滞后数")
print("- 如果ACF逐渐衰减，则可能需要尝试不同的q值")

# 5. 使用多个信息准则寻找最佳参数
print("\n--- 使用多个信息准则确定最优参数 ---")
print("正在搜索最优的p、d和q参数组合...")

best_criteria_score = float('inf')
best_param = None
best_model = None

# 使用之前动态确定的差分阶数，但限制最大值以避免过度差分
max_d = min(2, d)  # 限制最大差分阶数为2，防止过度差分
print(f"使用差分阶数范围: 0 到 {max_d}")

# 测试不同的差分阶数组合，避免过度差分
d_range = range(0, max_d + 1)  # 限制差分阶数，避免过度差分
p_range = range(0, 4)  # p的范围: 0到3
q_range = range(0, 4)  # q的范围: 0到3

for d in d_range:
    # 对每个差分阶数，计算差分后的序列
    if d == 0:
        diff_data_test = data['agricultural_income']
    else:
        diff_data_test = data['agricultural_income'].copy()
        for i in range(d):
            diff_data_test = diff_data_test.diff().dropna()
    
    # 检查差分后的序列是否平稳
    try:
        adf_result = adfuller(diff_data_test)
        print(f"差分阶数d={d}后ADF检验: p差={adf_result[1]:.5f}")
        # 不再跳过不完全平稳的情况，而是继续尝试建模
    except:
        print(f"差分阶时d={d}时ADF检验失败")
        continue
    
    for p in p_range:
        for q in q_range:
            try:
                model = ARIMA(data['agricultural_income'], order=(p, d, q))
                fitted_model = model.fit()

                # 使用综合信息准则评分 (AIC, BIC, HQIC的加权平均)
                aic = fitted_model.aic
                bic = fitted_model.bic
                hqic = fitted_model.hqic

                # 计算综合评分 (越小越好)
                criteria_score = (aic + bic + hqic) / 3

                # 额外考虑残差白噪声检验，惩罚残差不是白噪声的模型
                residuals_test = fitted_model.resid
                try:
                    lb_test = acorr_ljungbox(residuals_test, lags=min(10, len(residuals_test)//5), return_df=True)
                    ljung_box_pvalue = lb_test.iloc[0]['lb_pvalue']
                    
                    # 如果残差不是白噪声，增加惩罚
                    if ljung_box_pvalue <= 0.05:
                        criteria_score += 50  # 惩罚项
                        
                except Exception:
                    # 如果Ljung-Box检验失败，增加惩罚
                    criteria_score += 100

                print(f'ARIMA({p},{d},{q}) - AIC: {aic:.2f}, BIC: {bic:.2f}, HQIC: {hqic:.2f}, 综合评分: {criteria_score:.2f}')
                
                if criteria_score < best_criteria_score:
                    best_criteria_score = criteria_score
                    best_param = (p, d, q)
                    best_model = fitted_model
                    print(f"   -> 发现更优模型: ARIMA{best_param}, 综合评分: {best_criteria_score:.2f}")
                    
            except Exception as e:
                print(f'ARIMA({p},{d},{q}) 拟合失败: {str(e)}')
                continue

if best_param is None:
    print("未能找到满足条件的模型，尝试使用原始确定的差分阶数...")
    # 回退到原来的方法
    d_original = d  # 使用原始确定的差分阶数
    best_aic = float('inf')
    
    for p in p_range:
        for q in q_range:
            try:
                model = ARIMA(data['agricultural_income'], order=(p, d_original, q))
                fitted_model = model.fit()
                aic = fitted_model.aic
                
                print(f'ARIMA({p},{d_original},{q}) AIC: {aic:.2f}')
                
                if aic < best_aic:
                    best_aic = aic
                    best_param = (p, d_original, q)
                    best_model = fitted_model
                    
            except Exception as e:
                print(f'ARIMA({p},{d_original},{q}) 拟合失败: {str(e)}')
                continue

if best_param is None:
    print("仍然无法找到合适的模型参数，请检查数据")
    exit()

print(f'\n最佳ARIMA模型参数: {best_param}')
print(f'对应的综合评分: {best_criteria_score:.2f}' if best_criteria_score != float('inf') else f'对应的AIC값: {best_aic:.2f}')

# 6. 模型总结与建议
print(f'\n--- 最终参数建议 ---')
print(f'差分阶数 d = {best_param[1]} (基于平稳性检验)')
print(f'自回归阶数 p = {best_param[0]} (基于多准则优化)' if best_criteria_score != float('inf') else f'自回归阶数 p = {best_param[0]} (基于AIC准则)')
print(f'移动平均阶数 q = {best_param[2]} (基于多准则优化)' if best_criteria_score != float('inf') else f'移动平均阶数 q = {best_param[2]} (基于AIC准则)')
print(f'推荐使用 ARIMA{best_param} 模型')

# 7. 使用最佳参数拟合模型
print(f'\n--- 拟合ARIMA{best_param}模型 ---')
model_fit = ARIMA(data['agricultural_income'], order=best_param)
model_fit = model_fit.fit()

# 打印模型摘要
print(model_fit.summary())

# 8. 残差分析
residuals = model_fit.resid
print("\n--- 残差分析 ---")

# 残差白噪声检验 (Ljung-Box)
lb_test = acorr_ljungbox(residuals, lags=10, return_df=True)
print(f"Ljung-Box检验 (滞后10): p值 = {lb_test.iloc[0]['lb_pvalue']:.4f}")
if lb_test.iloc[0]['lb_pvalue'] > 0.05:
    print("残差是白噪声，模型已充分捕捉数据规律")
else:
    print("警告：残差不是白噪声，模型可能未充分捕捉数据规律")

# 残差图 - 包含模型信息
plt.figure(figsize=(10, 6))
plt.plot(residuals)
plt.title(f'模型残差 - ARIMA{best_param}')
plt.xlabel('年份')
plt.ylabel('残差')
plt.grid(True)
plt.savefig(f'residuals_ARIMA{best_param}.png', dpi=300, bbox_inches='tight')
print(f"\n残差图已保存为 residuals_ARIMA{best_param}.png")

# 残差ACF图 - 包含模型信息
plt.figure(figsize=(10, 6))
plot_acf(residuals, lags=20)
plt.title(f'残差自相关函数 (ACF) - ARIMA{best_param}')
plt.savefig(f'residuals_acf_ARIMA{best_param}.png', dpi=300, bbox_inches='tight')
print(f"\n残差ACF图已保存为 residuals_acf_ARIMA{best_param}.png")
