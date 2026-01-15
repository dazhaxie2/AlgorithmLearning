import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import itertools
import warnings

warnings.filterwarnings("ignore") #忽略输出警告
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 季节性自回归差分移动平均模型训练
# Seasonal AutoRegressive Integrated Moving Average
# %matplotlib inline 仅用于Jupyter，普通Python脚本中不需要此命令

import itertools  # 添加缺失的导入


# 1.数据准备


# 1)读取数据
df=pd.read_csv("MER_T02_06.csv",index_col="YYYYMM") #指定YYYYMM列作为索引列
# print(df)


# 2)数据预处理

# 删除Description列中值在key_list中的行
# df=df[~df['Description'].isin(key_list)]

key_list=['Coal Consumed by the Electric Power Sector']
df=df[df['Description'].isin(key_list)]
# print(df)

# 注意：数据中有些值为季度格式（如194913表示1949Q1），需要过滤
df.index = df.index.astype(str)
# 只保留月份数据（月份<=12），过滤掉季度数据
df = df[df.index.str[-2:].astype(int) <= 12]

# 将Value列转换为数值类型
df['Value'] = pd.to_numeric(df['Value'], errors='coerce')

# 删除转换失败的行（如果有）
df = df.dropna(subset = ['Value'])

# 将索引YYYYMM转换为datetime格式，便于图表显示
df.index = pd.to_datetime(df.index, format = '%Y%m')

# 3)数据可视化
# 电力行业碳排放
CCE = df['Value']
CCE.head()

# 折线图
fig, ax = plt.subplots(figsize = (15,15))
CCE.plot(ax = ax,fontsize = 15)
ax.set_title('电力行业碳排放',fontsize = 25)
ax.set_xlabel('时间(月)',fontsize = 25)
ax.set_ylabel('碳排放量(万亿英热≈2.93亿千瓦时)',fontsize = 25)
ax.legend(loc = "best",fontsize = 15)
ax.grid()
plt.savefig('电力行业碳排放.png', dpi = 300, bbox_inches = 'tight')
print("图像已保存为: 电力行业碳排放.png")
plt.close()

# 4)分解时序
# STL(Seasonal and Trend decomposition using Loess)
import statsmodels.api as sm
decomposition = sm.tsa.STL(CCE).fit()
fig = decomposition.plot()
fig.set_size_inches(15, 10)
plt.savefig('STL时序分解.png', dpi = 300, bbox_inches = 'tight')
print("图像已保存为: STL时序分解.png")
plt.close()
# 趋势效应
trend = decomposition.trend
# 季节效应
seasonal = decomposition.seasonal
# 随机效应
residual = decomposition.resid


# 2.平稳性检验


# 自定义函数用于ADF检查平稳性
from statsmodels.tsa.stattools import adfuller as ADF
def test_stationarity(timeseries,alpha): # alpha为检验选取的显著性水平
    adf = ADF(timeseries)
    p = adf[1] #p值
    critical_value = adf[4]['5%'] # 在95%置信区间下的临界的ADF检验值
    test_statistic = adf[0] # ADF统计量
    if p < alpha and test_statistic < critical_value:
        print(f"ADF平稳性检验结果：数据平稳")
        print(f"  - 显著性水平：{alpha}")
        print(f"  - p值：{p:.6e} < {alpha} ✓")
        print(f"  - ADF统计量：{test_statistic:.6f} < 临界值 {critical_value:.6f} ✓")
        return True
    else:
        print(f"ADF平稳性检验结果：数据不平稳")
        print(f"  - 显著性水平：{alpha}")
        print(f"  - p值：{p:.6e} {'< ' + str(alpha) + ' ✓' if p < alpha else '>= ' + str(alpha) + ' ✗'}")
        print(f"  - ADF统计量：{test_statistic:.6f} {'< 临界值 ' + f'{critical_value:.6f} ✓' if test_statistic < critical_value else '>= 临界值 ' + f'{critical_value:.6f} ✗'}")
        return False

#原始数据平稳性检验
test_stationarity(CCE, 0.05)

# 将数据化为平稳数据
# 一阶差分
CCE_diff1 = CCE.diff(1)
# 十二步差分
CCE_seasonal = CCE_diff1.diff(12) # 非平稳序列经过d阶常差分和D阶季节差分后，序列变得更加平稳
print(CCE_seasonal)
# 十二步季节差分平稳性检验结果
# test_stationarity(CCE_seasonal.dropna(), 0.05) #使用dropna()去除NaN值


# 3.白噪声检验


# LB白噪声检验
from statsmodels.stats.diagnostic import acorr_ljungbox
def test_white_noise(data, alpha):
    # 去除NaN值
    clean_data = data.dropna()
    if len(clean_data) == 0:
        print(f"白噪声检验结果：数据全为NaN，无法进行检验")
        print(f"  - 显著性水平：{alpha}")
        return False
    
    result = acorr_ljungbox(clean_data, lags=[1], return_df=True)
    p = result["lb_pvalue"].iloc[0]
    
    # 检查p值是否为NaN
    if pd.isna(p):
        print(f"白噪声检验结果：p值为NaN，无法进行检验")
        print(f"  - 显著性水平：{alpha}")
        return False
        
    if p >= alpha:
        print(f"白噪声检验结果：数据为白噪声")
        print(f"  - 显著性水平：{alpha}")
        print(f"  - p值：{p:.6e} >= {alpha} ✓")
        return True
    else:
        print(f"白噪声检验结果：数据不是白噪声")
        print(f"  - 显著性水平：{alpha}")
        print(f"  - p值：{p:.6e} {'>= ' + str(alpha) + ' ✓' if p >= alpha else '< ' + str(alpha) + ' ✗'}")
        return False

# test_white_noise(CCE_seasonal.dropna(), 0.05)


# 4.模型定阶


# 搜索法定阶
def SARIMA_search(data):
    p = q = range(0, 3)
    s = [12] # 周期为12
    d = [1] # 做了一次季节性差分
    PDQs = list(itertools.product(p, d, q, s)) # itertools.product()用于将输入的可迭代对象作为参数进行笛卡尔积运算
    pdq = list(itertools.product(p, d, q)) # list是python中的一种数据结构，序列中的每个元素都分配一个数字定位位置
    
    # 初始化最佳模型参数
    best_aic = float('inf')
    best_param = None
    best_seasonal_param = None
    best_result = None
    
    print("开始模型参数搜索...")
    print("格式: ARIMA(p,d,q)x(P,D,Q,s) - AIC值")
    print("="*50)
    
    for param in pdq:
        for seasonal_param in PDQs:
            # 建立模型
            try:
                mod = sm.tsa.SARIMAX(data, order=param, seasonal_order=seasonal_param,
                enforce_stationarity=False, enforce_invertibility=False)
                # 实现数据在模型中训练
                result = mod.fit(maxiter=50)
                # 输出模型的AIC值，:.6f表示保留6位小数
                current_aic = result.aic
                print(f"ARIMA{param}x{seasonal_param} - AIC:{current_aic:.6f}")
                
                # 如果当前模型的AIC更小，则更新最佳模型
                if current_aic < best_aic:
                    best_aic = current_aic
                    best_param = param
                    best_seasonal_param = seasonal_param
                    best_result = result
                    print(f"  >>> 新的最佳模型! AIC: {best_aic:.6f}")
                    
            except Exception as e:
                print(f"ARIMA{param}x{seasonal_param} - 拟合失败: {str(e)[:50]}...")
                continue
    
    print("\n" + "="*50)
    if best_result is not None:
        print(f"搜索完成! 最佳模型参数:")
        print(f"ARIMA{best_param}x{best_seasonal_param}")
        print(f"最低AIC值: {best_aic:.6f}")
        return best_result  # 返回最佳模型
    else:
        print("所有模型拟合均失败")
        return None
    
# SARIMA_search(CCE_seasonal.dropna())


# 5.模型的建立与检验


# 建立并训练SARIMA（0,1,2）x（0,1,2,12）12模型
model = sm.tsa.SARIMAX(CCE_seasonal, order=(1,1,2), seasonal_order=(1,1,2,12))
SARIMA_m = model.fit()
print(SARIMA_m.summary())

# 模型检验
test_white_noise(SARIMA_m.resid, 0.05) # SARIMA_m.resid提取模型残差，进行白噪声检验
fig = SARIMA_m.plot_diagnostics(figsize = (15, 12)) # plot_diagnostics对象允许我们快速生成模型诊断并调查任何异常行为
plt.savefig('SARIMA模型诊断.png', dpi = 300, bbox_inches = 'tight')
print("图像已保存为: SARIMA模型诊断.png")
plt.close()

# 模型预测
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae

#获取预测结果，自定义预测误差
def PredictionAnalysis(data,model,start_date,dynamic = False):
    pred = model.get_prediction(start = start_date, dynamic = dynamic)
    pci = pred.conf_int() # 获取预测值的置信区间
    pm = pred.predicted_mean # 获取预测值
    truth = data[start_date:] # 获取真实值
    pc = pd.concat([truth, pm, pci], axis = 1) # 按列拼接
    pc.columns = ['truth', 'predicted', 'lower_bound', 'upper_bound']
    print(f"1、MSE:{mse(truth, pm):.6f}")
    print(f"2、RMSE:{np.sqrt(mse(truth, pm)):.6f}")
    print(f"3、MAE:{mae(truth, pm):.6f}")
    return pc

# 绘制预测结果
def PredictionPlot(pc):
    fig, ax = plt.subplots(figsize = (10, 8))
    ax.fill_between(pc.index, pc['lower_bound'], pc['upper_bound'], color = 'grey',
    alpha = 0.15, label = 'confidence interval') # 画出执行区间
    ax.plot(pc['truth'], label = 'base data')
    ax.plot(pc['predicted'], label = 'predicted curve')
    ax.legend()
    return fig

# 静态预测：进行一系列的一步预测，即它必须用真实值来进行预测
pred=PredictionAnalysis(CCE_seasonal,SARIMA_m,'2018-09-01')
fig_static = PredictionPlot(pred)
fig_static.savefig('SARIMA模型静态预测.png', dpi = 300, bbox_inches = 'tight')
print("图像已保存为: SARIMA模型静态预测.png")
plt.close(fig_static)

# 动态预测：进行多步预测，除了第一个预测值是用实际值预测外，其后各预测值都是采用递推预测
pred=PredictionAnalysis(CCE_seasonal,SARIMA_m,'2018-09-01',dynamic=True)
fig_dynamic = PredictionPlot(pred)
fig_dynamic.savefig('SARIMA模型动态预测.png', dpi = 300, bbox_inches = 'tight')
print("图像已保存为: SARIMA模型动态预测.png")
plt.close(fig_dynamic)

