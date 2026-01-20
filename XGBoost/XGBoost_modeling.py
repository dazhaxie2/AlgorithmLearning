# 导入必要的库
import warnings
# 1. 导入numpy库并简写为np：Python数值计算核心库，用于高效处理数组、矩阵运算、数值计算
# 场景中：处理设备传感器数据（如温度、振动值数组）、特征矩阵运算、缺失值填充等
import numpy as np

# 2. 导入pandas库并简写为pd：Python数据处理与分析核心库，用于结构化数据的读取、清洗、特征工程
# 场景中：读取设备运行日志CSV/Excel、清洗故障标签数据、构造时序特征（滑动均值、方差等）
import pandas as pd

# 3. 导入matplotlib.pyplot并简写为plt：Python基础数据可视化库，用于绘制各类统计图表
# 场景中：绘制设备故障分布直方图、传感器数据趋势图、模型准确率变化曲线等
import matplotlib.pyplot as plt

# 4. 导入XGBoost库并简写为xgb：高性能集成学习库，主打梯度提升树算法，用于分类/回归任务
# 场景中：设备故障诊断（二分类/多分类）、设备剩余寿命回归预测，工业落地首选算法之一
import xgboost as xgb

# 5. 从sklearn数据集模块导入鸢尾花数据集：sklearn是Python机器学习工具库，load_iris是经典测试数据集
# 场景中：快速获取示例数据，用于验证故障诊断算法的可行性（原型验证），无需手动准备数据
from sklearn.datasets import load_iris

# 6. 从sklearn模型选择模块导入数据分割函数：用于将数据集划分为训练集和测试集
# 场景中：将设备历史数据分为训练集（训练故障模型）和测试集（评估模型泛化能力，避免过拟合）
from sklearn.model_selection import train_test_split

# 7. 从sklearn评估模块导入3个分类任务评估工具：
# accuracy_score：计算分类准确率（整体故障判断正确率）
# confusion_matrix：生成混淆矩阵（分析故障类型误判情况，如正常判为故障、故障判为正常）
# classification_report：生成详细分类报告（包含精确率、召回率、F1值，全面评估故障分类效果）
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 8. 导入seaborn库并简写为sns：基于matplotlib的高级可视化库，图表更美观、功能更丰富
# 场景中：美化混淆矩阵热力图、绘制设备特征相关性热力图、故障类型分布饼图等
import seaborn as sns

# 9. 从XGBoost库导入两个可视化工具：
# plot_tree：绘制XGBoost决策树结构（理解模型故障判断规则，提升可解释性）
# plot_importance：绘制特征重要性图（明确哪些传感器特征对故障诊断影响最大，如振动值、温度等）
from xgboost import plot_tree, plot_importance

# --- 配置项 ---
warnings.filterwarnings("ignore") 
# 设置中文字体，如果报错请尝试更换为 'Microsoft YaHei' 或 'SimHei'
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False 


# 1. 数据准备


iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# 数据集基本信息查看
print(f"数据集基本信息：\n{X.shape}, {y.shape}\n{X[:5]}, {y[:5]}\n{feature_names}, {target_names}")

# 将数据转换为DataFrame以便查看
iris_df = pd.DataFrame(X, columns=feature_names)
iris_df['target'] = y
iris_df['target_name'] = iris_df['target'].map({
    0: 'target_name[0]',
    1: 'target_name[1]',
    2: 'target_name[2]'
})

# 显示数据集前5行
print("\n数据集前5行：")
print(iris_df.head())

# 数据集基本统计信息
print("\n数据集基本统计信息：")
print(iris_df.describe())

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = 42, stratify = y
)
print(f"训练集形状：{X_train.shape}, {y_train.shape}")
print(f"测试集形状：{X_test.shape}, {y_test.shape}")


# 2. 模型训练


# XGBoost模型初始化（基本参数设置）
params = {
    'objective': 'multi:softmax', # 多分类问题
    'num_class': 3, # 类别数量
    'max_depth': 3, # 树的最大深度
    'learning_rate': 0.1, # 学习率
    'eval_metric': 'mlogloss', # 评估指标
    'seed': 42 # 随机种子
}

# 将数据转换为DMatricx格式
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names = feature_names)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names = feature_names)

# 训练模型
num_round = 100 # 迭代次数
model = xgb.train(
    params, # 参数
    dtrain, # 训练数据
    num_round, # 迭代次数
    evals=[(dtrain, 'train'), (dtest, 'test')], # 评估数据集
    early_stopping_rounds=10, # 早停轮数
    verbose_eval = 20, # 显示迭代信息
)

# 模型预测
y_pred = model.predict(dtest)
print(f"\n前10个预测结果：{y_pred[:10]}") # 显示前10个预测结果
print(f"实际标签前10个：{y_test[:10]}")


# 3. 模型评估


# 准确率评估
accuracy = accuracy_score(y_test, y_pred)
print(f"\n模型准确率：{accuracy:.4f}")

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
print(f"\n混淆矩阵：\n{cm}")

# 分类报告
report = classification_report(y_test, y_pred, target_names = target_names)
print(f"\n分类报告：\n{report}")


# 4. 特征重要性分析

# 特征重要性计算
importance = model.get_score(importance_type='weight')
print(f"特征重要性：")
for feature, score in importance.items():
    print(f"{feature}: {score}")

# 特征重要性可视化
plt.figure(figsize=(10, 6))
plot_importance(model, importance_type='weight')
plt.title("特征重要性")
plt.savefig("iris_feature_importance.png")


# 5. 可视化结果


# 混淆矩阵可视化
plt.figure(figsize=(8, 6)) # 设置图形大小
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', # 标注数据
            xticklabels=target_names, # x轴标签
            yticklabels=target_names) # y轴标签
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('混淆矩阵')
plt.savefig('iris_confusion_matrix.png')

# 数据分布可视化
plt.figure(figsize=(12, 10)) # 设置图形大小
for i, feature in enumerate(feature_names): 
    plt.subplot(2, 2, i+1) # 设置子图位置
    for target in range(3):
        plt.hist(iris_df[iris_df['target'] == target][feature], # 绘制直方图
             alpha=0.5, # 透明度
             label=target_names[target], # 标签
             bins=20) # 直方图柱数
    plt.xlabel(feature) # x轴标签
    plt.ylabel('频数') # y轴标签
    plt.legend() # 显示图例
plt.tight_layout() # 调整子图间距

plt.savefig('iris_feature_distribution.png') # 保存图片
print("图片已保存到 iris_feature_distribution.png")
