import os
from sympy.core.evalf import scaled_zero
import pandas as pd # 数据处理
import numpy as np # 数值计算
import matplotlib.pyplot as plt # 数据可视化
from sklearn.preprocessing import MinMaxScaler # 数据归一化
from tcn.tcn import TCN # TCN模型
from tensorflow import keras # Keras
import joblib # 用于保存scaler

window_size = 30    # 窗口大小（增大以捕捉更长的时序模式）
batch_size = 32     # 训练批次大小
epochs = 300        # 训练轮数
filter_nums = 64    # filter数量（增大以提升特征提取能力）
kernel_size = 3     # 卷积核大小

def get_dataset(window_size=60, csv_path='./上证指数历史数据.csv'):
    """
    加载并处理股票数据集（适配3159条中文列名带格式数据）
    :param window_size: 时间窗口大小，即用前window_size个开盘价预测下一个开盘价
    :param csv_path: 新数据集的文件路径（请替换为实际路径）
    :return: 训练/测试特征、标签及归一化器
    """
    # 1. 读取CSV文件
    df = pd.read_csv(csv_path)
    
    # 2. 数据预处理：处理带格式的"开盘"列（核心修改：解决千位分隔符问题）
    # 替换千位分隔符","，转换为浮点型数值（避免无法进行归一化计算）
    df['开盘'] = df['开盘'].astype(str).str.replace(',', '').astype(float)
    
    # 可选：若后续需要使用其他列，可同步处理（交易量、涨跌幅）
    # df['交易量'] = df['交易量'].str.replace('B', '').astype(float) * 1e9  # B指代十亿，转换为具体数值
    # df['涨跌幅'] = df['涨跌幅'].str.replace('%', '').astype(float) / 100  # 百分号转换为小数
    
    # 3. 检查并处理缺失值
    if df['开盘'].isnull().any():
        df['开盘'] = df['开盘'].fillna(method='ffill')  # 前向填充缺失值
    
    # 4. 数据归一化
    scaler = MinMaxScaler()
    open_arr = scaler.fit_transform(df['开盘'].values.reshape(-1, 1)).reshape(-1)
    
    # 5. 构建特征矩阵X和标签数组label（保持原逻辑，无修改）
    data_length = len(open_arr) - window_size
    X = np.zeros(shape = (data_length, window_size))
    label = np.zeros(shape = (data_length,))
    
    for i in range(data_length):
        X[i, :] = open_arr[i:i+window_size]
        label[i] = open_arr[i+window_size]
    
    # 6. 划分训练集和测试集（80%训练/20%测试）
    train_split = int(0.8 * data_length)
    train_X = X[:train_split, :]
    train_label = label[:train_split]
    test_X = X[train_split:, :]
    test_label = label[train_split:]
    
    return train_X, train_label, test_X, test_label, scaler

def RMSE(pred, true):
    """
    计算均方根误差（Root Mean Square Error, RMSE）
    用于评估预测值与真实值之间的误差大小，RMSE越小说明预测效果越好
    :param pred: 模型预测结果数组（一维numpy数组或列表）
    :param true: 真实标签数组（一维numpy数组或列表，与pred长度一致）
    :return: 计算得到的均方根误差（标量值）
    """
    # 步骤1：计算预测值与真实值的差值（对应元素相减）
    # 步骤2：对差值进行平方运算，消除正负误差抵消的问题
    # 步骤3：计算平方误差的平均值（np.mean）
    # 步骤4：对平均值开平方根（np.sqrt），还原误差的量纲，得到RMSE
    return np.sqrt(np.mean(np.square(pred - true)))

def plot_with_rolling(train_pred, train_true, test_pred, test_true, rolling_preds, scaler):
    """
    绘制训练集、测试集和滚动预测的对比图并保存
    :param train_pred: 训练集预测结果
    :param train_true: 训练集真实值
    :param test_pred: 测试集预测结果
    :param test_true: 测试集真实值
    :param rolling_preds: 滚动预测结果
    :param scaler: 归一化器
    """
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(111)
    
    # 训练集范围
    train_len = len(train_true)
    test_len = len(test_true)
    
    # 绘制训练集真实值和预测值
    ax.plot(range(train_len), train_true, label='Train True', color='#2E86AB', alpha=0.7)
    ax.plot(range(train_len), train_pred, label='Train Predicted', color='#A23B72', linestyle='--', alpha=0.7)
    
    # 绘制测试集真实值和预测值
    ax.plot(range(train_len, train_len + test_len), test_true, label='Test True', color='#F18F01', alpha=0.7)
    ax.plot(range(train_len, train_len + test_len), test_pred, label='Test Predicted', color='#C73E1D', linestyle='--', alpha=0.7)
    
    # 绘制滚动预测（未来预测）
    future_start = train_len + test_len
    ax.plot(range(future_start, future_start + len(rolling_preds)), rolling_preds, 
            label='Rolling Predictions', color='#8E44AD', linestyle='-.', linewidth=2, marker='o', markersize=6)
    
    # 添加分割线区分训练集和测试集
    ax.axvline(x=train_len, color='gray', linestyle=':', linewidth=2, label='Train/Test Split')
    
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('Opening Price', fontsize=12)
    ax.set_title(f'TCN Model Predictions with Rolling Forecast (window={window_size}, filters={filter_nums})', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('tcn_prediction_results.png', dpi=300, bbox_inches='tight')
    print("\n预测图表已保存为: tcn_prediction_results.png")
    plt.show()

def build_model_fn():
    """
    构建TCN模型
    """
    # 构建Sequential顺序模型（按层堆叠的模型结构）
    model = keras.models.Sequential([
        # 输入层：指定输入数据的形状为(window_size, 1)，适配TCN层的输入要求
        # window_size为时间窗口长度，1为单特征（仅开盘价）
        keras.layers.Input(shape=(window_size, 1)),
        # TCN层（时间卷积网络）：用于处理时序数据，提取时序特征
        TCN(nb_filters=filter_nums,  # 卷积滤波器数量，决定特征提取的维度
            kernel_size=kernel_size, # 卷积核大小，决定单次卷积的窗口长度
            dilations=[1, 2, 4, 8],  # 空洞卷积膨胀系数，扩大感受野且不增加参数量
        ),
        # 全连接输出层：将TCN层提取的特征映射为最终预测结果（单值输出）
        keras.layers.Dense(units=1, activation='linear')  # 回归任务使用linear激活函数
    ])
    
    return model

def train_model():
    """
    训练模型并保存
    """
    # 检查是否存在已训练的模型
    if os.path.exists('tcn_trained_model.weights.h5') and os.path.exists('tcn_model_config.json'):
        print("发现已训练的模型，跳过训练过程...")
        return
    
    print("开始训练模型...")
    # 调用自定义的get_dataset函数，加载并返回训练/测试数据及归一化器
    train_X, train_label, test_X, test_label, scaler = get_dataset()
    
    # 构建模型
    model = build_model_fn()
    
    # 打印模型结构摘要，展示各层名称、输出形状、参数数量等信息，便于检查模型结构
    model.summary()
    
    # 模型编译：配置训练所需的优化器、损失函数和评估指标
    model.compile(optimizer='adam',  # 优化器：选用adam，自适应学习率，训练效果稳定
                  loss='mse',        # 损失函数：选用均方误差（MSE），适用于回归预测任务
                  metrics=['mae'])   # 评估指标：选用平均绝对误差（MAE），辅助评估模型性能
    
    # 模型训练：使用训练集数据进行拟合，同时划分20%数据作为验证集监控训练效果
    model.fit(train_X, train_label,          # 训练特征和训练标签
              validation_split=0.2,          # 从训练集中划分20%作为验证集，用于监控过拟合
              epochs=epochs)                 # 训练轮数：迭代整个训练集的次数，由全局超参数epochs指定
    
    # 模型评估：在测试集上评估模型的性能，输出损失值（MSE）和评估指标（MAE）
    model.evaluate(test_X, test_label)
    
    # 保存模型权重和架构
    model.save_weights('tcn_trained_model.weights.h5')
    model_config = model.get_config()
    import json
    with open('tcn_model_config.json', 'w') as f:
        json.dump(model_config, f)
    print("模型权重已保存为: tcn_trained_model.weights.h5")
    print("模型配置已保存为: tcn_model_config.json")
    
    # 保存scaler
    joblib.dump(scaler, 'scaler.pkl')
    print("归一化器已保存为: scaler.pkl")


def rolling_predict(model, data, steps, scaler, window_size):
    """
    滚动预测函数
    :param model: 已训练的模型
    :param data: 输入数据（已归一化）
    :param steps: 预测步数
    :param scaler: 归一化器
    :param window_size: 窗口大小
    :return: 滚动预测结果
    """
    # 从数据末尾获取最后一个窗口的数据作为初始输入
    current_window = data[-window_size:].copy()
    predictions = []
    
    for i in range(steps):
        # 将当前窗口重塑为模型输入格式
        input_data = current_window.reshape(1, window_size, 1)
        # 预测下一个值
        next_pred = model.predict(input_data, verbose=0)
        # 添加到预测结果列表
        predictions.append(next_pred[0, 0])
        # 更新窗口：移除第一个值，添加新的预测值
        current_window = np.append(current_window[1:], next_pred[0, 0])
    
    return np.array(predictions)


def predict_with_saved_model():
    """
    使用已保存的模型进行预测
    """
    # 检查模型文件是否存在
    if not os.path.exists('tcn_trained_model.weights.h5') or not os.path.exists('tcn_model_config.json'):
        print("未找到已训练的模型，请先运行训练过程")
        return
    
    print("加载已训练的模型...")
    # 重建模型
    import json
    with open('tcn_model_config.json', 'r') as f:
        model_config = json.load(f)
    model = keras.models.Sequential.from_config(model_config, custom_objects={'TCN': TCN})
    model.build((None, window_size, 1))  # 重建模型结构
    model.load_weights('tcn_trained_model.weights.h5')
    
    # 加载scaler
    scaler = joblib.load('scaler.pkl')
    
    # 加载数据
    train_X, train_label, test_X, test_label, original_scaler = get_dataset()
    
    # 模型预测：分别对训练集和测试集进行预测
    train_prediction = model.predict(train_X)
    test_prediction = model.predict(test_X)
    
    # 反归一化：训练集
    scaled_train_prediction = scaler.inverse_transform(train_prediction.reshape(-1, 1)).reshape(-1)
    scaled_train_label = scaler.inverse_transform(train_label.reshape(-1, 1)).reshape(-1)
    
    # 反归一化：测试集
    scaled_test_prediction = scaler.inverse_transform(test_prediction.reshape(-1, 1)).reshape(-1)
    scaled_test_label = scaler.inverse_transform(test_label.reshape(-1, 1)).reshape(-1)
    
    # 执行滚动预测（预测未来10个时间步）
    # 获取完整数据进行滚动预测
    df = pd.read_csv('./上证指数历史数据.csv')
    df['开盘'] = df['开盘'].astype(str).str.replace(',', '').astype(float)
    if df['开盘'].isnull().any():
        df['开盘'] = df['开盘'].fillna(method='ffill')
    full_data = scaler.transform(df['开盘'].values.reshape(-1, 1)).reshape(-1)
    
    # 执行滚动预测
    future_steps = 10
    rolling_predictions = rolling_predict(model, full_data, future_steps, scaler, window_size)
    
    # 计算并打印RMSE值
    print(f'Train RMSE: {RMSE(scaled_train_prediction, scaled_train_label):.2f}')
    print(f'Test RMSE: {RMSE(scaled_test_prediction, scaled_test_label):.2f}')
    
    # 调用plot函数，绘制训练集和测试集的对比图，并包含滚动预测
    plot_with_rolling(scaled_train_prediction, scaled_train_label, scaled_test_prediction, scaled_test_label, rolling_predictions, scaler)


def build_model_full():
    """
    核心函数：完成数据加载、TCN模型构建、训练、评估、预测及结果可视化
    流程：加载预处理数据 -> 构建Sequential+TCN模型 -> 编译模型 -> 训练模型
          -> 测试集评估 -> 模型预测 -> 反归一化还原真实值 -> 计算RMSE -> 绘图对比
    """
    # 训练模型
    train_model()
    
    # 使用已保存的模型进行预测
    predict_with_saved_model()


if __name__ == '__main__':
    """
    程序入口：当该脚本被直接运行时，执行以下代码（被导入时不执行）
    作用：可以选择训练模型或使用已保存的模型进行预测
    """
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == 'train':
            print("仅执行模型训练...")
            train_model()
        elif sys.argv[1] == 'predict':
            print("仅执行模型预测...")
            predict_with_saved_model()
        else:
            print(f"未知参数: {sys.argv[1]}. 使用 'train' 或 'predict'")
    else:
        print("执行完整的训练和预测流程...")
        build_model_full()
