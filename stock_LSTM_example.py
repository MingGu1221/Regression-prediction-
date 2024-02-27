# example
# Independent Variable : A B C
# dependent varible: D

# -*- coding: utf-8 -*-

# 导入所需的库
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# 设置随机数种子
np.random.seed(1)
tf.random.set_seed(1)

# 数据预处理部分
# 读取比特币价格数据
df = pd.read_csv("Data/BTC-USDT.csv", parse_dates=True, index_col="candle_begin_time")
df = df[['open', 'high', 'low', 'volume', 'close']]  # 更新为新的特征集

# 更新features变量为新的特征集，不包括'close'，因为'close'是我们的目标变量
features = ['open', 'high', 'low', 'volume']
# 数据规范化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

# 使用滑动窗口创建数据集
look_back = 10
X, y = [], []
for i in range(len(scaled_data) - look_back):
    X.append(scaled_data.iloc[i:i + look_back][features].values)
    y.append(scaled_data.iloc[i + look_back]['close'])
X, y = np.array(X), np.array(y)

# 划分训练集和测试集
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 创建 LSTM 模型
def create_lstm_model(input_shape, dropout_rate=0.3, lstm_units=50):  # 可以调整LSTM单元的数量
    input_layer = Input(shape=input_shape)
    lstm_layer1 = LSTM(lstm_units, return_sequences=True)(input_layer)
    dropout_layer1 = Dropout(dropout_rate)(lstm_layer1)
    lstm_layer2 = LSTM(lstm_units, return_sequences=False)(dropout_layer1)
    dropout_layer2 = Dropout(dropout_rate)(lstm_layer2)
    output_layer = Dense(1)(dropout_layer2)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

# 初始化和训练 LSTM 模型
model_lstm = create_lstm_model(input_shape=(look_back, len(features)))
model_lstm.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test), verbose=1)  # 增加训练轮次以提高性能

# 预测
y_pred_lstm = model_lstm.predict(X_test)

# 反规范化预测值和真实值
y_pred_inverse_lstm = scaler.inverse_transform(np.concatenate((np.zeros((len(y_pred_lstm), len(features))), y_pred_lstm), axis=1))[:, -1]
y_test_inverse = scaler.inverse_transform(np.concatenate((np.zeros((len(y_test), len(features))), y_test.reshape(-1,1)), axis=1))[:, -1]

# 计算和打印评估指标
mse_lstm = mean_squared_error(y_test_inverse, y_pred_inverse_lstm)
rmse_lstm = np.sqrt(mse_lstm)
mape_lstm = mean_absolute_error(y_test_inverse, y_pred_inverse_lstm) / np.mean(y_test_inverse) * 100
smape = 100/len(y_test) * np.sum(2 * np.abs(y_pred_inverse_lstm - y_test_inverse) / (np.abs(y_pred_inverse_lstm) + np.abs(y_test_inverse)))
r2_lstm = r2_score(y_test_inverse, y_pred_inverse_lstm)

print(f"Mean Squared Error: {mse_lstm}")
print(f"Root Mean Squared Error: {rmse_lstm}")
print(f"Mean Absolute Percentage Error: {mape_lstm}%")
print(f"SMAPE: {smape}%")
print(f"R^2 Score: {r2_lstm}")

# 绘制测试集的预测结果
import matplotlib.pyplot as plt

# 将日期提取为横坐标
dates = df.index[train_size+look_back:]

# 绘制真实值和预测值
plt.figure(figsize=(15, 6))
plt.plot(dates, y_test_inverse, label='Actual Close Values', color='blue')
plt.plot(dates, y_pred_inverse_lstm, label='Predicted Close Values', color='red')
plt.title('Comparison of Actual and Predicted Close Values')
plt.xlabel('Date')
plt.ylabel('Close Value')
plt.legend()
plt.xticks(rotation=45)  # 旋转日期标签以便更容易阅读
plt.tight_layout()
plt.show()

