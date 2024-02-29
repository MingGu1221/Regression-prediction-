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
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata


np.random.seed(1)
tf.random.set_seed(1)

df = pd.read_excel("Data/one_GNP_timestep_voltage.xlsx")
print(df.columns)
df = df[['voltage', 'timestep', 'strain']]  # 更新为新的特征集

# 原图复现
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# 从DataFrame中提取数据
x = df['voltage']
y = df['timestep']
z = df['strain']
# 创建网格数据
xi = np.linspace(x.min(), x.max(), 100)
yi = np.linspace(y.min(), y.max(), 100)
xi, yi = np.meshgrid(xi, yi)
# 插值
zi = griddata((x, y), z, (xi, yi), method='cubic')
surf = ax.plot_surface(xi, yi, zi, cmap='viridis', edgecolor='none')
ax.set_xlabel('Voltage')
ax.set_ylabel('Time Step')
ax.set_zlabel('Strain')
# 添加颜色条
fig.colorbar(surf)
plt.show()
# represent good

# 自变量
x_flat = xi.flatten()
y_flat = yi.flatten()
z_flat = zi.flatten()
data = pd.DataFrame({
    'x_flat': x_flat,
    'y_flat': y_flat,
    'z_flat': z_flat
})# 更新为新的特征集

# # 指定文件路径和名称
# file_path = "Data/exported_data.xlsx"
# # 导出到Excel
# data.to_excel(file_path, index=False)
df = pd.read_excel("Data/exported_data.xlsx")
print(df.columns)
df = df[['voltage', 'timestep', 'strain']]  # 更新为新的特征集
# 更新features变量为新的特征集，不包括'z'，因为'z'是我们的目标变量
features = ['voltage', 'timestep']
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

look_back = 10
X, y = [], []
for i in range(len(scaled_data) - look_back):
    X.append(scaled_data.iloc[i:i + look_back][features].values)
    y.append(scaled_data.iloc[i + look_back]['strain'])
X, y = np.array(X), np.array(y)

train_size = int(len(X) * 0.7)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 创建 LSTM 模型
def create_lstm_model(input_shape, dropout_rate=0.01, lstm_units=1000):  # 可以调整LSTM单元的数量
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
model_lstm.fit(X_train, y_train, epochs=15, batch_size=30, validation_data=(X_test, y_test), verbose=2)  # 增加训练轮次以提高性能

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

import matplotlib.pyplot as plt

voltage_test_indices = range(train_size + look_back, len(df))  # 这将给出测试集对应的原始数据集中的索引
voltage_test = df.iloc[voltage_test_indices]['voltage'].values  # 提取相应的电压值

timestep_test_indices = range(train_size + look_back, len(df))
timestep_test = df.iloc[timestep_test_indices]['timestep'].values  # 提取相应的电压值

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# plt.plot(voltage_test, timestep_test, y_test_inverse, label='Actual Strain', color='blue', cmap='viridis', edgecolor='none')
# plt.plot(voltage_test, timestep_test, y_pred_inverse_lstm, label='Predicted Strain', color='red', cmap='viridis', edgecolor='none')
ax.scatter(voltage_test, timestep_test, y_test_inverse, label='Actual Strain', color='blue')
ax.scatter(voltage_test, timestep_test, y_pred_inverse_lstm, label='Predicted Strain', color='red')
ax.set_xlabel('Voltage')
ax.set_ylabel('Timestep')
ax.set_zlabel('Strain Value')
ax.legend()
plt.show()






