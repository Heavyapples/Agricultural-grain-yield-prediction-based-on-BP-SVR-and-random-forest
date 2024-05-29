import pickle

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.font_manager import FontProperties
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

# 使用黑体字体
font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置全局字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

# 读取预处理后的数据
data = pd.read_excel('preprocessed_data.xlsx')

# 加载scaler
with open('scaler_X.pkl', 'rb') as f:
    scaler_X = pickle.load(f)
with open('scaler_y.pkl', 'rb') as f:
    scaler_y = pickle.load(f)

# 分割特征和目标变量
X = data.drop('粮食产量(万吨)', axis=1)
y = data['粮食产量(万吨)']

# 分割训练集和测试集
X_train = X[X['年份'] <= 2017]
X_test = X[X['年份'] > 2017]
y_train = y[X['年份'] <= 2017]
y_test = y[X['年份'] > 2017]

# BP神经网络
mlp = MLPRegressor(hidden_layer_sizes=(100, 100), learning_rate_init=0.01, alpha=0.01, random_state=2, max_iter=1000).fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)

# 将预测结果从标准化的尺度转换回原始的尺度
y_pred_mlp_original = scaler_y.inverse_transform(y_pred_mlp.reshape(-1, 1))

print('BP神经网络预测结果：', y_pred_mlp_original.flatten())
print('BP神经网络MSE：', mean_squared_error(y_test, y_pred_mlp))
# 绘制预测值与真实值的对比图
plt.figure(figsize=(10, 6))
plt.plot(data['年份'], scaler_y.inverse_transform(mlp.predict(X).reshape(-1, 1)).flatten(), 'r-', label='Predicted')
plt.plot(data['年份'], scaler_y.inverse_transform(y.values.reshape(-1, 1)).flatten(), 'b-', label='Actual')
plt.title('BP神经网络预测结果')
plt.xlabel('年份')
plt.ylabel('粮食产量(万吨)')
plt.legend()
plt.show()

# BP神经网络的2018-2020年预测值与真实值对比图
plt.figure(figsize=(10, 6))
plt.plot(data['年份'][data['年份'].between(2018, 2020)], scaler_y.inverse_transform(mlp.predict(X[X['年份'].between(2018, 2020)]).reshape(-1, 1)).flatten(), 'r-', label='Predicted')
plt.plot(data['年份'][data['年份'].between(2018, 2020)], scaler_y.inverse_transform(y[data['年份'].between(2018, 2020)].values.reshape(-1, 1)).flatten(), 'b-', label='Actual')
plt.title('BP神经网络预测结果(2018-2020)')
plt.xlabel('年份')
plt.ylabel('粮食产量(万吨)')
plt.legend()
plt.show()

# 支持向量机
svr = SVR(C=10.0, gamma='auto', epsilon=0.01).fit(X_train, y_train)
y_pred_svr = svr.predict(X_test)

# 将预测结果从标准化的尺度转换回原始的尺度
y_pred_svr_original = scaler_y.inverse_transform(y_pred_svr.reshape(-1, 1))

print('支持向量机预测结果：', y_pred_svr_original.flatten())
print('支持向量机MSE：', mean_squared_error(y_test, y_pred_svr))

# 绘制预测值与真实值的对比图
plt.figure(figsize=(10, 6))
plt.plot(data['年份'], scaler_y.inverse_transform(svr.predict(X).reshape(-1, 1)).flatten(), 'r-', label='Predicted')
plt.plot(data['年份'], scaler_y.inverse_transform(y.values.reshape(-1, 1)).flatten(), 'b-', label='Actual')
plt.title('支持向量机预测结果')
plt.xlabel('年份')
plt.ylabel('粮食产量(万吨)')
plt.legend()
plt.show()

# 支持向量机的2018-2020年预测值与真实值对比图
plt.figure(figsize=(10, 6))
plt.plot(data['年份'][data['年份'].between(2018, 2020)], scaler_y.inverse_transform(svr.predict(X[X['年份'].between(2018, 2020)]).reshape(-1, 1)).flatten(), 'r-', label='Predicted')
plt.plot(data['年份'][data['年份'].between(2018, 2020)], scaler_y.inverse_transform(y[data['年份'].between(2018, 2020)].values.reshape(-1, 1)).flatten(), 'b-', label='Actual')
plt.title('支持向量机预测结果(2018-2020)')
plt.xlabel('年份')
plt.ylabel('粮食产量(万吨)')
plt.legend()
plt.show()

# 随机森林
rf = RandomForestRegressor(random_state=1).fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# 将预测结果从标准化的尺度转换回原始的尺度
y_pred_rf_original = scaler_y.inverse_transform(y_pred_rf.reshape(-1, 1))

print('随机森林预测结果：', y_pred_rf_original.flatten())
print('随机森林MSE：', mean_squared_error(y_test, y_pred_rf))
# 绘制预测值与真实值的对比图
plt.figure(figsize=(10, 6))
plt.plot(data['年份'], scaler_y.inverse_transform(rf.predict(X).reshape(-1, 1)).flatten(), 'r-', label='Predicted')
plt.plot(data['年份'], scaler_y.inverse_transform(y.values.reshape(-1, 1)).flatten(), 'b-', label='Actual')
plt.title('随机森林预测结果')
plt.xlabel('年份')
plt.ylabel('粮食产量(万吨)')
plt.legend()
plt.show()

# 随机森林的2018-2020年预测值与真实值对比图
plt.figure(figsize=(10, 6))
plt.plot(data['年份'][data['年份'].between(2018, 2020)], scaler_y.inverse_transform(rf.predict(X[X['年份'].between(2018, 2020)]).reshape(-1, 1)).flatten(), 'r-', label='Predicted')
plt.plot(data['年份'][data['年份'].between(2018, 2020)], scaler_y.inverse_transform(y[data['年份'].between(2018, 2020)].values.reshape(-1, 1)).flatten(), 'b-', label='Actual')
plt.title('随机森林预测结果(2018-2020)')
plt.xlabel('年份')
plt.ylabel('粮食产量(万吨)')
plt.legend()
plt.show()
