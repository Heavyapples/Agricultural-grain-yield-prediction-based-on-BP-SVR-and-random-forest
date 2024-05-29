import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
import pickle

# 读取Excel文件
data = pd.read_excel('data.xlsx')

# 处理缺失值，使用平均值填充
data = data.fillna(data.mean())

# 异常值处理，这里我们使用Z-score方法，Z-score是一个统计学概念，表示数据点与平均值的距离，以标准差为单位。
# Z-score的绝对值大于3通常被认为是异常值
z_scores = np.abs(stats.zscore(data.drop('年份', axis=1)))
filtered_entries = (z_scores < 3).all(axis=1)
data = data[filtered_entries]

# 保存年份列
years = data['年份']

# 对数变换，加1保证所有数据为正
data_log = np.log1p(data.drop(['年份', '粮食产量(万吨)'], axis=1))
data_y_log = np.log1p(data['粮食产量(万吨)'])

# 数据标准化，使得每一列数据都有0均值，1标准差
scaler_X = StandardScaler()
scaler_y = StandardScaler()
data_scaled_X = pd.DataFrame(scaler_X.fit_transform(data.drop(['年份', '粮食产量(万吨)'], axis=1)), columns=data.columns.drop(['年份', '粮食产量(万吨)']))
data_scaled_y = pd.DataFrame(scaler_y.fit_transform(data[['粮食产量(万吨)']]), columns=['粮食产量(万吨)'])

# 将年份列和目标变量列添加回去
data_scaled = pd.concat([years, data_scaled_X, data_scaled_y], axis=1)

# 保存预处理后的数据到Excel文件
data_scaled.to_excel('preprocessed_data.xlsx', index=False)

# 保存scaler到文件
with open('scaler_X.pkl', 'wb') as f:
    pickle.dump(scaler_X, f)
with open('scaler_y.pkl', 'wb') as f:
    pickle.dump(scaler_y, f)

# 打印处理后的数据
print(data_scaled)
