import inline
import pandas
import pandas as pd
import numpy as np
import openpyxl
import re
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import sys
import seaborn as sns
import warnings
import math
from sklearn.metrics import *
from sklearn.linear_model import *
from sklearn.neighbors import *
from sklearn.svm import *
from sklearn.neural_network import *
from sklearn.tree import *
from sklearn.ensemble import *
from xgboost import *
import lightgbm as lgb

from sklearn.preprocessing import *
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.model_selection import *
# 数据处理

df = pd.read_csv('../code/car.csv')
df = pd.DataFrame(df)
df.drop(columns=['车名'],inplace=True)
# print(df.info())

df = df.fillna(method='ffill',axis=0)
# print(df.info())
df=df[ ~ df['排量'].str.contains('km')]
df=df[ ~ df['发布时间'].str.contains('未上牌')]
df=df[ ~ df['排量'].str.contains(', ')]
df=df[ ~ df['发动机'].str.contains('纯电动')]
df=df[ ~ df['发动机'].str.contains('增程式')]
df=df[ ~ df['车型'].str.contains('跑车')]
df=df[ ~ df['车型'].str.contains('皮卡')]
df=df[ ~ df['车型'].str.contains('微')]
df=df[ ~ df['车型'].str.contains('轻客')]
df=df[ ~ df['发动机'].str.contains('-')]
df=df[ ~ df['燃油类型'].str.contains('0')]
df['发动机']= df['发动机'].astype(float)
df['排量']= df['排量'].astype(float)
# df['燃油类型']= df['燃油类型'].astype(float)
df['发布时间'] = df['发布时间'].str.replace('年',' ')
df['价格'] = df['价格'].str.replace(', , ',' ')
df['价格']= df['价格'].astype(float)
df=df[df['价格'] < 400]
df=df[df['转手次数'] < 10]
df['发布时间'] = df['发布时间'].replace('年',' ').map(lambda x:str(x.split(' ')[0]))
df['颜色'] = df['颜色'].str.replace('\/\D','',regex=True)
df['颜色'] = df['颜色'].str.replace('色','')
df['车型'] = df['车型'].str.replace('型车','')
df['车型'] = df['车型'].str.replace('型SUV','')
df['环保标准'] = df['环保标准'].str.replace('(国V)','')
df['环保标准'] = df['环保标准'].str.replace('+OBD','')
df['环保标准'] = df['环保标准'].str.replace('(国IV)','')
df['环保标准'] = df['环保标准'].str.replace('\/国V','')
df['环保标准'] = df['环保标准'].str.replace('(国VI)','')
# print(df.select_dtypes(include=['object']).describe())
# print(df.select_dtypes(include=['float']).describe())

warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
pd.set_option('display.max_rows', 100,'display.max_columns', 1000,"display.max_colwidth",1000,'display.width',1000)

#
# # # 连续变量箱线图
# fig = plt.figure(figsize=(4, 4))
# plt.rcParams["font.size"]=10#设置字体大小
# sns.boxplot(y=df['价格'])#绘制箱线图')
# plt.show()
# fig = plt.figure(figsize=(4, 4))
# plt.rcParams["font.size"]=10#设置字体大小
# sns.boxplot(y=df['里程'])#绘制箱线图')
# plt.show()
# fig = plt.figure(figsize=(4, 4))
# plt.rcParams["font.size"]=10#设置字体大小
# sns.boxplot(y=df['发布时间'])#绘制箱线图')
# plt.show()
# fig = plt.figure(figsize=(4, 4))
# plt.rcParams["font.size"]=10#设置字体大小
# sns.boxplot(y=df['转手次数'])#绘制箱线图')
# plt.show()
# fig = plt.figure(figsize=(4, 4))
# plt.rcParams["font.size"]=10#设置字体大小
# sns.boxplot(y=df['排量'])#绘制箱线图')
# plt.show()
# fig = plt.figure(figsize=(4, 4))
# plt.rcParams["font.size"]=10#设置字体大小
# sns.boxplot(y=df['发动机'])#绘制箱线图')
# plt.show()
# # 数据可视化分析
#
huanbao_counts = df.groupby('发布时间').size()
huanbao_counts.plot.bar()
plt.show()
time_counts = df.groupby('颜色').size()
time_counts.plot.bar()
plt.show()
drive_counts = df.groupby('驱动类型').size()
drive_counts.plot.pie(autopct= '%1.1f%%')
plt.show()
color_counts = df.groupby('环保标准').size()
color_counts.plot.pie(autopct= '%1.1f%%')
plt.show()
color_counts = df.groupby('车型').size()
color_counts.plot.pie()
plt.show()
# 机器学习建模


#特征工程-onehot编码
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

# #连续变量的相关系数热力图
# df1 = pandas.DataFrame(df[['价格','里程','发布时间','转手次数','排量','发动机']])
# plt.figure(figsize=(8,8))
# sns.set_style('whitegrid',{'font.sans-serif':['simhei','Arial']})
# plt.rcParams['axes.unicode_minus']=False
# sns.heatmap(df1.corr(),annot=True,vmax=1, square=True)
# plt.show()
#
# # 价格与排量的散点图
# x = df['排量']
# y = df['价格']
# plt.scatter(x, y, label='Data')
# slope, intercept = np.polyfit(x, y, 1)
# trendline = slope * x + intercept
# plt.plot(x, trendline, color='red', linestyle='--', label='Trendline')
#
# # 添加图例和标签
# plt.legend()
# plt.title('价格与排量的相关性')
# plt.xlabel('排量')
# plt.ylabel('价格')

# 显示图形
plt.show()

# 标准化
# 分类字段
cat_colums = ['燃油类型','挡位','驱动类型','环保标准','颜色']
onehot = OneHotEncoder(drop='first')
cat_features = onehot.fit_transform(df[cat_colums]).toarray()
# print(len(cat_features))
# 数值字段
num_columns = ['里程','发布时间','排量','转手次数',]
standardScaler = StandardScaler()
# 标准化
num_features = standardScaler.fit_transform(df[num_columns])
# print(len(num_features))
# 构建x和y
x = np.hstack([cat_features,num_features])
y = df['价格'].to_numpy()

# 建模
# 数据集划分
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=30)
# 评估标准
# 评价指标函数定义，其中R2的指标可以由模型自身得出，后面的score即为R2

def evaluation(model):
    ypred = model.predict(x_test)
    mae = mean_absolute_error(y_test, ypred)
    mse = mean_squared_error(y_test, ypred)
    rmse = math.sqrt(mse)
    print("MAE: %.2f" % mae)
    print("MSE: %.2f" % mse)
    print("RMSE: %.2f" % rmse)
    return ypred

# knn k取值为3
# model_knn = KNeighborsRegressor(n_neighbors=3)
# model_knn = model_knn.fit(x_train, y_train)
# print("params: ", model_knn.get_params())
# print("train score: ", model_knn.score(x_train, y_train))
# print("test score: ", model_knn.score(x_test, y_test))
# predict_y = evaluation(model_knn)
# ypred = model_knn.predict(x_test)
# for i in range(10):
#     print(f'样本{i+1} - 真实值：{y_test[i]}，预测值：{ypred[i]}')

# 误差百分比
# def calculate_error_percentage(y_test, y_pred):
#     error = np.abs(y_test - ypred)  # 计算误差
#     error_percentage = (error / y_test) * 100  # 计算误差百分比
#     return error_percentage
#
# error_percentage = calculate_error_percentage(y_test, ypred)
# labal_name = ['0%以下','1%-5%','6%-10%','11%-20%','21%-30%','31%-40%','41%-50%','50%以上']
# arr = [0,0,0,0,0,0,0,0]
# for i, error_percentage in enumerate(error_percentage):
#
#     if float(error_percentage) <= 0:
#         arr[0]+=1
#     elif 0 < float(error_percentage) <= 5 :
#         arr[1]+=1
#     elif 5 < float(error_percentage) <= 10 :
#         arr[2]+=1
#     elif 10 < float(error_percentage) <= 20 :
#         arr[3]+=1
#     elif 20 < float(error_percentage) <= 30:
#         arr[4]+=1
#     elif 30 < float(error_percentage) <= 40:
#         arr[5]+=1
#     elif 40 < float(error_percentage) <= 50:
#         arr[6]+=1
#     elif 50<float(error_percentage) :
#         arr[7]+=1
#
# plt.figure(figsize=(10, 6))
# plt.bar(labal_name,arr,color='skyblue')  # 绘制直方图
# for a,b in zip(labal_name,arr): #柱子上的数字显示
#     plt.text(a,b,'%.2f'%b,ha='center',va='bottom');
# plt.xlabel('误差百分比')
# plt.ylabel('频率数量')
# plt.title('误差百分比分布图')
# plt.grid(axis='y', alpha=0.75)
# plt.show()
#
#
# plt.figure(figsize=(10,10))
# plt.title('KNN-真实值预测值对比')
# plt.plot(predict_y[:50], 'ro-', label='预测值')
# plt.plot(y_test[:50], 'go-', label='真实值')
# plt.legend()
# plt.show()
# #
# # 随机森林 版本0.22决策树个数默认100
model_rfr = RandomForestRegressor(n_estimators=200,random_state=30)
model_rfr.fit(x_train, y_train)
print("params: ", model_rfr.get_params())
print("train score: ", model_rfr.score(x_train, y_train))
print("test score: ", model_rfr.score(x_test, y_test))
predict_y = evaluation(model_rfr)
ypred = model_rfr.predict(x_test)
for i in range(100):
    print(f'样本{i+1} - 真实值：{y_test[i]}，预测值：{ypred[i]}')

# 误差百分比
def calculate_error_percentage(y_test, y_pred):
    error = np.abs(y_test - ypred)  # 计算误差
    error_percentage = (error / y_test) * 100  # 计算误差百分比
    return error_percentage

error_percentage = calculate_error_percentage(y_test, ypred)
labal_name = ['0%以下','1%-5%','6%-10%','11%-20%','21%-30%','31%-40%','41%-50%','50%以上']
arr = [0,0,0,0,0,0,0,0]
for i, error_percentage in enumerate(error_percentage):

    if float(error_percentage) <= 0:
        arr[0]+=1
    elif 0 < float(error_percentage) <= 5 :
        arr[1]+=1
    elif 5 < float(error_percentage) <= 10 :
        arr[2]+=1
    elif 10 < float(error_percentage) <= 20 :
        arr[3]+=1
    elif 20 < float(error_percentage) <= 30:
        arr[4]+=1
    elif 30 < float(error_percentage) <= 40:
        arr[5]+=1
    elif 40 < float(error_percentage) <= 50:
        arr[6]+=1
    elif 50<float(error_percentage) :
        arr[7]+=1

plt.figure(figsize=(10, 6))
plt.bar(labal_name,arr,color='skyblue')  # 绘制直方图
for a,b in zip(labal_name,arr): #柱子上的数字显示
    plt.text(a,b,'%.2f'%b,ha='center',va='bottom');
plt.xlabel('误差百分比')
plt.ylabel('频率数量')
plt.title('误差百分比分布图')
plt.grid(axis='y', alpha=0.75)
plt.show()

plt.figure(figsize=(10,10))
plt.title('随机森林-真实值预测值对比')
plt.plot(predict_y[:50], 'ro-', label='预测值')
plt.plot(y_test[:50], 'go-', label='真实值')
plt.legend()
plt.show()



# XGBoost max_depth树的最大深度 也是用来避免过拟合的。max_depth越大，模型会学到更具体更局部的样本 n_estimators 基本分类器的数量
# model_xgbr = XGBRegressor(n_estimators = 200, max_depth=7, random_state=30)
# model_xgbr.fit(x_train, y_train)
# print("params: ", model_xgbr.get_params())
# print("train score: ", model_xgbr.score(x_train, y_train))
# print("test score: ", model_xgbr.score(x_test, y_test))
# predict_y = evaluation(model_xgbr)
# ypred = model_xgbr.predict(x_test)
# for i in range(10):
#     print(f'样本{i+1} - 真实值：{y_test[i]}，预测值：{ypred[i]}')
# # 误差百分比
# def calculate_error_percentage(y_test, y_pred):
#     error = np.abs(y_test - ypred)  # 计算误差
#     error_percentage = (error / y_test) * 100  # 计算误差百分比
#     return error_percentage
#
# error_percentage = calculate_error_percentage(y_test, ypred)
# labal_name = ['0%以下','1%-5%','6%-10%','11%-20%','21%-30%','31%-40%','41%-50%','50%以上']
# arr = [0,0,0,0,0,0,0,0]
# for i, error_percentage in enumerate(error_percentage):
#
#     if float(error_percentage) <= 0:
#         arr[0]+=1
#     elif 0 < float(error_percentage) <= 5 :
#         arr[1]+=1
#     elif 5 < float(error_percentage) <= 10 :
#         arr[2]+=1
#     elif 10 < float(error_percentage) <= 20 :
#         arr[3]+=1
#     elif 20 < float(error_percentage) <= 30:
#         arr[4]+=1
#     elif 30 < float(error_percentage) <= 40:
#         arr[5]+=1
#     elif 40 < float(error_percentage) <= 50:
#         arr[6]+=1
#     elif 50<float(error_percentage) :
#         arr[7]+=1
#
# plt.figure(figsize=(10, 6))
# plt.bar(labal_name,arr,color='skyblue')  # 绘制直方图
# for a,b in zip(labal_name,arr): #柱子上的数字显示
#     plt.text(a,b,'%.2f'%b,ha='center',va='bottom');
# plt.xlabel('误差百分比')
# plt.ylabel('频率数量')
# plt.title('误差百分比分布图')
# plt.grid(axis='y', alpha=0.75)
# plt.show()
#
# plt.figure(figsize=(10,10))
# plt.title('XGBR-真实值预测值对比')
# plt.plot(predict_y[:50], 'ro-', label='预测值')
# plt.plot(y_test[:50], 'go-', label='真实值')
# plt.legend()
# plt.show()

