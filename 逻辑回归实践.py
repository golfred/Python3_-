#! Python3
# coding:utf-8

# 导入pandas与numpy工具包。
import pandas as pd
import numpy as np
import pprint


correlationChecked = False   # 是否做相关性检测和丢弃

print('.-_' * 30)
print('Start of BernoulliNB test')
filename1 = 'lisan.csv'
# 使用pandas.read_csv函数从本地读取指定数据。
# read_csv默认有表头， 如果没有表头，则head=None
dataset1 = pd.read_csv(filename1, encoding='gbk',
                       names=None,
                       index_col=['row_id']
                       )
print('Loaded data file {0} with {1} rows'.format(filename1, len(dataset1)
                                                  )
      )
print('原始数据： ')
pprint.pprint(dataset1.head(5))
print('_' * 60)

# 去除重复行
temp = dataset1.index
dataset1.drop_duplicates(subset=None, keep='first', inplace=True)
print('去重复行后： \nLoaded data file {0} with {1} rows'.format(filename1, len(dataset1)
                                                           )
      )

# 丢弃带有缺失值的数据（只要有一个维度有缺失）。
dataset1 = dataset1.dropna(how='any')

# 输出data的数据量和维度。
print(dataset1.shape)
print()
print('.-.-' * 20)



if correlationChecked:
    # ----相关性分析----
    print('-.' * 30)
    print('各列的相关性分析： ')
    print('\t相关性R的取值范围为[-1, 1]')
    print('\t• 0.8‐1.0 极强相关')
    print('\t• 0.6‐0.8 强相关')
    print('\t• 0.4‐0.6 中等程度相关')
    print('\t• 0.2‐0.4 弱相关')
    print('\t• 0.0‐0.2 极弱相关或无相关')
    temp = dataset1.corr()
    print('各列相关性： ')
    pprint.pprint(temp)
    print()

    # temp等于各列与isBadCell的相关性
    temp = temp['KQI_Degradate_Flag']
    dataset_new = dataset1.copy()
    print('.' * 30)

    # 剔除与cause value极弱相关的列（R<=0.15)
    # temp.index为各行的标签，相当于原始数据各列的标签
    for i in temp.index:
        if abs(temp[i]) <= 0.15:
            dataset_new = dataset_new.drop(i, axis=1)
            print('删除列：', i.rjust(12), '\tR = ', round(temp[i], 4))
    print('删除极弱相关数据列后的dataset_new:')
    pprint.pprint(dataset_new.head(5))
    print('.' * 30)
    temp = dataset_new.corr()
    print('dataset_new各列相关性： ')
    #pprint.pprint(temp)
    print()


    # 剔除特征中强相关的列（R>=0.7)
    temp = temp['UE_Context_DR']
    for i in temp.index:
        if 1.0 > temp[i] >= 0.7:
            dataset_new = dataset_new.drop(i, axis=1)
            print('删除强相关列：', i.rjust(12), '\tR = ', round(temp[i], 4))
    print('.' * 30)
    temp = dataset_new.corr()


    temp = temp['Maximum_num_of_users']
    for i in temp.index:
        if 1.0 > temp[i] >= 0.7:
            dataset_new = dataset_new.drop(i, axis=1)
            print('删除强相关列：', i.rjust(12), '\tR = ', round(temp[i], 4))
    print('.' * 30)
    print('删除极强相关数据列后的dataset_new:')
    print('dataset_new各列相关性： ')
    temp = dataset_new.corr()
    #pprint.pprint(temp)
    print()
else:
    dataset_new = dataset1.copy()

# 类型转换为float64
dataset_new = dataset_new.astype('float64')

# 使用sklearn.model_selection里的train_test_split模块用于分割数据。
from sklearn.model_selection import train_test_split


# 随机采样25%的数据用于测试，剩下的75%用于构建训练集合。
X_train, X_test, y_train, y_test = \
    train_test_split(dataset_new.values[:, 1:],
                     dataset_new.values[:, 0],
                     test_size=0.25,
                     random_state=33
                     )

print('Split {0} rows into train '
      'with {1} and test with {2}'.format(len(dataset_new),
                                          len(X_train),
                                          len(X_test)
                                          )
      )
print('输入数据完毕！')
print('.-_' * 30)


# 从sklearn.preprocessing里导入StandardScaler。
from sklearn.preprocessing import StandardScaler
# 从sklearn.linear_model里导入LogisticRegression与SGDClassifier。
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

# 标准化数据，保证每个维度的特征数据方差为1，均值为0。使得预测结果不会被某些维度过大的特征值而主导。
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)


# 初始化LogisticRegression与SGDClassifier。
lr = LogisticRegression()
# 在0.19的版本中，SDGClassifier默认的迭代次数是5，0.21版本默认的迭代次数是1000.
# 要想不出现warnings只需手动设定SDGClassifier的迭代次数，代码如下：
sgdc = SGDClassifier(max_iter=50)

# 调用LogisticRegression中的fit函数/模块用来训练模型参数。
lr.fit(X_train, y_train)
# 使用训练好的模型lr对X_test进行预测，结果储存在变量lr_y_predict中。
lr_y_predict = lr.predict(X_test)

# 调用SGDClassifier中的fit函数/模块用来训练模型参数。
sgdc.fit(X_train, y_train)
# 使用训练好的模型sgdc对X_test进行预测，结果储存在变量sgdc_y_predict中。
sgdc_y_predict = sgdc.predict(X_test)


# 从sklearn.metrics里导入classification_report模块。
from sklearn.metrics import classification_report

# 使用逻辑斯蒂回归模型自带的评分函数score获得模型在测试集上的准确性结果。
print('Accuracy of LR Classifier:', lr.score(X_test, y_test))
# 利用classification_report模块获得LogisticRegression其他三个指标的结果。
print(classification_report(y_test, lr_y_predict, target_names=['Benign', 'Malignant']))

 # 使用随机梯度下降模型自带的评分函数score获得模型在测试集上的准确性结果。
print('Accuarcy of SGD Classifier:', sgdc.score(X_test, y_test))
# 利用classification_report模块获得SGDClassifier其他三个指标的结果。
print(classification_report(y_test, sgdc_y_predict, target_names=['Benign', 'Malignant']))
