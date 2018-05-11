#! python3
# coding:utf-8

import pprint
import numpy as np
from sklearn.naive_bayes import BernoulliNB
import pandas as pd
import random
# cross_validation模块被弃用了，改为支持model_selection这个模块
# 使用sklearn.cross_valiation里的train_test_split模块用于分割数据。
from sklearn.model_selection import train_test_split


print('.-_' * 30)
print('Start of BernoulliNB test')
filename1 = 'Naive_Beyes_with_Header.csv'
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

# 丢弃带有缺失值的数据（只要有一个维度有缺失）。
dataset1 = dataset1.dropna(how='any')

# 输出data的数据量和维度。
print(dataset1.shape)
print()
# 输出data的快速汇总统计
print('原始数据的快速汇总统计')
print(dataset1.describe())
print()
# 相关性分析
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
temp = temp['isBadCell']
dataset_new = dataset1.copy()
print('.' * 30)

# 剔除与cause value极弱相关的列（R<=0.2)
# temp.index为各行的标签，相当于原始数据各列的标签
for i in temp.index:
    if abs(temp[i]) <= 0.2:
        dataset_new = dataset_new.drop(i, axis=1)
        print('删除列：', i.rjust(12), '\tR = ', round(temp[i], 4))
print('删除极弱相关数据列后的dataset_new:')
pprint.pprint(dataset_new.head(5))
print('.' * 30)
temp = dataset_new.corr()
print('dataset_new各列相关性： ')
pprint.pprint(temp)
print()

# 剔除特征中强相关的列（R>0.6)
temp = temp[u'ERAB掉线率']
for i in temp.index:
    if 1.0 > temp[i] > 0.6:
        dataset_new = dataset_new.drop(i, axis=1)
        print('删除列：', i.rjust(12), '\tR = ', round(temp[i], 4))
print('.' * 30)
temp = dataset_new.corr()
print('删除极强相关数据列后的dataset_new:')
print('dataset_new各列相关性： ')
pprint.pprint(temp)
print()

#  ----欠采样处理----
# 将数据按isBadCell的值分类
dataset_cause1 = dataset_new[dataset_new['isBadCell'] > 0]
dataset_cause0 = dataset_new[dataset_new['isBadCell'] == 0]

# 统计isBadCell = 1的数据量
trainNumber_1 = len(dataset_new[dataset_new['isBadCell'] > 0])

dataset_cause0 = dataset_cause0.sample(n=trainNumber_1)

dataset_new = dataset_cause1.append(dataset_cause0, ignore_index=True)

# 输出data的数据量和维度。
print('欠采样后的dataset_new的数据量和维度')
print(dataset_new.shape)
print()
# 输出data的快速汇总统计
print('欠采样后的dataset_new的快速汇总统计')
print(dataset_new.describe())
print()
print('--' * 30)
print('dataset_new: ')

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

clf = BernoulliNB()

temp = clf.fit(X_train, y_train)
print('clf.fit(X_train, y_train) is ', temp, '\n')

# coef_：将多项式朴素贝叶斯解释feature_log_prob_映射成线性模型，
# 其值和feature_log_prob相同
print('clf.coef_ is ')
pprint.pprint(clf.coef_)
print()

# class_count_：训练样本中各类别对应的样本数，按类的顺序排序输出
print('clf.class_count_ is ')
pprint.pprint(clf.class_count_)
print()


# ②经过训练后，观察各个属性值
#  class_log_prior_：各类标记的平滑先验概率对数值，其取值会受fit_prior和class_prior参数的影响
temp = clf.class_log_prior_
print('clf.class_log_prior_ is ', temp)
print('.' * 60)

y_predict = clf.predict(X_test)
print('y_predict is: ')
pprint.pprint(y_predict)
print()
# 使用模型自带的评估函数进行准确性测评。
print('The Accuracy of BernoulliNB is', clf.score(X_test, y_test))
print('.' * 60)

# 依然使用sklearn.metrics里面的classification_report模块对预测结果做更加详细的分析。
from sklearn.metrics import classification_report

print(classification_report(y_test, y_predict,
                            target_names=['cause=0', 'cause=1']
                            )
      )

print('._' * 40)
