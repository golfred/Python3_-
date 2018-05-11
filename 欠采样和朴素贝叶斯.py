#! Python3
# coding:utf-8

import numpy as np
import pandas as pd
import pprint
# 使用sklearn.cross_valiation里的train_test_split模块用于分割数据。
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB


print('.-_' * 30)
print('Start of BernoulliNB test')
filename1 = 'lisan.csv'
# 使用pandas.read_csv函数从本地读取指定数据。
# read_csv默认有表头， 如果没有表头，则head=None
dataset1 = pd.read_csv(filename1, encoding='gbk',
                       names=None,
                       index_col=['row_id']
                      )
print('原始数据：Loaded data file {0} with {1} rows'.format(filename1, len(dataset1)
                                                  )
      )
#print('原始数据： ')
#pprint.pprint(dataset1.head(5))
print('_' * 60)

# 去除重复行
temp = dataset1.index
dataset1.drop_duplicates(subset=None, keep='first', inplace=True)
print('去重后的数据：Loaded data file {0} with {1} rows'.format(filename1, len(dataset1)
                                                  )
      )

# 丢弃带有缺失值的数据（只要有一个维度有缺失）。
dataset1 = dataset1.dropna(how='any')


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
temp = temp['KQI_Degradate_Flag']
dataset_new = dataset1.copy()
print('.' * 30)

# 剔除与cause value极弱相关的列（R<=0.15)
# temp.index为各行的标签，相当于原始数据各列的标签
for i in temp.index:
    if abs(temp[i]) <= 0.15:
        dataset_new = dataset_new.drop(i, axis=1)
        print('删除弱相关列：', i.rjust(12), '\tR = ', round(temp[i], 4))
# print('删除极弱相关数据列后的dataset_new:')
# pprint.pprint(dataset_new.head(5))
print('.' * 30)
temp = dataset_new.corr()
# print('dataset_new各列相关性： ')
# pprint.pprint(temp)
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
pprint.pprint(temp)
print()

dataset = dataset_new
print('dataset： ')
print(len(dataset))

# ----------下采样的方法，从负样本里取等量正样本的数据。---------

# Number of dataset points in the minority class
# 异常样本数量
number_records_fraud = len(dataset[dataset.KQI_Degradate_Flag == 1])
print('nuber_records_fraud = ', number_records_fraud)
# 异常样本的索引
fraud_indices = np.array(dataset[dataset.KQI_Degradate_Flag == 1].index)
print('fraud_indices: ', '\n', fraud_indices)
print()

# Picking the indices of the normal classes
# 正常样本的索引拿出
normal_indices = dataset[dataset.KQI_Degradate_Flag == 0].index
print('normal_indices: ')
pprint.pprint(normal_indices)
print()

# Out of the indices we picked, randomly select "x" number(number_records_fraud)
# choice在正常样本中随机选择指定数量的索引值。
random_normal_indices = np.random.choice(normal_indices,
                                         number_records_fraud,
                                         replace=False)

# 索引转为np.array
random_normal_indices = np.array(random_normal_indices)
print('random_normal_indices: ')
pprint.pprint(random_normal_indices)
print('.' * 60)

# Appending the 2 indices合并两部分索引
under_sample_indices = np.concatenate([fraud_indices, random_normal_indices])
print('under_sample_indices: ')
pprint.pprint(under_sample_indices)
print('.' * 60)

print('dataset.index: ')
pprint.pprint(dataset.index)
print('.' * 60)


# Under sample datasetset
# 取出下采样数据，iloc[行号，列号]
under_sample_data = dataset.loc[under_sample_indices]
print('欠采样后的数据 under_sample_data: ')
pprint.pprint(len(under_sample_data))
print('.' * 60)

# 随机采样25%的数据用于测试，剩下的75%用于构建训练集合。
X_train, X_test, y_train, y_test = \
    train_test_split(under_sample_data.values[:, 1:],
                     under_sample_data.values[:, 0],
                     test_size=0.25,
                     random_state=33
                     )

print('Split {0} rows into train '
      'with {1} and test with {2}'.format(len(under_sample_data),
                                          len(X_train),
                                          len(X_test)
                                          )
      )
print('输入数据完毕！')
print('.-_' * 30)

clf = BernoulliNB()

temp = clf.fit(X_train, y_train)
print('clf.fit(X_train, y_train) is ', temp, '\n')


# class_count_：训练样本中各类别对应的样本数，按类的顺序排序输出
print('clf.class_count_ is ')
pprint.pprint(clf.class_count_)
print()



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

clf = MultinomialNB()

temp = clf.fit(X_train, y_train)
print('clf.fit(X_train, y_train) is ', temp, '\n')


# class_count_：训练样本中各类别对应的样本数，按类的顺序排序输出
print('clf.class_count_ is ')
pprint.pprint(clf.class_count_)
print()


y_predict = clf.predict(X_test)
print('y_predict is: ')
pprint.pprint(y_predict)
print()
# 使用模型自带的评估函数进行准确性测评。
print('The Accuracy of MultinomialNB is', clf.score(X_test, y_test))
print('.' * 60)

# 依然使用sklearn.metrics里面的classification_report模块对预测结果做更加详细的分析。
from sklearn.metrics import classification_report

print(classification_report(y_test, y_predict,
                            target_names=['cause=0', 'cause=1']
                            )
      )

print('._' * 40)


clf = GaussianNB()

temp = clf.fit(X_train, y_train)
print('clf.fit(X_train, y_train) is ', temp, '\n')


# class_count_：训练样本中各类别对应的样本数，按类的顺序排序输出
print('clf.class_count_ is ')
pprint.pprint(clf.class_count_)
print()


# ②经过训练后，观察各个属性值

y_predict = clf.predict(X_test)
print('y_predict is: ')
pprint.pprint(y_predict)
print()
# 使用模型自带的评估函数进行准确性测评。
print('The Accuracy of GaussianNB is', clf.score(X_test, y_test))
print('.' * 60)

# 依然使用sklearn.metrics里面的classification_report模块对预测结果做更加详细的分析。
from sklearn.metrics import classification_report

print(classification_report(y_test, y_predict,
                            target_names=['cause=0', 'cause=1']
                            )
      )

print('._' * 40)

