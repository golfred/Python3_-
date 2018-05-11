#! python3
# coding:utf-8

import numpy as np
import pandas as pd
import pprint
# 使用sklearn.cross_valiation里的train_test_split模块用于分割数据。
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE    # 引入过采样


correlationChecked = False      # 是否做相关性检测和丢弃

print('.-_' * 30)
print('Start of NaiveBayes test')
print('未丢弃中间数据： ')
filename1 = 'lisan.csv'
# 使用pandas.read_csv函数从本地读取指定数据。
# read_csv默认有表头， 如果没有表头，则head=None
dataset1 = pd.read_csv(filename1, encoding='gbk',
                       names=None,
                       index_col=['row_id']
                       )
print('原始数据： \nLoaded data file {0} with {1} rows'.format(filename1, len(dataset1)
                                                  )
      )
# print('原始数据： ')
# pprint.pprint(dataset1.head(5))
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
# 输出data的快速汇总统计
# print('原始数据的快速汇总统计')
# print(dataset1.describe())
# print()


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

# ----过采样处理----
# 对训练集进行过采样，测试集不能动。
oversampler = SMOTE(random_state=0)
# os_features 过采样后的特征数据
# os_labels 过采样后的标签数据
os_features, os_labels = oversampler.fit_sample(X_train, y_train)


clf = BernoulliNB()

temp = clf.fit(os_features, os_labels)
print('clf.fit(os_features, os_labels) is ', temp, '\n')
print('原来的训练数据量： ', len(X_train))
print('过采样后的训练数据量： ', len(os_features))


# class_count_：训练样本中各类别对应的样本数，按类的顺序排序输出
print('clf.class_count_ is ')
pprint.pprint(clf.class_count_)
print()

y_predict = clf.predict(X_test)
# print('y_predict is: ')
# pprint.pprint(y_predict)
# print()
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


print()

clf = MultinomialNB()

temp = clf.fit(os_features, os_labels)
print('clf.fit(os_features, os_labels) is ', temp, '\n')
print('原来的训练数据量： ', len(X_train))
print('过采样后的训练数据量： ', len(os_features))


# class_count_：训练样本中各类别对应的样本数，按类的顺序排序输出
print('clf.class_count_ is ')
pprint.pprint(clf.class_count_)
print()

y_predict = clf.predict(X_test)
# print('y_predict is: ')
# pprint.pprint(y_predict)
# print()
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

print()

clf = GaussianNB()

temp = clf.fit(os_features, os_labels)
print('clf.fit(os_features, os_labels) is ', temp, '\n')
print('原来的训练数据量： ', len(X_train))
print('过采样后的训练数据量： ', len(os_features))


# class_count_：训练样本中各类别对应的样本数，按类的顺序排序输出
print('clf.class_count_ is ')
pprint.pprint(clf.class_count_)
print()

y_predict = clf.predict(X_test)
# print('y_predict is: ')
# pprint.pprint(y_predict)
# print()
# 使用模型自带的评估函数进行准确性测评。
print('The Accuracy of GaussianNB is', clf.score(X_test, y_test))
print('.' * 60)

# 依然使用sklearn.metrics里面的classification_report模块对预测结果做更加详细的分析。
from sklearn.metrics import classification_report

print(classification_report(y_test, y_predict,
                            target_names=['cause=0', 'cause=1']
                            )
      )

print('._' * 80)

# ------------------------------------------------------------------
# ------------------------------------------------------------------
print('丢弃中间数据（==1）算法 ：')
print('原始数据： ')
dataset_flag = dataset1[dataset1.columns[0]]
dataset_feature = dataset1[dataset1.columns[1:]]

# dataset_feature[dataset_feature==1.0] = np.nan
dataset_new = dataset_feature[dataset_feature != 1.0]
dataset_new[dataset_new == 2.0] = 1.0
print('丢弃中间值后数据： ')
pprint.pprint(len(dataset_new))

# 再拼接起来得到新的dataset1
dataset1 = pd.concat([dataset_flag, dataset_new], axis=1)
print('dataset1: ', len(dataset1))

# 丢弃带有缺失值的数据（只要有一个维度有缺失）。
dataset1 = dataset1.dropna(how='any')

# 去除重复行
temp = dataset1.index
dataset1.drop_duplicates(subset=None, keep='first', inplace=True)
print('去重复行后： \nLoaded data file {0} with {1} rows'.format(filename1, len(dataset1)
                                                  )
      )
# 输出data的数据量和维度。
print(dataset1.shape)
print()

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

    # 去除数据都是NaN的列
    dataset1 = dataset1.drop('UL_PRB_avg_Utilz._rate', axis=1)
    dataset1 = dataset1.drop('DL_PRB_avg_Utilz._rate', axis=1)


    temp = dataset1.corr()
    print('各列相关性： ')
    # pprint.pprint(temp)
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

    # pprint.pprint(temp)
    print()
else:
    dataset_new = dataset1.copy()


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

# ----过采样处理----
# 对训练集进行过采样，测试集不能动。
oversampler = SMOTE(random_state=0)
# os_features 过采样后的特征数据
# os_labels 过采样后的标签数据
os_features, os_labels = oversampler.fit_sample(X_train, y_train)


clf = BernoulliNB()

temp = clf.fit(os_features, os_labels)
print('clf.fit(os_features, os_labels) is ', temp, '\n')
print('原来的训练数据量： ', len(X_train))
print('过采样后的训练数据量： ', len(os_features))


# class_count_：训练样本中各类别对应的样本数，按类的顺序排序输出
print('clf.class_count_ is ')
pprint.pprint(clf.class_count_)
print()

y_predict = clf.predict(X_test)
# print('y_predict is: ')
# pprint.pprint(y_predict)
# print()
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


print()

clf = MultinomialNB()

temp = clf.fit(os_features, os_labels)
print('clf.fit(os_features, os_labels) is ', temp, '\n')
print('原来的训练数据量： ', len(X_train))
print('过采样后的训练数据量： ', len(os_features))


# class_count_：训练样本中各类别对应的样本数，按类的顺序排序输出
print('clf.class_count_ is ')
pprint.pprint(clf.class_count_)
print()

y_predict = clf.predict(X_test)
# print('y_predict is: ')
# pprint.pprint(y_predict)
# print()
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

print()

clf = GaussianNB()

temp = clf.fit(os_features, os_labels)
print('clf.fit(os_features, os_labels) is ', temp, '\n')
print('原来的训练数据量： ', len(X_train))
print('过采样后的训练数据量： ', len(os_features))


# class_count_：训练样本中各类别对应的样本数，按类的顺序排序输出
print('clf.class_count_ is ')
pprint.pprint(clf.class_count_)
print()

y_predict = clf.predict(X_test)
# print('y_predict is: ')
# pprint.pprint(y_predict)
# print()
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
