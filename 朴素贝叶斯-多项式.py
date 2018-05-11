#! Python3
# coding:utf-8

import numpy as np
from sklearn.naive_bayes import MultinomialNB
import pprint


# 在scikit-learn中，提供了3中朴素贝叶斯分类算法：
# GaussianNB(高斯朴素贝叶斯)、
# MultinomialNB(多项式朴素贝叶斯)、
# BernoulliNB(伯努利朴素贝叶斯)

# 多项式朴素贝叶斯：
# sklearn.naive_bayes.MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
# 主要用于离散特征分类，例如文本分类单词统计，以出现的次数作为特征值

# 参数说明：
# alpha：浮点型，可选项，默认1.0， 添加拉普拉修/Lidstone平滑参数
# fit_prior：布尔型，可选项，默认True，表示是否学习先验概率，参数为False表示所有类标记具有相同的先验概率
# class_prior：类似数组，数组大小为(n_classes,)，默认None，类先验概率

X = np.array([[1, 2, 3, 4], [1, 3, 4, 4], [2, 4, 5, 5], [2, 5, 6, 5],
              [3, 4, 5, 6], [3, 5, 6, 6]])
y = np.array([1, 1, 4, 2, 3, 3])
clf = MultinomialNB(alpha=2.0)
temp = clf.fit(X,y)
print('clf.fit(X,y) is ', temp, '\n')

# ②经过训练后，观察各个属性值
#  class_log_prior_：各类标记的平滑先验概率对数值，其取值会受fit_prior和class_prior参数的影响
# a、若指定了class_prior参数，不管fit_prior为True或False，class_log_prior_取值是class_prior转换成log后的结果
X = np.array([[1, 2, 3, 4], [1, 3, 4, 4], [2, 4, 5, 5], [2, 5, 6, 5],
              [3, 4, 5, 6], [3, 5, 6, 6]])
y = np.array([1, 1, 4, 2, 3, 3])
clf = MultinomialNB(alpha=2.0, fit_prior=True,
                    class_prior=[0.3, 0.1, 0.3, 0.2])
clf.fit(X, y)
temp = clf.class_log_prior_
print('clf.class_log_prior_ is ', temp)
print('class_prior取对数后的值是 ',
      np.log(0.3), np.log(0.1), np.log(0.3), np.log(0.2),
      '\n')


clf1 = MultinomialNB(alpha=2.0, fit_prior=False,
                     class_prior=[0.3, 0.1, 0.3, 0.2])
clf1.fit(X, y)
temp = clf1.class_log_prior_
print('clf1.class_log_prior_ is ', temp)
print('.' * 60)

# b、若fit_prior参数为False，class_prior=None，
# 则各类标记的先验概率相同等于类标记总个数N分之一
X = np.array([[1, 2, 3, 4], [1, 3, 4, 4], [2, 4, 5, 5], [2, 5, 6, 5],
              [3, 4, 5, 6], [3, 5, 6, 6]])
y = np.array([1, 1, 4, 2, 3, 3])
clf = MultinomialNB(alpha=2.0, fit_prior=False)
clf.fit(X, y)
print('各类标记的先验概率是 ', clf.class_log_prior_)
print('类标记总个数分之一的对数是 ', np.log(1/4))
print('.' * 60)

# c、若fit_prior参数为True，class_prior=None，
# 则各类标记的先验概率相同等于各类标记个数除以各类标记个数之和
X = np.array([[1, 2, 3, 4], [1, 3, 4, 4], [2, 4, 5, 5], [2, 5, 6, 5],
              [3, 4, 5, 6], [3, 5, 6, 6]])
y = np.array([1, 1, 4, 2, 3, 3])
clf = MultinomialNB(alpha=2.0, fit_prior=True)
clf.fit(X, y)
print('各类标记的先验概率是 ', clf.class_log_prior_)
print('各类标记个数除以各类标记个数之和的对数是 ',
      np.log(2/6), np.log(1/6), np.log(2/6), np.log(1/6))
print('.' * 60)

# intercept_：将多项式朴素贝叶斯解释的class_log_prior_映射为线性模型，
# 其值和class_log_propr相同
print('clf.class_log_prior is ', clf.class_log_prior_)
print('clf.intercept_ is \t', clf.intercept_)
print('.' * 60)

# feature_log_prob_：指定类的各特征概率(条件概率)对数值，
# 返回形状为(n_classes, n_features)数组
print('指定类的各特征概率(条件概率)对数值是 \n')
pprint.pprint(clf.feature_log_prob_)
print('.' * 60)
# 特征的条件概率=（指定类下指定特征出现的次数+alpha）
#  /（指定类下所有特征出现次数之和+类的可能取值个数*alpha）

# 特征条件概率计算过程，以类为1各个特征对应的条件概率为例
alpha = 2       # 参数设置
# np.array[0] = [1, 2, 3, 4]
# np.array[1] = [1, 3, 4, 4]
# 指定类下所有特征出现次数之和prob_sum_class1
prob_sum_class1 = 1+2+3+4 + 1+3+4+4
# 类的可能取值个数=4
pprint.pprint([np.log((1 + 1 + alpha)/(prob_sum_class1 + 4 * alpha)),
               np.log((2 + 3 + alpha)/(prob_sum_class1 + 4 * alpha)),
               np.log((3 + 4 + alpha)/(prob_sum_class1 + 4 * alpha)),
               np.log((4 + 4 + alpha)/(prob_sum_class1 + 4 * alpha))])
print('.' * 60)

# coef_：将多项式朴素贝叶斯解释feature_log_prob_映射成线性模型，
# 其值和feature_log_prob相同
print('clf.coef_ is ')
pprint.pprint(clf.coef_)

# class_count_：训练样本中各类别对应的样本数，按类的顺序排序输出
print('clf.class_count_ is ')
pprint.pprint(clf.class_count_)

# feature_count_：各类别各个特征出现的次数，返回形状为(n_classes, n_features)数组
print('clf.feature_count_ is  ')
pprint.pprint(clf.feature_count_)
print('class 1 的各特征出现的次数')
print([(1+1), (2+3), (3+4), (4+4)])    #以类别1为例
# np.array[0] = [1, 2, 3, 4]
# np.array[1] = [1, 3, 4, 4]
# 两两相加得到各特征出现的次数
print('.' * 60)
print('')

# fit(X, y, sample_weight=None)：根据X、y训练模型
X = np.array([[1, 2, 3, 4], [1, 3, 4, 4], [2, 4, 5, 5], [2, 5, 6, 5],
              [3, 4, 5, 6], [3, 5, 6, 6]])
y = np.array([1, 1, 4, 2, 3, 3])
clf = MultinomialNB(alpha=2.0, fit_prior=True)
temp = clf.fit(X, y)
pprint.pprint(temp)
print('')
# get_params(deep=True)：获取分类器的参数，以各参数字典形式返回
pprint.pprint(clf.get_params(True))
print('')
print('The accuracy of Naive Bayes Classifier is', clf.score(X, y))
print('.-_' * 20)
# partial_fit(X, y, classes=None, sample_weight=None)：对于数据量大时，提供增量式训练，在线学习模型参数，参数X可以是类似数组或稀疏矩阵，在第一次调用函数，必须制定classes参数，随后调用时可以忽略
clf = MultinomialNB(alpha=2.0,fit_prior=True)
# clf.partial_fit(X, y)   不写classes是错误的
temp = clf.partial_fit(X, y, classes=[1, 2])
temp = clf.partial_fit(X, y)
pprint.pprint(temp)
print('')

# predict(X)：在测试集X上预测，输出X对应目标值
temp = clf.predict([[1, 3, 5, 6], [3, 4, 5, 4]])
print('clf.predict is ', temp)
print()

# predict_log_proba(X)：测试样本划分到各个类的概率对数值
clf = MultinomialNB(alpha=2.0, fit_prior=True)
print(clf.fit(X, y))
print()
print('clf.predict_log_proba is ')
temp = clf.predict_log_proba([[3, 4, 5, 4], [1, 3, 5, 6]])
pprint.pprint(temp)

# predict_proba(X)：输出测试样本划分到各个类别的概率值
temp = clf.predict_proba([[3,4,5,4],[1,3,5,6]])
print()
print('clf.predict_proba is ')
pprint.pprint(temp
)
# score(X, y, sample_weight=None)：输出对测试样本的预测准确率的平均值
print()
print('clf.score([[3,4,5,4],[1,3,5,6]],[1,1]) is ')
temp = clf.score([[3, 4, 5, 4], [1, 3, 5, 6]], [1, 1])
pprint.pprint(temp)

# set_params(**params)：设置估计器的参数
print()
print(clf.set_params(alpha=1.0))
