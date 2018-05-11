#! Python3
# coding:utf-8

import numpy as np
from sklearn.naive_bayes import GaussianNB


# 在scikit-learn中，提供了3中朴素贝叶斯分类算法：
# GaussianNB(高斯朴素贝叶斯)、
# MultinomialNB(多项式朴素贝叶斯)、
# BernoulliNB(伯努利朴素贝叶斯)

# 1、高斯朴素贝叶斯：sklearn.naive_bayes.GaussianNB(priors=None)
X = np.array([[-1, -1], [-2, -2], [-3, -3], [-4, -4], [-5, -5],
              [1, 1], [2, 2], [3, 3]])
y = np.array([1, 1, 1, 1, 1, 2, 2, 2])
clf = GaussianNB()  # 默认priors=None
temp = clf.fit(X, y)
print('clf.fit(X,y): ', temp)
print('获取各个类标记对应的先验概率 clf.priors: ', clf.priors)
clf.set_params(priors=[0.625, 0.375])    # 设置估计器priors参数
print('获取各个类标记对应的先验概率 clf.priors: ', clf.priors)
print('clf.class_prior_: ', clf.class_prior_)
print('获取各类标记对应的训练样本数 clf.class_count_: ', clf.class_count_)
print('获取各个类标记在各个特征上的均值 clf.theta_: ', clf.theta_)
print('获取各个类标记在各个特征上的方差 clf.sigma_: ', clf.sigma_)

# fit(X, y, sample_weight=None)：训练样本，
# X表示特征向量，y类标记，
# sample_weight表各样本权重数组

# 设置样本不同的权重
temp = clf.fit(X, y, np.array([0.05, 0.05, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2]))
print(temp)
print('获取各个类标记在各个特征上的均值 clf.theta_:', clf.theta_)
print('获取各个类标记在各个特征上的方差 clf.sigma_: ', clf.sigma_)
# 对于不平衡样本，类标记1在特征1均值及方差计算过程：
# 均值 = ((-1*0.05)+(-2*0.05)+(-3*0.1)+(-4*0.1+(-5*0.1)))
# / (0.05+0.05+0.1+0.1+0.1) = -3.375
# 方差 = ((-1+3.375)**2*0.05 +(-2+3.375)**2*0.05
# +(-3+3.375)**2*0.1+(-4+3.375)**2*0.1+(-5+3.375)**2*0.1)
# / (0.05+0.05+0.1+0.1+0.1) = 1.73437501
print('*' * 60)

# partial_fit(X, y, classes=None, sample_weight=None)：增量式训练，
# 当训练数据集数据量非常大，不能一次性全部载入内存时，可以将数据集划分若干份，
# 重复调用partial_fit在线学习模型参数，在第一次调用partial_fit函数时，
# 必须制定classes参数，在随后的调用可以忽略
X = np.array([[-1, -1], [-2, -2], [-3, -3], [-4, -4], [-5, -5],
              [1, 1], [2, 2], [3, 3]])
y = np.array([1, 1, 1, 1, 1, 2, 2, 2])
clf = GaussianNB()      # 默认priors=None
temp = clf.partial_fit(X, y, classes=[1, 2],
                       sample_weight=np.array([0.05, 0.05, 0.1, 0.1,
                                               0.1, 0.2, 0.2, 0.2]))
print('clf.partial_fit: ', temp)
print(' clf.class_prior_: ',  clf.class_prior_ )

# predict(X)：直接输出测试集预测的类标记
temp = clf.predict([[-6,-6],[4,5]])
print('clf.predict([[-6,-6],[4,5]]) is ', temp)
print('')

# predict_proba(X)：输出测试样本在各个类标记预测概率值
temp = clf.predict_proba([[-6,-6],[4,5]])
print('clf.predict_proba([[-6,-6],[4,5]]) is ', '\n', temp)

# predict_log_proba(X)：输出测试样本在各个类标记上预测概率值对应对数值
temp = clf.predict_log_proba([[-6,-6],[4,5]])
print('clf.predict_log_proba([[-6,-6],[4,5]]) is ', '\n', temp)

print('')

# score(X, y, sample_weight=None)：返回测试样本映射到指定类标记上的得分(准确率)
temp = clf.score([[-6, -6], [-4, -2], [-3, -4], [4, 5]], [1, 1, 2, 2])
print('clf.score is ', temp)

temp = clf.score([[-6, -6], [-4, -2], [-3, -4], [4, 5]], [1, 1, 2, 2],
                 sample_weight=[0.3, 0.2, 0.4,0.1])
print('clf.score with weight is ', temp)

