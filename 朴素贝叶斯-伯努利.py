import numpy as np
from sklearn.naive_bayes import BernoulliNB
import pprint

# 伯努利朴素贝叶斯：
#   sklearn.naive_bayes.BernoulliNB(alpha=1.0, binarize=0.0,
#                                   fit_prior=True,class_prior=None)
#   类似于多项式朴素贝叶斯，也主要用户离散特征分类，和MultinomialNB的区别是：
#        MultinomialNB以出现的次数为特征值，
#        BernoulliNB为二进制或布尔型特性
# 参数说明：
#    binarize：将数据特征二值化的阈值

# ①利用BernoulliNB建立简单模型
X = np.array([[1, 2, 3, 4],
              [1, 3, 4, 4],
              [2, 4, 5, 5]])
y = np.array([1, 1, 2])
clf = BernoulliNB(alpha=2.0, binarize=3.0, fit_prior=True)
temp = clf.fit(X, y)
print('**' * 30)
print(temp)
print()

# 经过binarize = 3.0二值化处理，相当于输入的X数组为
X = np.array([[0, 0, 0, 1],
              [0, 0, 1, 1],
              [0, 1, 1, 1]])
pprint.pprint(X)
print()

# ②训练后查看各属性值
# class_log_prior_：类先验概率对数值，类先验概率等于各类的个数/类的总个数
pprint.pprint(clf.class_log_prior_)
print()

# feature_log_prob_ :指定类的各特征概率(条件概率)对数值，返回形状为(n_classes, n_features)数组
pprint.pprint(clf.feature_log_prob_)
print()

# 上述结果计算过程：
#    假设X对应的四个特征为A1、A2、A3、A4，类别为y1,y2,类别为y1时，特征A1的概率为：
#     P(A1|y=y1) = P(A1=0|y=y1)*A1+P(A1=1|y=y1)*A1
pprint.pprint([np.log((2+2)/(2+2*2)) * 0 + np.log((0+2)/(2+2*2)) * 1,
               np.log((2+2)/(2+2*2)) * 0 + np.log((0+2)/(2+2*2)) * 1,
               np.log((1+2)/(2+2*2)) * 0 + np.log((1+2)/(2+2*2)) * 1,
               np.log((0+2)/(2+2*2)) * 0 + np.log((2+2)/(2+2*2)) * 1])
print()

# class_count_：按类别顺序输出其对应的个数
pprint.pprint(clf.class_count_)
print()

# feature_count_：各类别各特征值之和，按类的顺序输出，
#  返回形状为[n_classes, n_features] 的数组
pprint.pprint(clf.feature_count_)
