
# coding: utf-8

# In[3]:

from sklearn.naive_bayes import GaussianNB
from IPython.display import display
import traceback,time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
# import statsmodels.api as sm
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 480)


# In[13]:

# 指定读取的文件，单引号前有个u表示unicode字符
#df_all=pd.read_csv(u'HRW_TEST.csv')
df_all=pd.read_csv(u'AB.csv', encoding='gbk')
# df_all['sdate']=pd.to_datetime(df_all['XXXX']) #如果需要用其它列作为时间索引
# df_all.set_index('sdate', inplace=True)
# df_all.index=df_all.index.astype('datetime64[ns]') # 时间格式必须满足，否则请用下面这条，速度会慢一些。
# df_all.index=pd.to_datetime(df_all.index) # If datetime format not like YYYY-MM-DD HH:MM:SS. Use pd.to_datetime()to convert. a little bit slower.
print(df_all.head(10)) # 查看数据的前十行
# df_used=df_all.iloc[:,2:]
# df_used.head(10)


# In[14]:

X = np.array(df_all.iloc[:,2:])  
y = np.array(df_all[['cause_value']])  
clf = GaussianNB()      # 默认priors=None
clf.fit(X,y)
print()
print('clf.class_prior_', clf.class_prior_)

