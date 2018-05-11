#! python3
# coding:utf-8

import pandas as pd
import numpy as np


dfc = pd.DataFrame({'A':['aaa','bbb','ccc'],'B':[1,2,3]})
dfc.loc[:, 'A'] = 11
print(dfc)

dfc = dfc.copy()
dfc['A'][0] = 111
print(dfc)
dfc.loc[0]['A'] = 1111
