#！Python3
# coding:utf-8

import numpy as np
import pandas as pd
import pprint
import matplotlib.pyplot as plt

# 2、  显示索引、列和底层的numpy数据：
print('2、  显示索引、列和底层的numpy数据')
df = pd.DataFrame({
                'A': ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'],
                'B': ['one', 'one', 'two', 'three', 'two', 'two', 'one', 'three'],
                'C': np.random.randn(8),
                'D': np.random.randn(8)})
print('行标签 df.index: ')
print(df.index)
print()
print('列标签 df.columns: ')
print(df.columns)
print()
print('值 df.values: ')
pprint.pprint(df.values)
print()
print('快速统计汇总 df.describe(): ')
pprint.pprint(df.describe())
print()

# 八、Reshaping
# 详情请参阅 Hierarchical Indexing 和 Reshaping。

print('八、Reshaping')
df = pd.DataFrame({
                'A': ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'],
                'B': ['one', 'one', 'two', 'three', 'two', 'two', 'one', 'three'],
                'C': np.random.randn(8),
                'D': np.random.randn(8)})

pprint.pprint(df.groupby('A').sum())

pprint.pprint(df.groupby(['A', 'B']).sum())

print('=' * 40)

tuples = list(zip(*[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                    ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']]))

index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])

df = pd.DataFrame(np.random.randn(8, 2), index=index, columns=['A', 'B'])

df2 = df[:4]

pprint.pprint(df)
pprint.pprint(df2)

stacked = df2.stack()

pprint.pprint(stacked)
print('=' * 40)

df = pd.DataFrame({
    'A': ['one', 'one', 'two', 'three'] * 3,
    'B': ['A', 'B', 'C'] * 4,
    'C': ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 2,
    'D': np.random.randn(12),
    'E': np.random.randn(12)})

pprint.pprint(df)

pprint.pprint(pd.pivot_table(df, values='D', index=['A', 'B', 'C'],
                             columns=['C']))

print('=' * 40)

# 九、时间序列
# Pandas在对频率转换进行重新采样时拥有简单、强大且高效的功能
# （如将按秒采样的数据转换为按5分钟为单位进行采样的数据）。
# 这种操作在金融领域非常常见。具体参考：Time Series section。

print('九、时间序列')
rng = pd.date_range('1/1/2017', periods=1000, freq='S')
ts = pd.Series(np.random.randint(0, 500, len(rng)), index=rng)
pprint.pprint(ts.resample('5min').sum())
print('=' * 40)

#    # 4、  时期和时间戳之间的转换使得可以使用一些方便的算术函数。
prng = pd.period_range('1999Q1', '2000Q4', freq='Q-NOV')
ts = pd.Series(np.random.randn(len(prng)), prng)
pprint.pprint(ts)
ts.index = (prng.asfreq('M', 'e') + 1).asfreq('H', 's') + 9
pprint.pprint(ts)
print('=' * 40)


# 十、 Categorical
# 从0.15版本开始，pandas可以在DataFrame中支持Categorical类型的数据，
# 详细 介绍参看：categorical introduction和API documentation。

df = pd.DataFrame({'id': [1, 2, 3, 4, 5, 6],
                   'raw_grade': ['a', 'b', 'b', 'a', 'a', 'e']})
df['grade'] = df['raw_grade'].astype('category')
pprint.pprint(df['grade'])
print('.' * 20)

df['grade'].cat.categories = ['very good', 'good', 'very bad']
pprint.pprint(df['grade'])
print('.' * 20)

df['grade'] = df['grade'].cat.set_categories(['very bad', 'bad', 'medium',
                                              'good', 'very good'])
pprint.pprint(df['grade'])
print('.' * 20)

pprint.pprint(df.sort_values('grade'))
pprint.pprint(df.groupby('grade').size())


# 十一、 画图
# 具体文档参看：Plotting docs

ts = pd.Series(np.random.randn(1000),
               index=pd.date_range('1/1/2000', periods=1000))
ts = ts.cumsum()
ts.plot()
plt.show()      # 显示图
print('=' * 40)

df = pd.DataFrame(np.random.randn(1000, 4), index=ts.index,
                  columns=list('ABCD'))

df = df.cumsum()

plt.figure()
df.plot()
plt.legend(loc='best')
plt.show()      # 显示图
print('.' * 40)

df3 = pd.DataFrame(np.random.randn(1000, 2), columns=['B', 'C']).cumsum()
df3['A'] = pd.Series(list(range(len(df))))
df3.plot(x='A', y='B')
plt.show()
