#! Python3
# coding:utf-8

import numpy as np
import numpy.linalg as nplg
import pprint


# Numpy简单介绍
print(np.version.version)
print('=' * 50)

x = np.array(((1, 2, 3), (4, 5, 6)))
pprint.pprint(x)

y = np.array([[1, 2, 3], [4, 5, 6]])
pprint.pprint(y)
print('=' * 50)

# numpy数据类型设定与转换
numeric_string2 = np.array(['1.23', '2.34', '3.45'], dtype=np.string_)
pprint.pprint(numeric_string2)

pprint.pprint(numeric_string2.astype(float))
print('=' * 50)

# numpy索引与切片
print(x[1, 2])
y = x[:, 1]
print(y)

y[0] = 10
print('y is ', y)
print('x is ', x)
print('=' * 50)

arr = np.arange(10)
print(arr)
arr[3:6] = 12
print(arr)

arr_copy = arr[3:6].copy()
arr_copy[:] = 24
print(arr_copy)
print(arr)

l1 = list(range(10))
l2 = l1[5:8]
l2[0] = 12
print(l2)
print(l1)
print('=' * 50)

names = np.array(['Bob', 'joe', 'Bob', 'Will'])
pprint.pprint(names == 'Bob')

data = np.array(np.random.randn(6, 4))
data[data < 0] = 0
pprint.pprint(data)
print('=' * 50)

# 数组文件输入输出
arr = np.arange(10)
np.save('some_array', arr)

arr2 = np.load('some_array.npy')
pprint.pprint(arr2)

# 存取文本文件
arr = np.array(np.random.randn(5, 3))
np.savetxt('Book1.csv', arr, delimiter=',', newline='\n')
arr = np.loadtxt('Book1.csv', delimiter=',')
pprint.pprint(arr)

print(np.arange(15))
print(np.arange(15).reshape(3, 5))
print(np.linspace(1, 10, 20))
print('np.zeros((3, 4))')
print(np.zeros((3, 4)))
print('np.ones((3, 4))')
print(np.ones((3, 4)))
print('np.eye(3)')
print(np.eye(3))

print('np.zeros((2,2,2))的维数：', end=' ')
a = np.zeros((2, 2, 2))
print(a.ndim)
print('np.zeros((2, 2, 2))的每一维大小：', end=' ')
print(a.shape)
print('数组的元素数：', end=' ')
print(a.size)
print('元素类型：', end=' ')
print(a.dtype)
print('每个元素占得字节数：', end=' ')
print(a.itemsize)
print('=' * 50)

x = np.array([[[0,  1,  2],
        [3,  4,  5],
        [6,  7,  8]],
       [[9, 10, 11],
        [12, 13, 14],
        [15, 16, 17]],
       [[18, 19, 20],
        [21, 22, 23],
        [24, 25, 26]]])
print('x.sum(axis=1): ')
pprint.pprint(x.sum(axis=1))
print('x.sum(axis=2): ')
pprint.pprint(x.sum(axis=2))
print('=' * 50)

# 合并数组
a = np.ones((2,2))
b = np.eye(2)
print(np.vstack((a,b)))
print(np.hstack((a,b)))

c = np.hstack((a,b))    # 属于深拷贝
a[1, 1] = 5
b[1, 1] = 5
print('c: ')
print(c)

b = a   # 属于浅拷贝
print('b is a: ', end=' ')
print(b is  a)

c = a.copy()    # 深拷贝
print('c is a: ', end=' ')
print(c is a)

# 基本的矩阵运算
a = np.array([[1, 0],[ 2, 3]])
print(a)
print(a.transpose())

a = np.array([[1, 0], [2, 3]])
print(nplg.eig(a))

------------------------------------------------------------------------

# 如果数组元素个数太多，numpy会自动忽略中间的元素以省略号表示，仅打印两边的元素。
# 如果想强制数组打印所有元素，可以设置set_printoptions参数：
np.set_printoptions(threshold=10000)

# 数组的算数运算适用于元素方式，即运算执行在每个元素上：
a = np.array( [20,30,40,50] )
b = np.arange( 4 )
print(a-b)
print(b**2)
print(10*np.sin(a))
print(a<35)

# 不同于其他矩阵运算，在numpy数组中，*乘法运算符操作与每个元素。而矩阵的乘积可以使用dot函数或dot方法运行：
A = np.array( [[1,1],
               [0,1]] )
B = np.array( [[2,0],
               [3,4]] )
print('A * B: ',A * B)        # 元素乘积
print('A.dot(B): ',A.dot(B))     # 矩阵乘积
print('np.dot(A, B): ',np.dot(A, B)) # 矩阵乘积

# 如+=、*=等操作符，在操作执行的地方替代数组元素而不是产生一个新的数组：
a = np.ones((2, 3), dtype=int)
b = np.random.random((2, 3))
a *= 3
print('a *= 3 ： ')
print(a)
b += a
print('b += a ： ')
print(b)
# b不能自动转换为int类型
a += b

# ndarray类提供了许多一元操作方法，比如元素求和，求最小元素等：
# 通常上述操作将数组视为包含数字的列表，忽略其shape类型。进一步，通过制定轴参数axis ，你可以在指定的维度上操作这些运算：
b = np.arange(12).reshape(3,4)
print('b:', b)
b.sum(axis=0)                            # 每一列的求和
b.min(axis=0)                            # 每列的最小值
b.min(axis=1)                            # 每行的最小值
b.cumsum(axis=1)                         # 沿着行累积求和


# 多维数组可以每轴有一个索引，这些指数存在于一个元组中，以逗号分离：
def f(x,y):
    return 10*x+y
# fromfunction(函数,shape维度,元素类型)，其中函数的参数个数由shape维度的秩决定决定，函数参数的值为数组中元素的下标。
# 比如shape为(2,2),函数参数有2个，则函数的参数值分别为：(0,0)、(0,1)、(1,0)、(1,1)。
b = np.fromfunction(f,(5,4),dtype=int)
print(b)

# 如果想对数组中的每个元素进行操作，可以使用flat属性，它是针对数组中每个元素的迭代器：
for element in b.flat:
    print(element)


# 不同的数组对象可以共享相同的数据，view方法产生一个有相同数据的数组对象：
a = np.arange(12)
a.shape = 3, 4
c = a.view()
c is a
c.base is a
c.shape = 2, 6
a.sharp         # a的shape没有改变
c[0,4] = 1234
a               # a的数据发生改变

# 数组的切片也是数组的一个视图：
s = a[ : , 1:3]     # 切片产生数组a的第二、第三列，s是一个视图
s[:] = 10
# s[:]是s的视图。注意s=10是将将s实例指向10，s[:]=10才是将切片数组s中的每个元素都设为10
print(a)

# 方法copy实现数组及其数据的复制。
d = a.copy()              # 一个包含数据的新数组对象被创建
d is a                    # False
d.base is a               # False, # 数组d不共享数组a的任何数据



