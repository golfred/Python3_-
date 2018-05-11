# coding:utf-8
__author__ = 'jliu002'
import pprint


def fibonacci(n):
    def fibo(m):
        a, b = 0, 1
        for j in range(m):
            yield b
            a, b = b, a+b

    if n < 1:
        return tuple([0])
    else:
        return tuple(fibo(n))


number = int(input('输入一个正整数：'))
temp = fibonacci(number)
f = {}
for i in range(number):
    f.update({i + 1: temp[i]})


for i in f:
    print('fibonacci sequence %3d \t--> \t%3d' % (i, f[i]))

# pprint.pprint(f)
