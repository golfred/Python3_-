# coding:utf-8
# 求一个正整数的阶乘


def factorial(n):
    if n == 0 or n == 1:
        result = 1
    else:
        result = n*factorial(n-1)
    return result


def factorial2(n):
    result = 1
    for i in range(1, n+1):
        result *= i
    return result


number = int(input('请输入一个正整数：'))
print('{}! = {}'.format(number, factorial(number)))
print('{}! = {}'.format(number, factorial2(number)))
