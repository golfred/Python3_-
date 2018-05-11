# coding:utf-8

listPrimeNumber = [2]

boolIsPrime = True

for i in range(3, 1001, 2):
    for p in listPrimeNumber:
        if i % p == 0:
            boolIsPrime = False
            break
    if boolIsPrime:
        listPrimeNumber.append(i)
    boolIsPrime = True

print('1000以内的素数：')
print(listPrimeNumber)
