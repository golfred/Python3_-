# coding:utf-8

import pprint

it = []
for i in range(1, 1000):
    it.append(i)
result = list(filter(lambda x: x % 3 == 0, it))
for i in range(len(result)):
    print('%5d' % (result[i]), end=' ')
    if (i + 1) % 10 == 0:
        print('')
print('')
pprint.pprint(result)
