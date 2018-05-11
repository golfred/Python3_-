#! python3
# coding:utf-8

import re


def strip(astring):
    strRegex = re.compile(r'(^\s+)|(\s+$)')
    print(strRegex.findall(astring))
    return strRegex.sub('', astring)


print(strip('asd fs ddf  ddd  '))
