#! Python3
# coding = utf-8

import pprint
"""
第一种方法 使用XML2Dict模块
"""
import encoder

obj = encoder.XML2Dict()

xmlFile = open('descript.xml', 'r', encoding='UTF-8')
xmlStr = xmlFile.read()
# print(xmlStr)
print('---------' * 10)
dict = obj.parse(xmlStr)
# pprint.pprint(dict)


""" 
第二种方法：使用ET
"""
print('########' * 10)

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

tree = ET.ElementTree(file='descript.xml')
root = tree.getroot()
print('--------' * 10)
print('child_of_root in root:')
for child_of_root in root:
    print(child_of_root.tag, child_of_root.text, child_of_root.attrib)

print('--------' * 10)
print('elements in tree.iter():')
no_packages = 1
for elem in tree.iter():
    if elem.tag == 'package':
        print(elem.tag, ':', no_packages)
        no_packages += 1
    else:
        print('    ', elem.tag, ':', elem.text)

print()
print('--------' * 10)
print('elements in root[6]：packages')
no_packages = 1
for elem in root[6]:
    print('<No. %d>' % no_packages)
    for sub_elem in elem:
        print(sub_elem.tag.rjust(12), ':', sub_elem.text, sep='  ')

    no_packages += 1
    print()

print('finished')
print('--------' * 10)

"""将packages转换成dict
"""
dictPackages = {}
no_packages = 1
for elem in root[6]:
    bakFileName = ''
    packageData = []
    for sub_elem in elem:
        if sub_elem.tag == 'bakFile':
            bakFileName = sub_elem.text
        else:
            packageData.append(int(sub_elem.text) if sub_elem.text.isdigit() else sub_elem.text)
    dictPackages.update({bakFileName: packageData})
    no_packages += 1
print('bakFile :')
print('    [packageName,feature,bakType,pkgSize,sdSize]')
for eachItem in dictPackages.items():
    print(eachItem[0], ':')
    print('    ', eachItem[1])
