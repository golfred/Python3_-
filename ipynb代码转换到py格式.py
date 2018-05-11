#! python3
# coding:utf-8

import simplejson as json
import pprint
import os

file_path = r'F:\python\Python Machine Learning and Practice\Chapter_4'
file_name = ''
os.chdir(file_path)
print('正在处理', 'file_path', '下的ipynb文件: ')

file_list = os.listdir(file_path)

for each_file in file_list:
    file_name = each_file
    if os.path.isfile(file_name) and ('.ipynb' in file_name):
        f_read = open(os.path.join(file_path, file_name), 'rb')
        text = f_read.read()
        f_read.close()
        print('\t\t', file_name, '已经读取！')

        source_codes = []
        ipynb_text = json.loads(text)
        temp = ipynb_text['cells']
        for each_cell in temp:
            source_codes.append(''.join(each_cell['source']))

        f_write = open(os.path.join(file_path, file_name.replace('.ipynb', '.py')), 'w')

        for each_code in source_codes:
            if each_code != '':
                f_write.writelines(each_code)
                f_write.writelines('\n')

        f_write.close()
        print('\t\t写入', file_name.replace('.ipynb', '.py'), '完成！')
        print()
