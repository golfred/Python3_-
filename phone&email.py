#! python3
# coding:utf-8


import pyperclip
import re

"""
phoneRegex = re.compile(r'''(
                             (\d{3}|\(\d{3}\))?    # area code
                             (\s|-|\.)?          # separator
                             (\d{3})             # first 3 digits
                             (\s|-|\.)           # separator
                             (\d{4})             # last 4 digits
                             (\s*(ext|x|ext.)\s*(\d{2,5}))?  # extension
                            )''', re.VERBOSE
                        )
"""
phoneRegex = re.compile(r'''(
                             (0[1-9]\d)   # area code
                             (\s|-)?                # separator
                             (\d{8})                # last 8 digits
                            )''', re.VERBOSE
                        )

mobileRegex = re.compile(r'''(
                             (400|1[3-9]\d)           # first 3 digits: 400 or 1nn
                             (\d{7})                # last 7 digits
                            )''', re.VERBOSE
                        )

emailRegex = re.compile(r'''(
                             [a-zA-Z0-9._%+-]+      # username
                             @                      # @ symbol
                             [a-zA-Z0-9.-]+         # domain name
                             (\.[a-zA-Z]{2,4})      # dot-something
                            )''', re.VERBOSE
                        )

text = str(pyperclip.paste())
matches = []
for groups in phoneRegex.findall(text):
    phoneNum = '-'.join([groups[1], groups[3]])
    matches.append(phoneNum)

for groups in mobileRegex.findall(text):
    phoneNum = groups[1] + groups[2]
    matches.append(phoneNum)

for groups in emailRegex.findall(text):
    matches.append(groups[0])

if len(matches) > 0:
    pyperclip.copy('\n'.join(matches))
    print('Copied to clipboard:')
    print('\n'.join(matches))
else:
    print('No phone numbers or email addresses found.')
