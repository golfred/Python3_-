#! python3
# coding:utf-8

import json
import requests


# Compute location from command line arguments.
location = '浦东'
#location += input('请输入需要查询的地点： ')

# Download the JSON data from OpenWeatherMap.org's API.
url ='http://wthrcdn.etouch.cn/weather_mini?city=%s' % location
response = requests.get(url)
response.raise_for_status()

# Load JSON data into a Python variable.
weatherData = json.loads(response.text)['data']

# Print weather descriptions.
w = weatherData['forecast']

print('==================')

for key in w:
    if len(key['fengxiang']) == 2:
        key['fengxiang']=key['fengxiang'][0]+'  '+key['fengxiang'][1]
    key['fengli'] = key['fengxiang'] + ':\t' + key['fengli'][9:-3].rjust(5)

tempStr = weatherData['ganmao'].split(sep='，')
print('今天 %s 天气:' % location)
print('感冒指数：', tempStr[1], tempStr[2])
print('当前温度：', weatherData['wendu'], '℃')
print()

for day in w:
    print(day['date'], '  ', day['type'].rjust(3), '\t', day['high'][3:], '~', day['low'][3:], '  ', day['fengli'])

print()

print('==================')
