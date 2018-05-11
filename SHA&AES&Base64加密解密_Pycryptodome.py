"""
            加密IMSI	                 SOURCE_IMSI
        qNO5Tjr6U8qjCgSMZMvnGA==	460028218511709
        B2+pCRaktmJ2lhgoECfXCQ==	460008243193488
        v38p5KqKzxRvQ4neueXQ4w==	460008543196075
        CA7UY0Hh9bo/kDuBf2NNRw==	460004760233060
        I4vn6OwIkCId1vEMrH+3JQ==	460029021626473
        zX6zVPeYk/j8jTS3aFweIw==	460001641655681
        B7/Kawu0QQJfbXRbYco5Tg==	460028009509047
        xPzEweo4/LjRo2OdTiykVw==	460007353118845
        +5qGeVhCfXozqqrf3zn5xw==	460007144120460
        AJZuCN/v6oJkGAKh2TSE2A==	460004170611355
"""

# !/usr/bin/env python
# coding=utf-8

import hashlib
from Cryptodome.Cipher import AES
import base64


###########################################################
secretStr = 'cmcc.sec.ipms.20150701'  # 密码
sha = hashlib.sha1(secretStr.encode('utf-8'))   # 字符串先要转出bytes格式
keyBytes = sha.digest()              # 得到byte格式的hash码
print("hash: ", keyBytes)
print("length of hash:: ", len(keyBytes))

keyBytes = keyBytes[0:16]               # 密钥
print("key: ", keyBytes)
print("length of key:: ", len(keyBytes))
print('-' * 60)

###########################################################

strInput = "460008243193488"
key = keyBytes

# 加密-----------------
aes_cipher = AES.new(key, AES.MODE_ECB)
x = AES.block_size - (len(strInput) % AES.block_size)
# 填充到Block Size的整数倍
if x != 0:
    strInput = strInput + chr(x) * x
strInput = bytes(strInput, encoding="utf-8")    # 字符串转换成bytes类型
ciphertext = aes_cipher.encrypt(strInput)
# Base64编码
final_text = base64.b64encode(ciphertext).decode("utf-8")
print("final_text: ", final_text)

# 解密-----------
# Base64解码
ciphertext = base64.b64decode(final_text)
strInput = aes_cipher.decrypt(ciphertext)
strInput = strInput.decode("utf-8")
# 去填充的字符
paddingLen = ord(strInput[len(strInput) - 1])
strInput = strInput[0:-paddingLen]
print("strInput: ", strInput)
