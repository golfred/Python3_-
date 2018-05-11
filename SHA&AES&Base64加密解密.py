"""
            加密IMSI	                 SOURCE_IMSI
        qNO5Tjr6U8qjCgSMZMvnGA==	460028218511709
        B2+pCRaktmJ2lhgoECfXCQ==	460008243193488
        v38p5KqKzxRvQ4neueXQ4w==	460008543196075
        CA7UY0Hh9bo/kDuBf2NNRw==	460004760233060
        eL20HxBlvjlG4s0JiSiQyg==	460023005559579
        69a65kU8IUtehk2kyhrOHQ==	460022017458058
        ufhiOqpvBlRhVUXP+rG5HQ==	460006744147805
        DR8kVb41uKCbdE0Tawv9+A==	460004170612479
        HejDQQnaVE6TxXRjmcC+Mg==	460001641679744
        wnVMUdIr1MV7hBafBfMJzA==	460001941558412
        IjHebnWWzHhd2lOGXZicig==	460006351373526
        JLaNjrPiQS9IYs8/XljN6w==	460001641651277
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
from Crypto.Cipher import AES
import base64


# AES 加密

"""
    aes加密算法
    padding : PKCS5
"""


class AESUtil:

    __BLOCK_SIZE_16 = BLOCK_SIZE_16 = AES.block_size

    @staticmethod
    def encryt(str, key):
        cipher = AES.new(key, AES.MODE_ECB)
        x = AESUtil.__BLOCK_SIZE_16 - (len(str) % AESUtil.__BLOCK_SIZE_16)
        # 填充到Block Size的整数倍
        if x != 0:
            str = str + chr(x)*x
        # 转换成bytes类型
        msg = cipher.encrypt(str.encode("utf-8"))
        # 将base64编码后的bytes类型转为str类型
        msg = base64.b64encode(msg).decode("utf-8")
        return msg

    @staticmethod
    def decrypt(enStr, key):
        cipher = AES.new(key, AES.MODE_ECB)
        decryptByts = base64.b64decode(enStr)
        msg = cipher.decrypt(decryptByts)
        msg = msg.decode("utf-8")
        # 去除填充的字符
        paddingLen = ord(msg[len(msg) - 1])
        msg = msg[0:-paddingLen]
        return msg



###########################################################

secretStr = 'cmcc.sec.ipms.20150701'  # 密码
sha = hashlib.sha1(secretStr.encode('utf-8'))   #字符串先要转出bytes格式
keyBytes = sha.digest()              # 得到byte格式的hash码
print("sha.digest(): ", keyBytes)
print("length of sha:: ", len(keyBytes))

keyBytes = keyBytes[0:16]               # 密钥
print("keyBytes[0:16]= ", keyBytes)
print("length of key:: ", len(keyBytes))
print('-' * 60)

###########################################################
imsi = "460004170611355"
ciphered_imsi = "AJZuCN/v6oJkGAKh2TSE2A=="

print(imsi, " is ciphed to ", AESUtil.encryt(imsi, keyBytes))
print(ciphered_imsi, " is deciphered to ", AESUtil.decrypt(ciphered_imsi, keyBytes))

