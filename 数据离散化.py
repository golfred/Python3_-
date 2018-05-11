#! Python3
# -*- coding: utf-8 -*-

"""
    Created on Mon Nov 27 16:00:56 2017

    @author: jingchen
"""


import pandas as pd
import numpy as np
import pprint


print('.-_' * 20)
print('---------start---------')

df=pd.read_excel(r'20170920KPI关联DATA.XLSX')
print('df[0:7]: ')
print(df[0:7])
print()
print('df.loc[0:7]')
print (df.loc[0:7])
print('. ' * 30)
#df_used=df.iloc[:,2:]

#将df的中文名换成英文
df.columns=["KQI_Degradate_Flag",
"site_name",
"E_RAB_Estab.-SR",
"RRC_Recon._rate",
"RRC_Estab.-SR",
"DL_PRB_avg_Utilz._rate",
"PDCCH_avg_Utilz._rate",
"Maximum_num_of_users",
"Mean_num_of_users",
"UL_PRB_avg_Utilz._rate",
"UE_Context_DR",
"E_RAB_Context_DR",
"radio_UL_num_of_Pckt_loss",
"radio_UL_num_of_Pckt_loss_rate",
"radio_DL_num_of_Pckt_loss",
"radio_DL_num_of_Pckt_loss_rate",
"DL_avg_Pckt_delay",
"Inner_HO-SR",
"X2_HO-SR",
"DL_dualstream_rate",
"Chan1_RSSI",
"Chan2_RSSI",
"Chan3_RSSI",
"Chan4_RSSI",
"RSRP_Coverage_rate",
"CQI_up_seven",
"DLSCH_16QAM_TB_rate",
"DLSCH_64QAM_TB_rate",
"DLSCH_QPSK_TB_rate",
"RSSI",
"Avg_distance"]
print('df.columns: ')
pprint.pprint (df.columns)
print('.. ' * 20)

#制定离散规则
split_policy={"E_RAB_Estab.-SR":["+",[99.5]],
        "RRC_Recon._rate":["-",[2]],
        "RRC_Estab.-SR":["+",[99.5]],
        "DL_PRB_avg_Utilz._rate":["-",[40,80]],
       "Maximum_num_of_users":["-",[200]],
       "Mean_num_of_users":["-",[150]],
       "UL_PRB_avg_Utilz._rate":["-",[80,40]],
       "UE_Context_DR":["-",[0.2]],
       "E_RAB_Context_DR":["-",[0.15]],
       "DL_avg_Pckt_delay":["-",[35]],
       "Inner_HO-SR":["+",[98]],
       "X2_HO-SR":["+",[95]],
       "RSRP_Coverage_rate":["+",[85]],
       "CQI_up_seven":["+",[90]],
       "RSSI":["-",[-95]],
       "Avg_distance":["-",[500]]}
feature_columns=[x for x in list(df.columns[1:]) if x in split_policy]
print('feature_columns: 被选择离散化的列的index： ')
print(feature_columns)
print()
print('.' * 40)
df_used=df[feature_columns]


# 离散化用的函数: discretization(symbol, policy, value, column_idx)
# symbol； 用+或-号表示指标正相关还是负相关。
# policy： 离散化用的门限值， 可以为1个或多个。
# value; 需要离散化的数据， 为Series类型。
# column_idx: 数据类编号， 从0开始。
# 返回和value相同长度的离散化数据， 为Series类型
def discretization(symbol, policy, value, column_idx):
    # 初始化r， 与value长度相同的dataset， 值都为0， index与value相同
    r = pd.Series([0]*len(value) , index=value.index)
    # r初始赋值为column_idx * 100
    r += column_idx * 100
    i_start = 1

    if symbol=='+':
        # 合格数据：[value>=x]的index对应的r赋值为column_idx*100+i_start
        # 不合格数据：[value<x]的index对应的r值为初始值column_idx * 100。
        policy.sort()
        for x in policy:
            # value>=x是一个dataset, index与value的一样，返回True或False
            r[value>=x] = column_idx * 100 + i_start
            i_start += 1
    elif symbol=='-':
        # 合格数据：[value<x]的index对应的r赋值为column_idx*100+i_start
        # 不合格数据：[value>=x]的index对应的r值为初始值column_idx * 100。
        policy.sort(reverse=True)
        for x in policy:
            r[value<x] = column_idx * 100 + i_start
            i_start += 1
    return r


df_lisan=df[df.columns[0]]

# 每一个列进行离散化
for i in range(len(feature_columns)):
    # 取得离散化的列的index， 赋值给col_name
    col_name=feature_columns[i]
    # 取得离散化的门限值， 赋值给policy
    policy=split_policy[col_name][1]
    # 对每一列进行离散化， 赋值给r。但r是没有列名的。
    r= discretization(split_policy[col_name][0], policy,
                       df[col_name], 0)

    # 将r数据加上列名， 赋值给X
    X=pd.DataFrame(r, columns=[col_name])
    # 将离散以后的列数据拼接起来， 赋值给df_lisan
    # axis=1 表示按列拼接。
    df_lisan=pd.concat([df_lisan,X], axis=1)

print('.' * 40)
print('df_lisan:')
pprint.pprint(df_lisan.head(5))
pprint.pprint(df_lisan.tail(5))

df_lisan.to_csv("lisan.csv")

print('---------end---------')
print('.-_' * 20)
