#!/usr/bin/env python
# coding: utf-8

# 라이브러리 import
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import re

url = 'C:/Users/JWKIM/PROJECT/연관규칙/DATA/소액연관분석 상품_kt만_상품명 삭제후_번호재지정1.xlsx' #파일경로 확인
df = pd.read_excel(url)

df_1 = df[['주문번호', '상품명']]
df_1.주문번호.unique().shape, df_1.상품명.unique().shape #unique행 갯수 확인

# 상품명 전처리
regex = "\(.*\)|\s-\s.*" 
for i in range(len(df_1)):
    df_1['상품명'][i] = re.sub(regex, '', df_1['상품명'][i])


df_1["상품명"] = df_1['상품명'].str.replace(" ", "") 
df_1["상품명"] = df_1['상품명'].str.replace('[a-zA-Z]', "") 
df_1["상품명"] = df_1['상품명'].str.replace(r'[^\w]', "") 
df_1["상품명"] = df_1['상품명'].str.replace('[0-9]', "") 
df_1.head(20)


order_ID = list(df_1.주문번호.unique())
item_NAME = list(df_1.상품명.unique())
orderItems = [[] for _ in range(73190)] #unique행 갯수+1로 수정


# One-hot 인코딩
num = 0

for i in df_1.상품명:
  orderItems[df_1.주문번호[num]].append(i)
  num += 1


ddd = pd.DataFrame(orderItems)
ddd[ddd.columns[0:3]].head(10) #컬럼명이 없고, 레인지 인텍스형태로 되있을경우

orderItems.pop(0)


#아이템 중복 제거
num = 0

for i in orderItems:
  orderItems[num] = list(set(orderItems[num]))
  num += 1


orderItems = pd.DataFrame(orderItems)


#TransactionEncoder()
from mlxtend.preprocessing import TransactionEncoder

TSE = TransactionEncoder()
Transac_Array = TSE.fit_transform(orderItems)


order_DF = pd.DataFrame(Transac_Array, columns = TSE.columns_)

# 열은 상품명을, 행은 주문번호를 나타내며, 주문번호별로 주문한 상품명을 True로 표기 
order_DF = pd.DataFrame(Transac_Array, columns = TSE.columns_)

# support 0.05 이상 주문 추출
# use_colnames : item_name 으로 출력
# max_len : 주문의 최대 길이 지정
frequent_itemsets = apriori(order_DF, min_support = 0.0015, use_colnames = True, max_len = None)


# antecedents(조건절) -> consequents(결과절)
# 전체 주문 중 조건절과 결과절을 포함한 비율

sup = association_rules(frequent_itemsets, metric = 'support', min_threshold = 0.002)

sup.to_excel(excel_writer='연관분석_kt(support0.0015).xlsx') # 엑셀파일로 저장


# Confidence가 최소 0.3인 연관관계 출력
# 조건절이 있을 때 결과절도 있는 비율
# 조건부 확률
# 방향성 존재

association_rules(frequent_itemsets, metric = 'confidence', min_threshold = 0.3)
association_rules(frequent_itemsets, metric = 'lift', min_threshold = 0.1)


lift = association_rules(frequent_itemsets, metric = 'lift', min_threshold = 0.1)
lift.sort_values(by = 'lift', ascending = False) # lift순으로 정렬


#시각화
import matplotlib.pyplot as plt
import numpy as np

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.001).sort_values(by = ['lift', 'confidence', 'support'], ascending =False)
print('rules.shape :', rules.shape)
display(rules.head(10))
display(rules.tail(10))


fit = np.polyfit(rules['lift'], rules['confidence'], 1)
fit_fn = np.poly1d(fit)
plt.plot(rules['lift'], rules['confidence'], 'yo', rules['lift'], fit_fn(rules['lift']))





