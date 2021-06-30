#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

url = 'C:/Users/JWKIM/PROJECT/연관규칙/DATA/소액연관분석 상품_kt만_상품명 삭제후_번호재지정1.xlsx' #파일경로 확인
df = pd.read_excel(url)

df_1 = df[['주문번호', '상품명']]
df_1.주문번호.unique().shape, df_1.상품명.unique().shape #unique행 갯수 확인


# In[3]:


df_1.head(20)


# In[4]:


import re
regex = "\(.*\)|\s-\s.*" #가로안에 포함된 텍스트 삭제
for i in range(len(df_1)):
    df_1['상품명'][i] = re.sub(regex, '', df_1['상품명'][i])
df.head()


# In[4]:


df_1.head()


# In[5]:


df_1["상품명"] = df_1['상품명'].str.replace(" ", "") #공백 제거
df_1["상품명"] = df_1['상품명'].str.replace('[a-zA-Z]', "") #영어제거
df_1["상품명"] = df_1['상품명'].str.replace(r'[^\w]', "") #특수문자 제거
df_1["상품명"] = df_1['상품명'].str.replace('[0-9]', "") #숫자문자 제거
df_1.head(20)


# In[8]:


order_ID = list(df_1.주문번호.unique())
item_NAME = list(df_1.상품명.unique())
orderItems = [[] for _ in range(73190)] #unique행 갯수+1로 수정


# In[12]:


# order_id별로 데이터 정리(중요)
num = 0

for i in df_1.상품명:
  orderItems[df_1.주문번호[num]].append(i)
  num += 1


# In[19]:


ddd = pd.DataFrame(orderItems)


# In[61]:


ddd[ddd.columns[0:3]].head(10) #컬럼명이 없고, 레인지 인텍스형태로 되있을경우


# In[42]:


ddd.head(10)


# In[21]:


# 첫 번째 빈 리스트 제거, 아이템 중복 제거
orderItems.pop(0)

#아이템 중복 제거
num = 0

for i in orderItems:
  orderItems[num] = list(set(orderItems[num]))
  num += 1


# In[66]:


orderItems = pd.DataFrame(orderItems)


# In[68]:


orderItems[orderItems.columns[0:3]].head(10)


# In[22]:


#TransactionEncoder() 원핫인코딩
from mlxtend.preprocessing import TransactionEncoder

TSE = TransactionEncoder()
Transac_Array = TSE.fit_transform(orderItems)


# In[24]:


order_DF = pd.DataFrame(Transac_Array, columns = TSE.columns_)


# In[34]:


order_DF[['멀티탭','생수','캡슐커피','티포트','종이컵','벽시계', '모니터', '건전지', '티백차', '음료', '액상차']].head(10)


# In[22]:


# 열은 상품명을, 행은 주문번호를 나타내며, 주문번호별로 주문한 상품명을 True로 표기 
order_DF = pd.DataFrame(Transac_Array, columns = TSE.columns_)

# support 0.05 이상 주문 추출
# use_colnames : item_name 으로 출력
# max_len : 주문의 최대 길이 지정
frequent_itemsets = apriori(order_DF, min_support = 0.0015, use_colnames = True, max_len = None)


# In[23]:


frequent_itemsets


# In[24]:


# Support가 최소 0.05 이상인 연관관계 출력
# antecedents(조건절) -> consequents(결과절)
# 전체 주문 중 조건절과 결과절을 포함한 비율
# 방향성 없음

sup = association_rules(frequent_itemsets, metric = 'support', min_threshold = 0.002)
sup


# In[29]:


sup.to_excel(excel_writer='연관분석_kt(support0.0015).xlsx') # 엑셀파일로 저장


# In[25]:


# Confidence가 최소 0.3인 연관관계 출력
# 조건절이 있을 때 결과절도 있는 비율
# 조건부 확률
# 방향성 존재

association_rules(frequent_itemsets, metric = 'confidence', min_threshold = 0.3)

# Lift가 최소 0.1 이상인 연관관계 출력
# Lift가 1이라면 조건절과 결과절은 독립 관계
# 1보다 크거나 작다면 우연이 아닌, 필연적 관계
# 1보다 큼 : 함께 거래될 가능성 있음
# 1보다 작음 : 함께 거래될 가능성 적음

association_rules(frequent_itemsets, metric = 'lift', min_threshold = 0.1)


# In[26]:


lift = association_rules(frequent_itemsets, metric = 'lift', min_threshold = 0.1)

lift.sort_values(by = 'lift', ascending = False) # lift순으로 정렬


# In[27]:


#시각화
import matplotlib.pyplot as plt
import numpy as np

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.001).sort_values(by = ['lift', 'confidence', 'support'], ascending =False)
print('rules.shape :', rules.shape)
display(rules.head(10))
display(rules.tail(10))


# In[28]:


fit = np.polyfit(rules['lift'], rules['confidence'], 1)
fit_fn = np.poly1d(fit)
plt.plot(rules['lift'], rules['confidence'], 'yo', rules['lift'], fit_fn(rules['lift']))


# In[ ]:




