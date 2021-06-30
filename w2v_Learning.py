#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import re
from gensim.models import Word2Vec
from konlpy.tag import Okt
import nltk
import zipfile
from lxml import etree
from nltk.tokenize import word_tokenize, sent_tokenize
import gensim
import time
import datetime
from konlpy.tag import Okt
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/LG PC.ttf").get_name()
rc('font', family=font_name)


df = pd.read_excel("C:/Users/JWKIM/PROJECT/W2V/DATA/소액연관분석 상품_전체_상품명 삭제후_번호재지정1.xlsx")
df.head()


#텍스트 전처리
regex = "\(.*\)|\s-\s.*"
for i in range(len(df)):
    df['상품명'][i] = re.sub(regex, '', df['상품명'][i])

df["상품명"] = df['상품명'].str.replace(" ", "") 
df["상품명"] = df['상품명'].str.replace('[a-zA-Z]', "") 
df["상품명"] = df['상품명'].str.replace(r'[^\w]', "") 
df["상품명"] = df['상품명'].str.replace('[0-9]', "") 

agg = df.groupby(['주문번호'])['상품명'].agg({'unique'})

#인코딩
sentence = []
for user_sentence in agg['unique'].values:    
    #user_sentence = [word for word in user_sentence if not word in stopwords]
    sentence.append(list(map(str, user_sentence)))



start = time.time()
#W2V 학습
embedding_model = Word2Vec(sentence, size=300, window = 10, min_count=40, workers=50, sg=1)

sec = time.time()-start
print(sec)

#학습모델 확인
embedding_model.wv.vectors.shape

print(embedding_model.wv.most_similar("종이컵")) # 가장가까운 단어 출력
print(embedding_model.wv.similarity("종이컵", '안전모')) #두단어 사이의 유사도 출력

dff = pd.DataFrame(embedding_model.wv.most_similar("데이터다중화장치", topn=10), columns=['단어','유사도']) 


#가장많이 등장한 상위 품목 출력
from collections import Counter
len(embedding_model.wv.vocab)
vocab = embedding_model.wv.vocab
sorted(vocab, key=vocab.get, reverse=True)[:10]
dict(Counter(vocab).most_common(50))

min(vocab, key=vocab.get) #가장적게 등장하는 단어


# 시각화(PCA)
word_vectors = embedding_model.wv
vocabs = word_vectors.vocab.keys()
word_vectors_list = [word_vectors[v] for v in vocabs]

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
xys = pca.fit_transform(word_vectors_list)
xs = xys[:,0]
ys=xys[:,1]



def plot_2d_graph(vocabs, xs, ys):
    plt.figure(figsize=(15,10))
    plt.scatter(xs,ys,marker='o')
    for i,v in enumerate(vocabs):
        plt.annotate(v,xy=(xs[i], ys[i]))
        
plot_2d_graph(vocabs, xs,ys)


#html로 출력
# annotation text 만들기 (시각화할 때 벡터 말고 단어도 필요하니까)
# vocabs = word_vectors.vocab.keys()

text=[]
for i,v in enumerate(vocabs):
    text.append(v)


import plotly
import plotly.graph_objects as go
fig = go.Figure(data=go.Scatter(x=xs,
                                y=ys,
                                mode='markers+text',
                                text=text)) 

fig.update_layout(title='전체 Word2Vec1')
fig.show()

plotly.offline.plot(fig, filename='전체_word2vec1.html')

#모델 저장
from gensim.models import KeyedVectors
embedding_model.wv.save_word2vec_format('KT1144_w2v')

# Embedding Projector용 모델 저장
get_ipython().system('python -m gensim.scripts.word2vec2tensor --input KT1144_w2v --output KT1144_w2v')





