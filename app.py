#streamlit run w2b_streamlit1.py

import streamlit as st
from gensim.models import Word2Vec
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import re
from PIL import Image
from gensim.models import KeyedVectors
from collections import Counter
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/LG PC.ttf").get_name()
rc('font', family=font_name)

@st.cache(allow_output_mutation=True)
def lord():
    return KeyedVectors.load_word2vec_format('전체_w2v_mincnt50_win10_size300_work30_결과750개')
model = lord()

st.title("KTC 연관상품 찾기(테스트)")

image = Image.open('img1.jpg')
st.image(image)

st.subheader('*참고 : 상품 빈도 TOP300 리스트')
len(model.wv.vocab)
vocab = model.wv.vocab
sorted(vocab, key=vocab.get, reverse=True)[:300]
dict(Counter(vocab).most_common(300))

v1 = pd.Series(vocab.keys())

#일반
#textInput= st.text_input("상품명을 입력해주세요")
#returnOutput= st.slider("몇개의 연관상품을 찾길 원하나요?", 1, 20, 10)
#st.write('You selected:', returnOutput)

#사이드바 활용
textInput= st.sidebar.text_input("상품명을 입력해주세요")
returnOutput= st.sidebar.slider("몇개의 연관상품을 찾겠습니까?", 1, 10, 5)
st.sidebar.write('You selected:', returnOutput)

if st.button('결과보기') :
    if (textInput in vocab):
        #loads page
        similarWords = model.wv.most_similar(textInput,topn=int(returnOutput))
        st.table(pd.DataFrame(similarWords, columns=['단어', '유사도']))
        
        with st.spinner('Wait for it...'):
             time.sleep(3)
        
        st.write("<검색상품과 유사도가 높은TOP5 상품의 공통된 연관상품>")
        
        df = pd.DataFrame(model.wv.most_similar(textInput, topn=100), columns=['단어', '유사도'])
        df = df[df['단어'].isin(v1)]
        df1 = pd.DataFrame(model.wv.most_similar(df.iloc[0]['단어'], topn=100), columns=['단어', df.iloc[0]['단어']])
        df2 = pd.DataFrame(model.wv.most_similar(df.iloc[1]['단어'], topn=100), columns=['단어', df.iloc[1]['단어']])
        df3 = pd.DataFrame(model.wv.most_similar(df.iloc[2]['단어'], topn=100), columns=['단어', df.iloc[2]['단어']])
        df4 = pd.DataFrame(model.wv.most_similar(df.iloc[3]['단어'], topn=100), columns=['단어', df.iloc[3]['단어']])
        df5 = pd.DataFrame(model.wv.most_similar(df.iloc[4]['단어'], topn=100), columns=['단어', df.iloc[4]['단어']])
        dfs = [df.set_index(['단어']) for df in [df1, df2, df3, df4, df5]]
        dff = pd.concat(dfs, join='inner', axis=1).reset_index()
        dff['전체'] = dff.mean(numeric_only=True, axis=1)
        dffs = dff.sort_values(by='전체', ascending=False); dffs = dffs[:30]
        
        sns.set(style='whitegrid', font="LG PC", font_scale=1.4)
        g = sns.PairGrid(dffs.sort_values("전체", ascending=False), x_vars=dffs.columns[1:6], y_vars=["단어"])
        g.fig.set_size_inches(15,14)
        g.map(sns.stripplot, size=10, orient="h", palette="ch:s=1, r=-.1, h=1_r",
              linewidth=1, edgecolor="w")
        g.set(xlim=(0.7, 1), xlabel="유사도", ylabel="")
        titles = dffs.columns[1:6]

        for ax, title in zip(g.axes.flat, titles) : 
            ax.set(title=title)
            ax.xaxis.grid(False); ax.yaxis.grid(True)

        sns.despine(left=True, bottom=True)
        st.pyplot(g)       
        #st.table(dffs)     

        st.success('Done!')
        
    else:
        # Out of Vocabulary
        st.error("없는 상품명 입니다. 상품빈도 리스트를 참고하세요")