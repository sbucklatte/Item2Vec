# KTC Item2Vec Project

# Word2vec 알고리즘을 응용한 Item2Vec 방식으로 상품간 연관성을 도출
# python으로 작성 gensim 라이브러리를 활용

# 머신러닝 학습을 통해 상품간 동시에 구매가 이루어지는 상품을 벡터화하여 도출
# 실제 주문데이터 약40만건으로 학습을 진행

# 학습에 사용된 파라미터는 mincount = 50, window = 5, size = 300, model = Skip-gram

# streamlit와 heroku를 통해 웹공유
https://w2v-web.herokuapp.com/ 통해 상품간 유사도 결과를 확인가능
