# KTC Item2Vec Project

Word2vec 알고리즘을 응용한 Item2Vec 방식으로 상품간 연관성을 도출

Python과 gensim, Apriori 라이브러리를 활용

머신러닝 학습을 통해 상품간 동시에 구매가 이루어지는 상품을 벡터화하여 유사도 결과 산출

실제 주문데이터 약40만건으로 학습을 진행

학습에 사용된 파라미터는 mincount = 50, window = 5, size = 300, model = Skip-gram

streamlit와 heroku를 통해 웹공유

https://w2v-web.herokuapp.com/ 통해 상품간 유사도 결과를 확인가능

ㅇ File Index
 * w2v_Learning.py : 작성 코드(Item2vec 활용)
 * association Rules.py : 작성 코드(Apriori 활용)
 * app.py : 학습모델 활용 시각화 및 streamlit 웹 출판용
 * requirements.txt : 사용한 프로그램 및 라이브러리
 * w2v_mincnt50_win10~ : ML학습 모델 
 * Procfile, setup.sh : stteamlit 업로드용 파일
