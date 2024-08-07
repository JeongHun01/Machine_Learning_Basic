# Recommendation
사용자의 취향을 이해하고 맞춤 상품과 컨텐츠를 제공해 조금이라도 고객을 머무르게 하기 위한 시스템 </br>
정교한 추천 시스템은 사용자에게 높은 신뢰도를 얻게 되며, 이를 기반으로 서비스 프로바이더는 고객 충성도를 크게 향상 시킬 수 있다</br></br>
**주요 방식** </br>
**Content Based Filtering** : 컨텐츠에 있는 정보들을 기반으로 고객과 잘 맞는다고 판단하면 제공하는 형식 </br>
**Collaborative Filtering** : 고객과 비슷한 사람들이 선택한 상품들을 제공 </br>
\- 이들 방식 중 1가지만 선택하거나 이들을 결합하여 hybird 방식으로 이용

## **Content Based Filtering**
컨텐츠의 속성에 기반해 유사도를 가진 제품을을 추천하는 방식 </br>
문제점 : 마케팅적인 메시지로 정보가 명확하지 않게 되는 점이 존재 / 그래도 보완을 통해 자주 사용되는 방식 </br></br>

**순서** </br>
step1. 콘텐츠에 대한 여러 텍스트 정보들 벡터화 </br>
step2. 코사인 유사도로 콘텐츠별 유사도 계산 </br>
step3. 콘텐츠 별로 가중 평점을 계산</br>
step4. 유사도가 높은 콘텐츠 중에 평점이 좋은 콘텐츠 순으로 추천 </br>
다른 추가적인 요인들도 유동적으로 추가/수정하며 성능 향상 가능

## **Collaborative Filtering**
고객과 비슷한 다른 사람을 찾아 그의 선호도를 보여주는 방식</br>
surprise 패키지로 구현 가능 </br></br>
**유형**</br>
**최근접 이웃 기반(Nearest Neighbor)** : 사용자 기반(User-user CF), 아이템 기반(Item-item CF)</br>
**잠재 요인 기반(Latent Factor)** : 행렬 분해 기반(Matrix Factorization)</br></br>
데이터 세트를 사용자-row, 아이템-column 변환 필요 (pivot_table(value, index, column) 이용) </br>

### **Nearest Neighbor**
User behavior(item 구매이력, 영화 평점 이력, . . .)에만 기반하여 추천 알고리즘들을 지칭 </br>
상품, 영화 등 사용자가 아직 평가하지 않은 item에 대한 평가(rating)를 예측 하는 것이 주요 역할 </br></br>

**사용자 기반** </br>
1\. 특정 사용자와 비슷한 고객들을 기반으로 비슷한 고객들이 선호하는 다른 상품을 추천 </br>
2\. 특정 사용자와 비슷한 상품을 구매해온 고객들은 비슷한 고객으로 간주 </br>
-> Customers like you also bought this items </br></br>
**아이템 기반** </br>
1\. 특정 상품과 유사한 좋은 평가를 받은 다른 비슷한 상품 추천 </br>
2\. 사용자들로부터 특정 상품과 비슷한 평가를 받은 상품들은 비슷한 상품으로 간주 </br>
-> Customers who bought this item also bought these items </br>
row - item / column - user (일반적으로 더 많이 선호되는 방식 - 정확도가 더 높음) </br></br>

**순서**</br>
step0. 아이템 기반이라면 transpose()로 행과 열 교환 </br>
step1. row vector끼리의 유사도 추출(코사인 유사도)로 비슷한 벡터들을 파악</br>
step2. 유사도를 기준으로 평점에 가중치를 부여 (유사도가 높을 수록 평점의 영향이 더 커짐) - 개인 예측 평점 = ∑ (S(유사도) * R(평점)) / ∑ |S| </br>
\# R행렬.dot(S행렬) / np.ndarray([np.abs(S행렬).sum(axis = 1)]) - NaN(0으로 치환)부분만 따로 뽑아서 계산</br>
step3. 예측 평점이 높은 순서대로 추천</br></br>

### **Latent Factor**
사용자-아이템 평점 행렬속에 숨어 있는 잠재요인 분해(SVD)로 추출해, 추천 예측을 할 수 있게 하는 기법</br>
잠재요인을 기반으로 행렬을 재구성하며 추천을 구현 </br></br>
R ≈ P * Q^T </br>
\# P - 사용자x잠재요인 행렬로 사용자에 대한 latent factor 반영 / Q - 아이템x잠재요인 행렬로 아이템에 대한 latent factor 반영 </br>
사용자-아이템 평점 행렬(Sparse Matrix)을 Dense Matrix 형태인 P과 Q^T로 분해한 후 이를 재결합하여 Dense Matrix 형태의 사용자-아이템 행렬을 생성하여 새로운 아이템을 추천</br></br>

SVD는 기본적으로 Missing Value가 없는 행렬에 적용이 가능 </br>
-> 경사하강법을 이용해 예측 R과 실제 R의 최소한의 오류를 찾기위해 비용함수 최적화로 P와 Q 최적화 유추 / 또는 ALS이용(속도 우세)</br>
**경사하강법 버전 순서**</br>
step1. P와 Q를 임의의 값을 가진 행렬로 설정 (정규 분포 형태 권장 - np.random.normal()) </br>
step2. P와 Q^T 값을 곱해 예측 R행렬을 계산하고 예측 R행렬과 실제 R행렬에 해당하는 오류 값을 계산 </br>
step3. 이 오류값을 최소화 할 수 있도록 P와 Q행렬을 적절한 값으로 각각 업데이트 (경사 하강법) </br>
step4. 만족할만한 오류 값을 가질 때까지 2\~3을 반복하며 P와 Q를 업데이트하여 근사화</br></br>
비용 최소 함수 : ( min ∑(r - pq^t)^2 ) + ( λ(||q||^2 + ||p||^2) ) </br>
앞 괄호 - 실제값과 예측값 오류 최소화 / 뒤 괄호 - 과적합 개선을 위한 L2 규제</br>
update p/q = p/q + η(오류값 * p/q - λ * p/q)

## **Baseline Rating**
사용자의 성향(평상시에 평점을 짜게 줄수도 후하게 줄수도 있는)을 반영한 추천 기법 </br></br>
예측 평점 = u(전체 사용자의 평균 영화 평점) + bu(사용자 편향 점수) + bi(아이템 편향 점수)</br>
사용자 편향 점수 = 특정 사용자 평균 평점 - 전체 사용자 평균 평점</br>
아이템 편향 점수 = 특정 아이템 평균 평점 - 전체 사용자 평균 평점</br></br>
비용 최소 함수 = min (∑ 오차^2 + λ(bi^2 + bu^2 + ||q||^2 + ||p||^2) )

## **Surprise Package**
사이킷런에서 따로 recommendation에 대한 API는 없지만 pyhone의 surprise 패키지를 이용하면 구현 가능 (사이킷런 API와 유사)<br></br>
수행 프로세스 : 데이터 로딩 -> 모델 설정(추천 알고리즘) + 학습 -> 예측 및 평가 </br>
from surprise import . . . </br></br>
유의점 : model.test(data) = list 형태로 사용자, 아이템, 평점, 예측 평점, . . . 객체들 출력 (기존 사이킷런의 predict()와 유사)</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
predict(user, item) = 개별 한 건의 결과 prediction 객체 반환 </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
cross_validate = 사이킷런의 cross_val_score / GridSearchCV에 모델 객체가 아닌, 모델 클래스를 넣어야 함 / scoring -> measures 파라미터
</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
데이터 로딩 시 Dataset 패키지 이용 - 이때 데이터 파일 column은 user, item, rating 순서</br></br>

**주요 method** </br>
Data.load_from_file(file path, reader) = OS 파일에서 데이터를 로딩할 때 사용</br>
Data.load_from_df(df, reader) = pandas의 DataFrame에서 데이터 로딩, 사용자 아이템 평점 순 column 정해져 있어야 함 </br>
Reader(line_format, sep, rating_scale) = Dataset로 로딩 규칙(포맷)을 지정하기 위해 사용</br>
\- line_format : column 순서 / sep : column 분리자(default = '\t') / rating_scale : tuple로 평점 값의 최소 \~ 최대 </br>
accuracy.rmse(pred) = 평가</br></br>
**대표적 추천 클래스**</br>
**SVD** : 잠재적 요인 협업 필터링을 위한 SVD 알고리즘 </br>
**KNNBasic** : 최근접 이웃 협업 필터링을 위한 KNN 알고리즘 </br>
**BaselineOnly** : 사용자 Bias와 아이템 Bias를 감안한 SGD 베이스라인 알고리즘 </br>
이외 알고리즘은 surprise 사이트 문서에서 참조 </br></br>

**SVD 파라미터** : n_factors = 잠재 요인의 개수 / default = 100 / 커질수록 정확도는 높아질 수 있지만, 과적합 발생 가능</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
n_epochs = SGD(Stochastic Gradient Descent) 수행 시 반복 횟수 (업데이트 횟수) / default = 20</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
biased = 베이스라인 사용자 편향 적용 여부 / default = True