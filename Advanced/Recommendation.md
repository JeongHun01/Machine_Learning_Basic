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
고객과 비슷한 다른 사람을 찾아 그의 선호도를 보여주는 방식</br></br>
**유형**</br>
**최근접 이웃 기반(Nearest Neighbor)** : 사용자 기반(User-user CF), 아이템 기반(Item-item CF)</br>
**잠재 요인 기반(Latent Factor)** : 행렬 분해 기반(Matrix Factorization)

### **Nearest Neighbor**
User behavior(item 구매이력, 영화 평점 이력, . . .)에만 기반하여 추천 알고리즘들을 지칭 </br>
상품, 영화 등 사용자가 아직 평가하지 않은 item에 대한 평가(rating)를 예측 하는 것이 주요 역할 </br>
데이터 세트를 사용자-row, 아이템-column 변환 필요 (pivot_table(value, index, column) 이용) </br>
row기준으로 유사한 vector의 요소를 빈 공간에 추천</br></br>

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
\# 빈 공간(NaN)의 예측값들을 구하는 과정</br>
step3. 예측 평점이 높은 순서대로 추천
