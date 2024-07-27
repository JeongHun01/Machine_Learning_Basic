## **Dimension Reduction Algorithm**
차원 축소를 통해 데이터를 좀 더 잘 설명할 수 있는 잠재적인 요소를 추출하는 것 </br></br>
차원이 커질 수록 데이터 포인터들간 거리 증가 + 데이터가 희소화(sparse) </br>
-> feature들간 거리에 기반한 ML 알고리즘 무력화 + 다수의 feature에 다중 공선성 문제로 모델 성능 저하 </br></br>
해결법 : 원본 데이터의 정보를 최대한 유지한 채로 차원 축소 진행</br>
-> 학습 데이터 크기를 줄여 학습 시간 절약 + 불필요한 피처 줄여서 모델 성능 향상 + 다차원 데이터를 3차원 이하로 축소하여 시각적으로 쉬운 패턴 인지 </br>
활용 분야 : 추천 엔진, 이미지 분류 및 변환(CNN이 성능이 더 높음), 문서 토픽 모델링</br>

**Feature Selection** : 특정 feature에 종속성이 강한 불필요한 feature은 아예 제거하고 데이터의 특징을 잘 나타내는 주요한 feature만 선택 </br>
**Feature Extraction** : 기존 feature를 저차원의 중요 feature로 압축해서 추출 (기존 feature의 특성을 반영하지만 새로운 feature로 추출하는 것) </br>
\# 단순 압축이 아닌, feature를 함축적으로 더 잘 설명할 수 있는 또 다른 공간으로 mapping 