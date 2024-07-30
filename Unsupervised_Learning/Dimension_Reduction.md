# **Dimension Reduction Algorithm**
차원 축소를 통해 데이터를 좀 더 잘 설명할 수 있는 잠재적인 요소를 추출하는 것 </br></br>
차원이 커질 수록 데이터 포인터들간 거리 증가 + 데이터가 희소화(sparse) </br>
-> feature들간 거리에 기반한 ML 알고리즘 무력화 + 다수의 feature에 다중 공선성 문제로 모델 성능 저하 </br></br>
해결법 : 원본 데이터의 정보를 최대한 유지한 채로 차원 축소 진행</br>
-> 학습 데이터 크기를 줄여 학습 시간 절약 + 불필요한 피처 줄여서 모델 성능 향상 + 다차원 데이터를 3차원 이하로 축소하여 시각적으로 쉬운 패턴 인지 </br>
활용 분야 : 추천 엔진, 이미지 분류 및 변환(CNN이 성능이 더 높음), 문서 토픽 모델링</br>

**Feature Selection** : 특정 feature에 종속성이 강한 불필요한 feature은 아예 제거하고 데이터의 특징을 잘 나타내는 주요한 feature만 선택 </br>
**Feature Extraction** : 기존 feature를 저차원의 중요 feature로 압축해서 추출 (기존 feature의 특성을 반영하지만 새로운 feature로 추출하는 것) </br>
\# 단순 압축이 아닌, feature를 함축적으로 더 잘 설명할 수 있는 또 다른 공간으로 mapping -> 차원 축소에서 사용하는 방법 </br></br>
**대표적인 종류들** </br>
PCA, LDA, SVD, NMF </br>
기본적으로 fit으로 학습하고 transform으로 변환 DataFrame 반환

## **PCA(Principal Component Analysis)**
고차원의 원본 데이터를 저차원의 부분 공간으로 투영하여 데이터를 축소하는 기법 - sklearn.decomposition / PCA()</br>
원본 데이터가 가지는 데이터의 변동성을 가장 중요한 정보로 간주하며 이 변동성에 기반한 원본 데이터 투영으로 차원 축소 진행 </br></br>
**진행 순서** </br>
step1. 원본 데이터의 공분산 행렬 추출후 고유값 분해 (방향성을 유지하며 차원을 낮추기 위함)</br>
\# 공분산 행렬 : 두 변수간의 변동을 포함한 정방향 행렬(대칭행렬-고유값 분해에 용이) </br>
step2. 데이터 변동성이 가장 큰 방향(고유값이 큰) 순서대로 PCA의 변환 차수만큼 벡터 축 생성</br>
\#이때, n번 째 PCA축은 n-1번 째 PCA축과 직교한다 </br>
step3. 새로운 축으로 데이터 투영(projection) </br>
step4. 새로운 축 기준으로 데이터 표현 (원본 데이터를 고유벡터로 선형변환 - Ax = λv)</br>
step5. 벡터 축의 개수만큼의 차원으로 원본 데이터가 축소 된다 (orthogonal basis vector's span) </br>
고유벡터 - PCA의 주성분 벡터로 데이터 분산이 큰 방향 / 고유값 - 입력 데이터 분산 </br> (선형대수학적 논리 부분은 따로 노트에 필기되어 있음)</br></br>

**파라미터** : n_components - PCA 변환 차수로 축소시킬 차원</br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;나머지는 default해도 괜찮지만 필요시 구글링</br>
</br>
**속성** : .explaned_variance_ratio_ - 개별 PCA 컴포넌트 별 차지하는 변동성 비율 반환 (fit만 해도 만들어짐)

**유의사항** : PCA를 적용하기 전 개별 feature들 스케일링(일반적으론 표준화) 필수 - PCA는 scale에 영향을 받기 때문</br></br>
클래스 값 없이, 데이터 분석만으로 중요한 정보를 찾아 축소를 하기에 비지도 학습이라 불린다 </br> 
이후 컴포넌트에 따라 데어티의 target의 분포 파악 가능 + 모델로 학습하여 성능 평가

## **LDA(Linear Discriminant Analysis)**
지도 학습의 분류에서 쉽게 사용할 수 있도록, 개별 클래스를 분별할 수 있는 기준을 최대한 유지하며 차원 축소 </br>
\- sklearn.discriminant_alaysis / LinearDiscriminantAnalysis() </br></br>
PCA는 가장 큰 변동성의 축을 찾았다면, LDA는 입력 데이터의 결정 값 클래스를 최대한 분리할 수 있는 축을 찾음</br>
-> 같은 클래스는 최대한 근접, 다른 클래스는 최대한 떨어뜨리는 축 매핑 </br>
 = 서로 다른 클래스간의 분산은 크게, 개별 클래스 내부 분산은 최대한 작게하는 축을 찾아 차원을 축소</br></br>

 **진행 순서** (PCA와 행렬 종류 제외 동일)</br>
 step1. 클래스 간 분산과 클래스 내부 분산 행렬 생성 후 고유값 분해 </br>
 step2. 고유값이 큰 순서대로 LDA 변환 차수만큼 벡터 축 생성 </br>
 step3. 이후 선형 변환을 통해 원본 데이터 변환</br></br>

 **유의점** : PCA와 다르게 분류 class값이 주어져야 하므로 fit 파라미터에 feature와 target을 모두 넣어준다 </br>
 ->사실상 완전한 비지도 학습이라 보기엔 어려움 존재

 ## **SVD(Singular Value Decomposition)**
원본이 아닌 다른 정방행렬의 고유값 분해를 이용한 PCA, LDA와 달리 원본 행렬 자체의 특이값 분해를 통해 차원 축소하는 기법 </br>
\- sklearn.decomposition / TruncatedSVD()</br></br>
**유형** </br>
Full SVD : 기본적인 특이값 분해(mxm, mxn, nxn) - 이는 차원 축소가 일어나지 않기에 이를 변형해 아래 방식들을 사용</br>
Compact SVD : 시그마(특이값)의 비대각 부분 제거 + 대각 원소가 0인 부분을 제거하여 차원 축소 </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\#원본 행렬의 종속성이 높을수록 Σ값이 줄어져 0인 경우가 많아짐</br>
Truncated SVD : Compact SVD의 시그마 대각 원소 가운데 상위 r개만 추출하여 차원 축소 - 일반적인 SVD 차원축소 방법</br></br>
일반적으로 A를 Truncated SVD으로 행렬 분해후 나온 U'Σ'Vt' = A'로 차원 축소가 진행 </br>
A = m X n matrix -> U' = m과 Latent Factor의 관계정도 / Σ' = Latent Factor / Vt' = n의 Latent Factor 구성 </br>
사이킷런에선 원본 행렬에 UΣ 적용하여 차원 축소</br></br>
**분해 API 사용법** </br>
numpy.linalg / svd()</br>
U, Σ, Vt = svd(A) - matrix A 대입시 분해하여 순서대로 U,Σ,Vt를 ndarray로 반환 </br>
U, Σ, Vt = svds(A) - matrix A 대입시 Truncated SVD를 수행하여 ndarray로 반환</br>
원본 복귀 : np.dot을 쓰되, sigma는 np.diag(sigma)로 0포함 대칭행렬 생성 후 적용</br>

**파라미터** : n_components - 특이값 r의 수로 축소시킬 차원 </br></br>

SVD 데이터에 스케일링 적용 시 PCA와 동일한 결과를 가져온다 - PCA가 SVD으로부터 유도 되었으며 / SVD는 원점기반, PCA는 평균 기반이기 때문

## **NMF(Non Negative Matrix Factorization)**
원본 행렬 내의 모든 원소가 양수임이 보장되면 좀 더 간단하게 두 개의 기반 양수 행렬로 분해될 수 있는 기법 </br>
\- sklearn.decomposition / NMF() </br></br>
V = W * H </br>
W = 원본 행렬의 행 크기는 같지만 열 크기는 작은 행렬 </br>
H = 원본 행렬의 행 크기는 작지만 열 크기는 같은 행렬
</br>
각각 Latent Factor를 특성으로 가진다 - W : 원본 행에 대한 잠재 요소의 값 / H : 원본 열에 대한 잠재 요소의 값 </br></br>

파라미터 : n_components