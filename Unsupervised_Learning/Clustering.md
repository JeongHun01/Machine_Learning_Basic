# **Clustering Algorithm**
유사성이 높은 데이터들을 동일한 그룹으로 분류하고 서로 다른 군집들이 상이성을 가지도록 그룹화하는 알고리즘 </br>
좋은 군집일수록 뭉쳐있으며, 다른 군집과 떨어져있다(차원 축소로 확인 가능)</br></br>

**대표적인 종류들**</br>
K-Means, Mean shift, Gaussian Mixture Model, DBSCAN </br>
fit()으로 군집화 후, fit_predict()로 label 반환

## **Test_Data for Clustering**
Clustering Algorithm 성능 평가를 위한 테스트 데이터 생성 방법 - sklearn.datasets / makeblobs() </br></br>

**파리미터** : n_samples = 생성할 총 데이터의 개수 </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
n_features = 데이터 feature의 개수 (시각화가 목적일 시 2로 설정해 2차원 평면 표현)</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
centers = 군집의 개수 / ndarray로 중심점의 좌표 (=target의 개수)</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
cluster_std = 생성될 군집 내 데이터의 표준편차 </br></br>

makeblobs()를 통해 반환 된 feature data를 평가할 모델에 학습시켜 테스트 target과 비교

## **성능 평가**
**실루엣 분석** : 각 군집간의 거리가 얼마나 효율적으로 분리돼 있는지를 나타낸다 -> s(i) = ( b(i) - a(i) ) / max( a(i), b(i) )</br>
\# ai = i번째 데이터에 대해 자신이 속해있는 군집에서 다른 데이터 포인터들의 거리 평균</br>
&nbsp;&nbsp;
bi = i번째 데이터에 대해 가장 가까운 군집 내에 다른 데이터 포인터들의 거리 평균 </br>
&nbsp;&nbsp;
a(i)는 작을수록, b(i)는 클수록 좋은 지표 </br>
&nbsp;&nbsp;
계수는 -1\~1의 범위를 가지며, 1로 가까워질수록 이상적인 지표며, 0에 가까울수록 다른 군집과 가깝다는 의미 </br>
&nbsp;&nbsp;
'-'값은 아예 다른 군집 데이터 포인터가 할당 됐음을 의미 (군집화에 문제가 있음) </br></br>
개별 데이터가 가지는 군집화의 지표인 실루엣 계수(silhouette coefficient)를 기반으로 한다 </br>
이는 해당 데이터가 같은 군집 내의 데이터와 얼마나 가깝게 군집화 되어있고, 다른 군집 데이터와는 얼마나 멀리 분리되어 있는지를 나타내는 지표 </br></br>

silhouette_samples(X, labels, metric = 'euclidean', **kwds) : 실루엣 계수를 계산해 반환 </br>
silhouette_score(X, labels, metric = 'euclidean', sample_size = None **kwds) : 전체 데이터의 실루엣 계수를 평균해 반환 </br></br>
**전체 실루엣 계수 평균과 더불어 개별 군집의 평균값의 편차가 작아야지 좋은 성능의 군집이다 -> 개별 군집의 계수와 평균이 유사** </br></br>
모델에서의 n_clusters의 값, 즉 군집의 개수에 따라 실루엣 계수가 달라짐</br>
개수가 작으면 또 너무 일반화가 일어나므로, 군집화 레벨에 따른 시각화를 통해 각각 군집의 계수와 전체 평균을 비교하며 좋은 성능을 나타내는 모양의 군집의 개수를 찾는 최적화 과정이 필요

## **K-Means**
군집의 Centroid를 설정해 군집화하는 기법 - sklearn.cluster / KMeans()</br></br>
**순서** </br>
step1. n개의 군집 중심점(centroid) 설정 </br>
step2. 각 데이터들은 거리를 계산해 가장 가까운 중심점 선택 </br>
step3. 중심점은 본인에게 할당 된, 데이터들의 평균 중심(거리)으로 이동 </br>
step4. 각각의 중심점에 할당 된 데이터의 종류가 변경되지 않을 때 까지, 1\~3 반복 </br> </br>
장점 : 일반적이며 가장 많이 활용 + 쉽고 간결 + 대용량 데이터 활용 가능 </br>
단점 : 거리 기반으로 속성 수가 많으면 성능 하락(PCA활용 필요할 수 있음) + 반복 횟수 많을 시 속도 저하 + 이상치 데이터에 취약 </br></br>

**파라미터** : n_clusters = 중심점의 갯수 </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
init = 초기 중심점 좌표 설정 방식 (일반적으론 'k-means++' 설정) </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
max_iter = 최대 반복 횟수 </br></br>
**속성** : labels_ = 각 데이터가 속한 군집 중심점을 나타낸 레이블 반환</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
cluster_centers_ = 각 군집 중심점 좌표 -> 시각화 이용 가능</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
\#shape는 (군집 개수, feature 개수)를 나타냄 </br></br>

유의사항 :  fit()만 해도 군집 할당되며, fit_predict(feature)은 labels_를 2d ndarray로 / fit_transform은 데이터별로 각각의 중심점과의 거리를 2d ndarray로 반환

## **Mean Shift**
KDE(Kernel Density Estimation)를 이용하여 데이터 포인터들이 데이터 분포가 높은 곳으로 이동하며 군집화 수행 - sklearn.cluster / MeanShift()</br>
별도의 군집 개수를 지정하지 않으며 Bandwidth에 기반하여 자동으로 개수를 정함 </br>
비모수적 추정 : 데이터가 특정 분포를 따르지 않는다는 가정 하에서 밀도 추정 - 관측된 데이터 만으로 밀도 찾기 </br></br>
**순서**</br>
step1. 개별 데이터의 특정 반경 내에 주변 데이터를 포함한 데이터 분포도 계산  </br>
\# 데이터 각각에 커널 함수를 적용한 값을 모두 더한 후 데이터 건수로 나눔 = 확률 밀도 함수 PDF</br>
(쉽게 생각하면 각각 데이터들의 밀도함수들을 더해 전체 수로 나누어 하나의 연속적인 전체 밀도를 나타냄)</br>
step2. 중심점을 데이터 분포도가 가장 높은 곳으로 이동 </br>
step3. 중심점을 따라 해당 데이터들 주변 데이터와의 거리값을 kernel 함수값으로 입력한 뒤, 반환 값을 현재 위치에서 업데이트하며 이동 </br>
step4. 데이터의 움직임이 없을 때까지 1\~3 반복하여 최종 군집 중심점 찾기 </br></br>

**KDE** = (1 / n*h) * Σ( K (X - Xi) / h ) >> K : 커널 함수, h = bandwidth (표준편차와 동일) </br>
**kernel function** : Gaussian Distribution(대표적), Uniform Distribution</br>
K - Gaussian Distribution = ( 1 / (2𝝿σ^2)^(1/2) ) * e^(- ( x - μ)^2 / 2σ^2) ) = General Normal ~ N(μ, σ^2)</br>
h - 작은 h값은 좁고 spike한 KDE로 변동성이 큰 PDF를 추정 (overfitting 위험성) / 큰 h값은 과도하게 smoothing된 KDE로 단순화된 PDF를 추정 (undefitting 위험성) -> bandwidth가 작을수록 많은 중심점 생성 </br> 
**최적의 h** = (4σ^5 / 3n)^(1/5) ≈ 1.06σn^(-1/5) - estimate_bandwidth(data, quantile = sampling시 필터링 비율) 함수로 계산 및 반환 </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
데이터가 클수록 quantile을 키워 수행 시간을 줄여줄 필요 존재</br>
**KDE 시각화** : seaborn에서의 displot()이 KDE 방식으로 PDF를 나타냄 - sns.distplot(data)</br>
파라미터 - rug = 밀집도 표시 True/False, hist = 히스토그램 True/False, kde = KDE함수 True/False, . . . </br>
혹은 sns.kdeplot(data)으로 kde 함수만 자세히 파악 가능 - 파리미터 : bw = bandwidth 수치</br></br>

**파라미터** : bandwidth = bandwidth값이며 MeanShift 군집화의 성능을 결정하는 가장 중요한 요소</br></br>
bandwidth에 너무 민감한 이슈가 존재 -> 데이터 마이닝보단 영상 처리에 주로 사용

## **GMM(Gaussian Mixture Model)**
K-means는 거리기반 알고리즘 이기에 중심점을 기반으로 비슷한 거리적으로 퍼져있는 데이터에는 효율적이지만, 군집들이 일직선 상이 놓여 있다든지, 겹쳐있다든지 등에 대한 데이터 분포에는 어려움이 존재한다</br>
이를 해결하기 위한 군집들의 데이터들이 여러 가우시안 분포(Gaussian Distribution)를 가지는 모델로 가정하는 기법 - sklearn.mixture / GaussianMixture()</br>
모수적 추정 : 데이터가 특정 데이터 분포를 따른다는 가정하에 데이터 분포를 찾는 방법</br>
->개별 정규 분포들의 평균과 분산, 데이터가 특정 정규 분포에 해당될 확률 추정 필요</br></br>
원본 데이터 곡선이 여러 정규분포로 이루어져있다고 가정하고 데이터가 어느 정규분포에 해당하는 지 찾는 것</br></br>

**순서** </br>
step1. Expectation : 개별 데이터들 각각에 대해 정규 분포에 소속될 확률을 구하고 가장 높은 확률을 가진 정규 분포에 소속 </br>
\# 단, 최초 시에는 원본 곡선으로부터 분포를 나누지 않은 상황이므로, 임의로 특정 정규 분포로 소속 </br>
step2. Maximization : 소속된 후, Expectation에서 구한 정규 분포들의 평균과 분산을 구하고 이를 기반으로 다시 데이터가 발견될 가능도를 최대화(Maximum likelihood) 할 수 있도록 평균과 분산(모수)를 구함 </br>
step3. 개별 정규분포들의 평균과 분산이 변경되지 않고, 데이터들의 소속이 변경되지 않을 때까지 1\~2(EM) 반복</br>
(데이터들을 먼저 할당하고 계산해 변경하고 다시 할당하고 반복한다는 점에서 K-Means와 유사) </br></br>

**파라미터** : n_components = Mixture Model의 개수로 군집화의 개수를 의미</br>
predict()으로도 label 반환 가능

## **DBSCAN(Density Based Spatial Clustering of Applications with Noise)**
특정 공간 내에 데이터 밀도 차이를 기반으로 하는 알고리즘 - sklearn.cluster / DBSCAN()</br>
근처 데이터들이 적정 밀도가 유지된다면, 계속 군집을 이어나감 </br>
-> 복잡한 기하하적 분포도를 가진 데이터 세트에서도 군집화를 잘 수행 / 데이터 밀도 차이를 감지하여 자동으로 군집 생성 </br></br>
단점 : 데이터 밀도가 자주 변하거나, 모든 데이터의 밀도가 크게 변하지 않으면 성능 하락 + feature 개수가 많으면 성능 하락 </br>
\# 밀도가 이어지지 않으면 noise로 처리하기 때문 </br></br>

**구성 요소** : epsilon = 개별 데이터를 중심으로 입실론 반경을 가지는 원형의 영역 </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
min points = 개별 데이터의 입실론 주변 영역에 포함되는 최소 타 데이터의 개수 ( = 군집을 위한 적정 밀도 )</br></br>
**포인트 종류** : 핵심 포인트(Core Point) = epsilon 내에 min points 이상의 데이터를 가지고 있는 포인트 </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
이웃 포인트(Neighbor Point) = epsilon 내 위치한 다른 포인트 </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
경계 포인트(Border Point) = epsilon 내 min points보다 적은 데이터를 가졌지만, 핵심 포인트를 이웃 포인트로 가진 포인트 </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
잡음 포인트(Noise Point) = epsilon 내 min points보다 작은 데이터를 가지며, 핵심 포인트도 이웃 포인트로 가지지 않은 포인트 </br></br>

**순서** </br>
step1. eplison과 min points를 기반으로 Core Point를 하나 선택 </br>
step2. 이후 반경 내 Core Point 하나를 더 선택 </br>
step3. Core Point들을 연결한 후 이동하였을 때 그려지는 영역 내 포인트들은 모두 같은 군집으로 파악 </br>
step4. 이후 새로운 Core Point를 기준으로 2\~3 반복하며 반경 내 Core Point가 없을 때까지 영역을 확장 </br></br>

**파라미터** : eps = epsilon / min_samples = min points + 1 (자신 데이터도 포함) / metric = 'euclidean' - 거리 측정 방식 </br></br>
Noise Point는 label을 -1로 반환