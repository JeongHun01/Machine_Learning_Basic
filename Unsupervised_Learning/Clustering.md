# **Clustering Algorithm**
유사성이 높은 데이터들을 동일한 그룹으로 분류하고 서로 다른 군집들이 상이성을 가지도록 그룹화하는 알고리즘 </br>
좋은 군집일수록 뭉쳐있으며, 다른 군집과 떨어져있다(차원 축소로 확인 가능)</br></br>

**대표적인 종류들**</br>
K-Means, Mean shift, Gaussian Mixture Model, DBSCAN

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
silhouette_score(X, labels, metric = 'euclidean', sample_size = None **kwds) = 전체 데이터의 실루엣 계수를 평균해 반환 </br></br>
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

## **Gaussian Mixture**

모수적 추정 : 데이터가 특정 데이터 분포를 따른다는 가정하에 데이터 분포를 찾는 방법