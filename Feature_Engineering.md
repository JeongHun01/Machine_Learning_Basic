**주의 : 이론상 Feature Engineering을 거치면, 모델의 성능이 좋아져야 하지만 실제로 적용시 별 차이가 없거나 오히려 더 떨어지는 경우가 존재하니 유의**

### **NaN(Null) Processing**
info() / isna() -> count()로 NaN 파악후 fillna()로 NaN 데이터 치환 - pandas method

### **Drop**
Train에 불필요한 / 방해되는 feature, index 삭제 - drop(labels, axis, inplace) - pandas method

### **Replace**  
특정 비정상적인 값을 직접 다른 표준적인 수로 치환 - df[feature].replace(이상 값, 변경할 값) - pandas method

### **Outlier Removal**
데이터상 이상치를 제거 / 처리하는 과정 </br>
**단, 지울수록 원래 성능이 좋아지므로 막 지우면 안되고, 정말로 필요한 수치들만 제거하도록 한다 - 적게 지울수록 좋은 모델** </br>

**#corr** : column들 간의 상관도(상관계수)를 확인 - pandas method</br>
DataFrameObject.corr()시 상관도를 DataFrame 형태로 반환 / 비례한다면 1, 반비례한다면 -1에 가까워짐 </br>
sns.heatmap()으로 시각화 가능 - import seaborn as sns</br>
(또는 subplot 이용해서 각각 시각화 하여 확인하여도 됨) </br> </br>
**IQR(Inter Quantile Range)** : 데이터의 1/4 ~ 3/4 구간을 IQR으로 정하고 이를 기준으로 이상치 를 제외하는 기법 - numpy method</br>
최대값 = 3/4분위 + IQR* 1.5 , 최소값 = 1/4분위 - IQR* 1.5으로 정한 박스 플롯 외부 값은 모두 이상치로 처리 </br>
1. np.percentile(1d ndarray, percent) : 해당 ndarray에서 하위 percent에 해당하는 값 </br>
2. iqr = percentile(75) - percentile(25) </br>
3. 최대 = percentile(75) + iqr * 1.5 / 최소 = percentile(25) - iqr * 1.5 </br>
4. Boolean indexing이용 Outlier index 추출 .index (여기까지 하나의 함수로 묶기)</br>
5. 기존 DataFrame에서 해당 index들 row drop

### **Duplication Processing**
중복명 처리 - feature_name.groupby() -> count()로 중복 확인 후 cumcount()이용 / feature_name 파일에서 직접 수정 - pandas method

### **Feature Selection**
모델을 구성하는 주요 feature들을 선택 / 불 필요한 다수의 feature들이 모델 성능을 떨어뜨릴 가능성 제거 + 설명 가능한 모델이 될 수 있도록 feature 선별 </br>
단, LightGBM과 같은 좋은 성능의 모델은 feature들을 모두 포용 가능한 부분이 있어 상황에 따라 유의 </br>
feautre값의 분포 / Null / feature간의 높은 상관도 / 결정값(label)과의 독립성 / 모델의 Feature Importance 등 고려 </br></br>

-sklearn.feature_selection</br>
**RFE(Recursive Feature Elimination)** : 반복적으로 feature importance에서 낮은 중요도의 feature들을 제거하고 학습을 통해 다시 중요도를 평가하며 최적의 feature 추출 </br>
RFECV(estimator, step, cv, scoring) 생성 후 fit(feature, label)</br>
시간이 오래 걸리며, 메커니즘이 정확한 feature selection의 목표에 부합하지 않을 수 있음 </br></br>

**SelectFromModel** : 최초 학습 후, feature importance에 따라 평균/중앙값의 특정 비율 이상인 feature들을 선택 </br>
SelectFromModel(model, threshold) 생성 후 fit(feature, label)</br></br>

-sklearn.inspection </br>
**Permutation importance** : 학습된 모델에 대해 테스트(검증) 데이터의 feature값을 완전 변조 했을 때(shuffle 방식), 모델 성능이 얼마나 저하되는 지에 따라 중요도를 산정</br>
정해진 iteration에 따라 shuffle을 반복하여 각각의 accuracy를 구한 후 평균을 내어 원본 accuracy와의 차이로 중요도를 판별한다</br>
permutation_importance(model, feature, label, n_repeats) 객체 생성 / object.importances_mean : feature 별로 mean값을 ndarray로 반환

### **Encoding**
숫자형 이외의 자료형을 러닝을 위해 숫자형 할당하는 과정  - sklearn.processing </br></br>
**LabelEncode()** : 0부터 1씩 증가시켜 정수값 할당 </br>
객체 생성 후, fit(해당 feature) -> transform(해당 feature) = fit_transform(해당 feature), 인코딩 완료된 1d ndarray 반환</br>
classes_ - 0부터 mapping한 자료형을 ndarray 형태로 반환 / inverse_transform(숫자 집합) - 각 숫자에 대응하는 원본 데이터의 집합 반환 </br></br>
**OneHotEncoder()** : n비트 구조를 이용하여, 한 개의 자리만 1이고 나머지는 모두 0으로 서로 구분되게 할당</br>
해당 feature를 reshape(-1,1)로 2d ndarray로 변환 후 진행 </br></br>
주의점 : 이후 test는 transform만


### **Scaling**
feature들 간의 '수'의 단위 차이가 심할 때, 특정한 기준으로 통합하는 과정 - sklearn.processing</br>
기본적으로 2d ndarray가 인자로 들어감 </br></br>
**StandardScaler()** : 표준화, Standard Normal ~ N(0,1)로 변경</br>
객체 생성 후, fit(feature_df) -> transform(feature_df) = fit_transform(feature_df), 스케일링 완료된 DataFrame 반환 </br></br>

**MinMaxScaler()** : 정규화, 최소값은 0 최대값은 1로 대응되게 변경</br>

주의점 : 이후 test는 transform만

### **Conversion**
**Log Conversion** : 단일 feature 내부 데이터의 불균등이 심할 때, 이를 비교적 정규 분포와 비슷하게 변환 (음이 아닌 실수 제한) - numpy method </br>
np.log1p(feature)시 log변환 완료후 해당 자료형으로 반환 </br>
#log1p를 하는 이유 : 수치해석적으로 너무 작은 데이터로 인해 log 0 = -inf이 발생할 수 있어, 이를 방지하고자 +1를 한 상태로 계산

### **Sampling**
원본 데이터의 label이 매우 불균등한 분포를 가진다면, 학습에 어려움이 존재하기에 sampling을 통해 비율을 맞추는 작업</br>
Under Sampling : 많은 데이터를 가진 class의 세트를 적은 데이터를 가진 class수준으로 감소 </br>
Over Sampling : 적은 데이터를 가진 class의 세트를 많은 데이터를 가진 class수준으로 증식 </br></br>
**SMOTE(Synthetic Minority Over-Sampling Technique)** : 적은 데이터의 class들이 KNN-Neighbor를 하고 그 사이에 random하게 데이터를 증식 - imblearn.over_sampling </br>
smote.fit_resample(feature, label) : over된 데이터양의 feature와 label를 순서대로 반환</br>
주의점 : test set은 SMOTE 적용 금지