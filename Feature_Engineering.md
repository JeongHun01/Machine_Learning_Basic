**주의 : 이론상 Feature Engineering을 거치면, 모델의 성능이 좋아져야 하지만 실제로 적용시 별 차이가 없거나 오히려 더 떨어지는 경우가 존재하니 유의**

### **NaN(Null) Processing**
info() / isna() -> count()로 NaN 파악후 fillna()로 NaN 데이터 치환 - pandas method</br>
\# fillna()할때 바로 파라미터에 df.mean() 대입시, 숫자형에 한해 따로 column지정 없이 알아서 평균값 대입

### **Drop**
Train에 불필요한 / 방해되는 feature, index 삭제 - drop(labels, axis, inplace) - pandas method

### **Replace**  
특정 비정상적인 값을 직접 다른 표준적인 수로 치환 - df[feature].replace(이상 값, 변경할 값) - pandas method

### **Outlier Removal**
데이터상 이상치를 제거 / 처리하는 과정 - 잘쓰면 좋은 성능 </br>
**단, 지울수록 원래 성능이 좋아지므로 막 지우면 안되고, 정말로 필요한 수치들만 제거하도록 한다 - 적게 지울수록 좋은 모델** </br>
유의 사항 : 값이 혼자 떨어져 있어도 이상적인 데이터일 수 있으니, 여러 기능이나 시각화로 파악을 먼저 하는 것을 지향 (특정 데이터들이 이상해 보여도 테스트 데이터에 똑같이 존재할 수도 있기에 확인 필요) </br></br>
**#corr** : column들 간의 상관도(상관계수)를 확인 - pandas method</br>
DataFrameObject.corr()시 상관도를 DataFrame 형태로 반환 / 비례한다면 1, 반비례한다면 -1에 가까워짐 </br>
sns.heatmap()으로 시각화 가능 - import seaborn as sns</br>
(또는 subplot 이용해서 각각 시각화 하여 확인하여도 됨) </br> </br>
**직접 제거** : 상관도가 높거나 중요한 데이터가 있다면, 산점도와 같은 기능을 이용해 이상치 조건 구하기</br>
boolean indexing으로 조건들을 작성 후 직접 drop (내가 원하는 특정 데이터들만 골라 제거 가능) </br></br>
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
숫자형 이외의 자료형 혹은 카테고리성 feature에 러닝을 위해 정해진 숫자형을 할당하는 과정  - sklearn.processing </br></br>
**LabelEncode()** : 0부터 1씩 증가시켜 정수값 할당 </br>
객체 생성 후, fit(해당 feature) -> transform(해당 feature) = fit_transform(해당 feature), 인코딩 완료된 1d ndarray 반환</br>
classes_ - 0부터 mapping한 자료형을 ndarray 형태로 반환 / inverse_transform(숫자 집합) - 각 숫자에 대응하는 원본 데이터의 집합 반환</br>
단점 : 데이터간의 크기의 차이가 발생 </br></br>
**OneHotEncoder()** : n비트 구조를 이용하여, 한 개의 자리만 1이고 나머지는 모두 0으로 서로 구분되게 할당</br>
해당 feature를 reshape(-1,1)로 2d ndarray로 변환 후 진행 </br>
pd.get_dummies(df, columns, dummy_na)를 이용시 별도의 변환 필요 X </br>
-> 장점: Null값도 같이 인코딩 (dummy_na = False(default) - Null은 모든 자리수가 0, dummy_na = True - NaN - Column생성) + columns 없을 시 자동으로 object만 인코딩</br></br>
주의점 : 이후 test는 transform만


### **Scaling**
feature들 간의 '수'의 단위 차이가 심할 때, 특정한 기준으로 통합하는 과정 - sklearn.processing</br>
기본적으로 2d ndarray가 인자로 들어감 </br></br>
**StandardScaler()** : 표준화, Standard Normal ~ N(0,1)로 변경</br>
객체 생성 후, fit(feature_df) -> transform(feature_df) = fit_transform(feature_df), 스케일링 완료된 DataFrame 반환 </br></br>

**MinMaxScaler()** : 정규화, 최소값은 0 최대값은 1로 대응되게 변경</br>

주의점 : 이후 test는 transform만

### **Skweness**
mean(평균)과 median(중앙값)의 차이로 인한 분포 왜도 - scipy.stats</br>
**Right Skew** : mode > median > mean / **Left Skew** : mode < median< mean </br></br>
df.apply(lambda x : skew(x))시 skew수치가 series형태로 반환</br>
skew 값이 -0.5~0.5라면 대칭에 가까움 </br>
-1보다 작거나(Left Skew), 1보다 큰 경우(Right Skew) 왜도가 심함 </br></br>
Conversion을 통해 skewness 처리

### **Conversion**
단일 feature/target 내부 데이터의 불균등이 심할 때, 이를 비교적 정규 분포와 비슷하게 변환 </br>
스케일링과 동시에 진행할 시, 변환을 먼저 적용하는 것을 권장(둘다 해본 후 비교해도 됨) </br></br>
**Log Conversion** :  log를 이용해 변환- numpy method </br>
주로 Right Skew된 경우 적용</br>
np.log1p(feature)시 log변환 완료후 해당 자료형으로 반환, np.expm1()으로 원본 변환 </br>
#log1p를 하는 이유 : 수치해석적으로 너무 작은 데이터로 인해 log 0 = -inf이 발생할 수 있어, 이를 방지하고자 +1를 한 상태로 계산 </br>
음수 값이 데이터에 포함된 경우, 모두 양수가 되는 최소 값을 일괄적으로 더해서 보정 후 변환</br>


**Exponential/Power Conversion** : </br>
주로 Left Skew된 경우 적용 </br>
지수 변환 - Exponential / 거듭제곱 변환 - Power (numpy methods)</br>
Left Skew에 -를 붙여 y축 대칭이 되게 한 후, 일정 양수 값으로 모두 0 이상이 되게 하고 log conversion 진행해도 됨

### **Sampling**
원본 데이터의 label이 매우 불균등한 분포를 가진다면, 학습에 어려움이 존재하기에 sampling을 통해 비율을 맞추는 작업</br>
Under Sampling : 많은 데이터를 가진 class의 세트를 적은 데이터를 가진 class수준으로 감소 </br>
Over Sampling : 적은 데이터를 가진 class의 세트를 많은 데이터를 가진 class수준으로 증식 </br></br>
**SMOTE(Synthetic Minority Over-Sampling Technique)** : 적은 데이터의 class들이 KNN-Neighbor를 하고 그 사이에 random하게 데이터를 증식 - imblearn.over_sampling </br>
smote.fit_resample(feature, label) : over된 데이터양의 feature와 label를 순서대로 반환</br>
주의점 : test set은 SMOTE 적용 금지

### **DateTime**
날짜와 시간이 feature로 들어갈 때, 구분화 하기 - pandas method </br></br>
df_object['column'].datetime.apply(pd.to_datetime) : 해당 column이 날짜 + 시간을 나타내며 이를 datetime dtype으로 변경 </br>
df_object['date/time'] = df_object.datetime.apply(lambda x : x.year/month/day/hour) : 해당 feature column을 추가 (int형)


### Feature Engineering for Regression
회귀는 feature와 target 데이터가 모두 정규 분포인 형태를 선호 </br>
target : Log Conversion - Skewness되어 있는 경우 적용 </br>
feature : Scaling - feature들에 대해 표준화/정규화 적용</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
PolynomialFeature - 표준화/정규화 수행한 데이터 세트에 적용 (overfitting 유의)</br> 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
Log Conversion - Skewness가 심한 중요 feature들에 대해 적용

### **Text Engineering**
1\.간단한 단어나 단순 카테고리가 아닌 경우</br>
2\.카테고리로 담아두기엔 unique가 너무 많을 경우</br>
3\.어떠한 설명이라든지 중요한 정보인 경우 </br>
-> Text 전처리 및 벡터화 </br></br>
단, 벡터화시 sparse matrix형태로 반환되기에, 기존 feature matrix에 추가할 시 변환이 필요 </br>
step1. 다른 feature들을 OneHotEncode 또는 LabelEncode에서 파라미터 sparse(_output)= True (onehot default = True / label default = False)로, 희소 행렬로 반환 - dense로 합치기엔 text feature의 column수가 매우 많기에 비효율적</br>
step2. sparse matrix들은 tuple형식으로 묶어 준다 </br>
step3. scipy.sparse / hstack()을 import한 후, hstack(tuple).tocsr()로 데이터 셋들이 결합된 최종 sparse matrix를 구한다 </br>
\- 유의사항 : 합쳐진 데이터 셋은 메모리를 많이 차지하므로 사용 용도가 끝났으면 바로 메모리에서 삭제하는 것을 권장 (del data)

### **Text Parsing**
만약 데이터를 불러왔을 때 텍스트에 []나 {}가 사용 되었다면, 이는 str 형식으로 써진 것이므로 파이썬 객체로 바꿔주는 작업이 필요</br>
from ast import literal_eval</br>
이후 .apply(literal_eval)시 파이썬 객체로 바뀌며 원하는 데이터 처리 진행