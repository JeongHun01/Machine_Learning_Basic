# **Classification Algorithm**
데이터의 feature와 label을 머신러닝 알고리즘으로 학습하여 모델을 생성한 뒤, 새로운 데이터 값에 대해 미지의 label을 예측하는 것 </br>
이산적인 결과 값을 예측

대표적인 종류들 </br>
**Naive Bayes** : 베이즈 통계와 생성 모델에 기반 </br>
**Logistic Regression** : 독립변수와 종속변수의 선형 관계성에 기반한 로지스틱 회귀 </br>
**Decision Tree** : 데이터 균일도에 따른 규칙 기반의 결정 트리 </br>
**Support Vector Machine** : 개별 클래스 간의 최대 분류 마진을 효과적으로 파악 </br>
**Nearest Neighbor** : 근접 거리를 기준으로 하는 최소 근접 알고리즘 </br>
**Neural Network** : 심층 연결 기반의 신경망 </br>
**Ensemble** : 서로 다른(또는 같은) 머신러닝 알고리즘을 결합한 앙상블

## **Bayesian 하이퍼 파라미터 튜닝**
**GridSearchCV** : 수행시간 많이 소요, 다수의 개별 파라미터들을 Grid형태로 지정하는 것에 한계 존재 -> 데이터 세트가 작을 때 / 파라미터 수가 적을 때 유리</br>
**RandomizedSearch** : 수행시간 줄여주지만, Random 선택에 의해 최적 파라미터 검출에는 제약 -> 데이터 세트가 클 때 유리 </br>
(위의 두 가지 방법은 모두 Iteration 중에 최적화된 파라미터를 활용하며 최적화를 진행하는 것에 어려움)</br></br>
**Bayesian Optimization** : 미지의 함수가 반환하는 값의 최소 또는 최대값을 만드는 최적해를 짧은 반복을 통해 찾아내는 최적화 방식 </br>
Step1. 최초에는 랜덤하게 하이퍼 파라미터들을 샘플링하여 성능 결과를 관측 </br>
Step2. 관측된 값을 기반으로 대체 모델은 최적 함수를 예측 추정 </br>
Step3. 획득 함수에서 다음으로 관측할 하이퍼 파라미터 추출(추천) - 이때 알고리즘(획득 함수)에 따라 추천하는 것이 다름 HyperOpt에서는 TPE사용</br>
Step4. 해당 하이퍼 파리미터로 관측된 값을 기반으로 대체 모델은 다시 최적 함수 예측 추정 </br>
-> 이를 반복하며 목표 최적 함수와 대체 모델로 예측한 최적 함수를 유사하게 만드는 것이 목적</br></br>
주요 사용하는 패키지 : HyperOpt, Bayesian Optimization, Optuna</br>
HyperOpt(TPE) : 입력값 범위(Search Space-dict) -> 목적 함수 -> fmin() - 목적 함수 최소값 유추 (최적 하이퍼 파라미터 자체를 구하는 것은 다른 모듈 - 필요시 구글링)</br></br>
hp.quniform(label, low, high, q) : label로 지정된 입력 값 변수 검색 공간을 최소값 low에서 최대값 high까지 q의 간격을 가지고 설정</br>
hp.uniform(label, low, high) : 최소값 low에서 최대값 high까지 정규 분포 형태의 검색 공간 설정</br>
hp.randint(label, upper) : 0부터 upper까지 random 정수 값으로 검색 공간 설정 </br>
hp.loguniform(label, low, high) : exp(uniform(low,high))값을 반환하며, 반환 값의 log 변환 된 값의 정규 분포 형태를 가지는 검색 공간 설정</br></br>
목적 함수는 search space를 입력 받아 로직에 따라 loss값을 계산하고 반환 , dict형태로 return {'loss' : 식 , 'status' : STATUS_OK} </br>
이때 주로 정확도를 반환</br></br>
fmin(fn = 목적함수, space = 입력값 범위, algo = 최적화 알고리즘, max_evals = 수행할 횟수, trials = Trials객체, rstate = 랜덤 시드(고정X가 좋은 성능 나오는 경향 존재))</br>
목적 함수의 최소 loss를 찾은 후 해당 하이퍼 파라미터 반환(dict) / trials - .results, .vals</br></br>
**fmin()호출 후 목적함수(모델)에서 algo의 알고리즘에 따라 search space에서 파라미터를 결정해 모델 수행 후 평가를 하며, 이를 정해진 횟수만큼 반복하여 최적값 찾기**</br>
단, 첫 번째 수행 시 예측 함수가 없기에 임의의 파라미터로 수행 -> 따라서 시드 설정이 없다면 시행 시 마다 다른 최적 파라미터 도출</br>
튜닝은 해당 모델을 기준으로 해야하므로 목적 함수는 해당 모델에 관한 식으로 작성</br>
**이후 얻은 파라미터(튜닝 완료) best[파라미터]들로 다시 모델 class에 적용시켜 객체 생성 이후 학습 진행**

## **성능 평가**
sklearn.metrics / sklearn.processing</br></br>
**정확도(accuracy)** : accurcy_score(target, predict) - 정확도 반환 (단, 이진 분류에서 사용하기엔 맹점이 존재)</br></br>
**오차 행렬(confusion matrix)** : confusion_matrix(target, predict) - TN, FP, FN, TP 반환 (2d ndarray) </br></br>
**정밀도(precision)** : precision_score(target, predict) - 정밀도 반환 = TP / (TP + FP) </br></br>
**재현율(recall)** : recall_score(target, predict) - 재현율 반환 = TP / (TP + FN) </br><br>
**F1 Score** : f1_score(target, predict) - f1 socre 반환 = 2 / (precision^-1 + recall^-1) </br></br>
**ROC AUC** : </br>
임계값에 따른 FPR과 TPR의 비율 </br>
FPR(False Positive Rate) = FP / (FP + TN) , TPR(True Positve Rate) = TP / (TP + FN) = recall </br></br>
fpr, tpr, thresholds = roc_curve(target, predict probability) - 임계값들에 대한 fpr, tpr, 임계값 반환 </br>
roc_auc_score(target, predict probability, average) - AUC 반환 </br>
average / binaray = default, micro = total값 이용, macro = 지표 계산 후 평균 이용, weighted = 가중치 + 평균</br></br>
**확률 반환 및 임계값 변경** </br>
각각의 성능 지표들의 수치를 변경하고 싶을 때, 임계값을 변경 </br></br>
model_objcet.predict_proba(feature) - train이 된 model에서의 feature를 넣었을 때 출력되는 결과들의 확률들을 반환</br>
precision, recall, threshold = precision_recall_curve(target, predict probability) - 임계값들에 대한 정밀도, 재현율, 임계값 반환 </br>
Binarizer(threshold) - 임계값이 설정된 객체 생성 -> .fit_transform(predict probability.2d ndarray형태) - 임계값 기준으로 확률들을 판별

## **Decision Tree**
데이터에 있는 규칙을 학습을 통해 자동으로 찾아내 트리 기반의 분류 규칙을 생성(if-else 기반) - sklearn.tree / DeicisionTreeCLassfier() </br>
장점 : 쉽고 직관적 + 데이터 전처리 영향이 크지 않다 / 단점: 과적합으로 알고리즘 성능 저하 -> 트리의 크기를 사전에 제한하는 튜닝 필요</br>
#### **정보 균일도 측정 방법**
**정보 이득(information gain)** : 엔트로피 기반(데이터 집합의 혼잡도) / = 1 - 엔트로피 -> 정보 이득이 높은 속성을 기준으로 분할 </br>
**지니 계수(gini)** : 불평등 지수(높을수록 불평등) / 지니 계수가 낮은 속성을 기준으로 분할 </br></br>

#### **Decision Tree 주요 하이퍼 파라미터**
문자열 : 리스트의 딕셔너리 형태로 
모델 객체 생성 때 혹은 method에 적용</br>
max_depth : tree의 최대 깊이 규정 / default = None ->  완벽한 class 결정값 혹은 노드가 가지는 samples 개수가 min_samples_split보다 작아질 때까지 깊이를 증가</br>
max_features : 최적의 분할을 고려할 최대 feature 개수(%) / default = None</br>
#'sqrt' - 전체^(1/2) / 'auto' - 'sqrt' / 'log' - log2(전체) </br>
min_samples_split : 노드가 분할 가능한 최소의 samples 데이터 수 제한 / default = 2</br>
min_samples_leaf : 분할의 결과인 왼쪽, 오른쪽 자식 노드에서 가져야할 최소한의 sample 데이터 수 제한</br>
max_leaf_nodes : 밑단 노드의 최대 개수</br></br>

#### **Overfitting**
모든 케이스를 분류하기 위해 복잡한 기준들이 설정된다면, 복잡한 경계들로 이루어진 model이 형성 </br>
그렇게 된다면, 유연성이 떨어지기에 test data를 제대로 예측하지 못할 가능성이 커져 성능이 떨어지며, 이를 과적합이라고 호칭 </br>
general한 정도가 너무 커지면 underfitting이 발생하기에 적절한 하이퍼 파라미터 혹은 여러 기법으로 overfitting을 막는 것이 중요

## **Ensemble**
여러 개의 분류기(Classifier)를 생성하고 그 예측을 결합함으로써 보다 정확한 최종 예측을 도출하는 기법 -> 단일 모델의 약점을 다수 모델의 결합으로 보완</br>
비슷한 모델이나 성능 우선보단 다른 유형의 모델을 섞는 것이 전체 성능에 도움이 될 수 있음 - sklearn.ensemble</br></br>

### **Voting**
같은 데이터 set에 대해 서로 다른 알고리즘 분류기를 이용 - VotingClassfier()</br></br>
**Hard Voting** : 다수의 Classifier간 다수결 투표로 최종 class 결정 </br>
**Soft Voting** : 다수의 Classifier간 class 확률을 평균하여 결정 최종 class 결정

### **Bagging**
=bootstrap aggragating &nbsp;/ &nbsp;서로 다른 데이터 set에 대해 같은 알고리즘 분류기를 이용</br></br>

**Random Forest** : 전체 데이터에서 각각 분할 및 샘플링해 개별적으로 학습한 후 모든 분류기가 voting을 통해 class를 결정-RandomForestClassifier() </br>
부트스트래핑 분할 방식 : 전체 데이터에서 일부가 중첩되게 샘플링하는 방식 </br>
하이퍼 파라미터 : n_estimators(결정 트리 개수 지정, default = 100), max_features(default = 'auto'), max_depth, min_samples_leaf, . . .굉장히 </br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;많으니 필요시 구글링</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
ex. 전체 데이터 : 16개, 각각의 데어터 set에 임의의 4개를 가져가 중복하여 sampling </br>

### **Boosting**
여러 개의 약한 학습기(weak learner)를 순차적으로 학습-예측하여 잘못 예측한 데이터나 학습 트리에 가중치 부여를 통해 오류를 개선해나가는 방식 </br></br>
**AdaBoost** : 분류 기준에 벗어난 데이터에 가중치를 부여하여 새로운 분류 기준을 만들며, 만들어지는 분류 기준과 기존의 것을 종합하여 예측 후 반복 </br></br>
**Gradient Boost** : AdaBoost와 비슷한 논리이지만 경사 하강법을 이용해 오류식을 최소화 하는 방향으로 가중치를 업데이트하는 형식 </br>
GBM 하이퍼 파라미터 : loss(경사 하강법 비율), learning_rate(학습률, default = 0.1), n_estimators, subsample(학습에 사용할 샘플링 비율) </br>
-> 문제점 : 학습에 많은 시간이 소요 + 고차원 데이터 다루기에 어려움 -> 보완하고자 나온 모델 : XGBoost, LightGBM</br></br>
**XGBoost** : GBM에서 병렬처리를 지원 - xgboost / xgb()</br>
장점 - 분류,회귀 모두 좋은 성능 / GBM 대비 빠른 속도(CPU 병렬처리 + GPU 지원) / 다양한 성능 향상 기능(규제, Tree Pruning) / 다양한 편의 기능(Early Stopping, . . .)</br></br>
**-XGBoost, LightGBM 실전에서는 사이킷런 wrapper를 사용하지만, 이해를 위해 파이썬 wrapper도 아는 것이 중요-**</br>
C/C++로 작성 된 모듈 -> 파이썬 Wrapper(xgb) -> 사이킷런 Wrappper(XGBClassifier) </br>
파이썬 Wrapper : 데이터 세트 - xgb.DMatrix(feature, target) </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
학습 API - 학습되어 반환되는 객체 = xgb.train() (모델 객체를 만들고 학습하는 것이 아니라, 학습된 모델 객체를 만듦)</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
예측 API - 예측 결과를 추정하는 확률(=predict_proba) = 학습된객체.predict() </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
feature 중요도 시각화 - plot_importance </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
하이퍼 파라미터 - eta(=learning_rate), num_boost_rounds(=n_estimators), min_child_weight, max_depth</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
, sub_sample(=subsample), lambda(=reg_lambda), alpha(=reg_alpha), colsample_bytree, scale_pos_weight, gamma, </br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;()는 GBM에 동일한 기능이 존재하는 파라미터</br>
사이킷런 Wrapper : 기존 다른 모델과 동일하게 사용 가능 (객체 생성 : 하이퍼 파라미터 / fit : early stopping 세팅)</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
하이퍼 파라미터 - 기존 GBM을 그대로 따르되, GBM에 없는 것은 파이썬 Wrapper의 파라미터 사용</br></br>
학습 파라미터</br>
objective : 회귀인지, 어떠한 분류인지 학습 방법 설정 </br>
eval_metirx : 오류함수의 성능 평가 지표 </br></br>
조기 중단 기능(Early Stopping) : 과적합 방지를 위해 특정 반복 횟수 만큼 더 이상 비용함수가 감소하지 않으면 지정된 반복 횟수를 완료하지 않고 수행 종료 </br>
early_stopping_rounds - 더 이상 비용 평가 지표가 감소하지 않는 최대 반복 횟수 </br>
eval_metric - 반복 수행 시 사용하는 평가 지표 </br>
eval_set - 평가를 수행하는 별도의 검증 데이터 세트. 일반적으로 검증 데이터 세트에서 반복적으로 비용 감소 성능 평가 </br></br>
**LightGBM** : XGBoost의 단점들을 보완한 모델 - 코드 진행 논리는 동일하며 파리미터가 다르다(num_leavs 중심) - lightgbm / LGBMClassifier()</br>
장점 - XGBoost보다 빠른 학습과 예측 수행시간 / 더 적은 메모리 / 카테고리형 feature의 자동 변환과 최적 분할</br></br>
기존 GBM, XGBOOST 분할법 : 균형 트리 분할(Level Wise) vs LightGBM : 리프 중심 트리 분할(Leaf Wise) </br></br>
C/C++로 작성 된 모듈 -> 파이썬 Wrapper(xgb) -> 사이킷런 Wrappper(XGBClassifier) </br>
파이썬 Wrapper : 
하이퍼 파라미터 - num_iterations(=n_estimators), learning_rate, max_depth, min_data_in_leaf(=min_child_samples), num_leaves  </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
, bagging_fraction(=subsample), feature_fraction(=colsample_bytree), early_stopping_round(=early_stopping_rounds)</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
, lambda_l2(=reg_lambda), lambda_l1(=reg_alpha), min_sum_hessian_in_leaf(=miㅜ_child_weight)</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
()는 XGBoost에 동일한 기능이 존재하는 파라미터</br>
사이킷런 Wrapper : 기존 다른 모델과 동일하게 사용 가능</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
하이퍼 파라미터 - XGBoost를 그대로 따르되, XGBoost에 없는 것은 파이썬 Wrapper LightGBM의 파라미터 사용</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
Leaf Wise 방식으로 주로 num_leaves를 중심으로 min_child_samples, max_depth 조정</br></br>
boost_from_average = False : label값이 극도로 불균등할 때, 재현율 및 ROC-AUC 성능을 높이기 위함 (default = True)</br>
너무 많은 파라미터 튜닝은 오히려 방해가 될 수 있음 </br>
num_leaves, max_depth, min_child_samples, min_child_depth / subsample, colsample_bytree / reg_lambda, reg_alpha / learning_rate -> 대표 파라미터들</br>
XGBoost, LightGBM 모두 정확한 정확도 추출을 위해선 검증데이터를 KFold로 번갈아 가며 확인 필요 (early stopping을 적용 시키려면 직접 코드 작성 / cross함수 X)</br>

### **Stacking**
같은 데이터 set를 여러 기반 모델들이 학습한 후 예측한 데이터를, 다시 메타 모델이 이를 학습 후 예측 </br>
현실 모델에 적용 가능한가? 무조건적인 성능 향상을 보이는가?에 대한 의견들이 분분함 </br>
사용한다면 original형태 보단(overfitting 문제), 교차 검증 기반 형태로 이용 - 다만 많이 복잡</br></br>
**CV 셋 기반 Stacking** - 직접 쌓는 과정을 그려가며 설계해야 직관적 (회귀에서도 사용 가능)</br>
step1. 각 모델별로 교차검증으로 원본 학습/테스트 데이터를 예측한 결과 값을 기반으로 메타 모델을 위한 학습용/테스트용 데이터를 생성 </br>
#교차 학습 후 나온 모델 - 검증 데이터 예측값->학습 데이터(세로로 쌓기) / 테스트 데이터 예측값 -> 테스트 데이터(가로로 쌓기 + 교차 완료 후 평균내기) </br>
step2. 위에서 생성한 학습용/테스트용 데이터들을 stacking 형태로 합치어 최종 학습용/테스트용 데이터 생성</br>
step3. 메타 모델은 최종 학습 데이터와 원본 학습 label를 기반으로 학습한 뒤, 최종 테스트 데이터를 예측하고 원본 테스트 label를 기반으로 평가 </br>
#학습 - 최종학습+원본학습label / 예측 - 최종테스트 / 평가 - 원본테스트label

## **Feature Importance**
모든 Tree모델에서 범용적으로 사용 </br>
trained_model.feature_importances_ - 1d ndarray형으로 중요도 순서대로 반환 </br>
zip(feature name, feature importance) - 순서대로 feature 이름과 중요도를 tuple 형식으로 mapping 후 1d ndarray화 </br>
sns.barplot(x = feature importance, y = feature name) - 시각화 </br></br>

단, tree구조를 만들기 위한 feature들의 impurity가 중요 기준(label값과 관련이 없어도 높은 중요도 가질 수 있음) / 학습 데이터 기반이므로 테스트에서는 또 다름 / 숫자형의 높은 cardinality feature에 biased 되어있음 -> 따라서 feature importance는 절대적인 feature selection 기준이 될 순 없음</br>
따라서 모델 자체에 영향을 주는 중요도를 좀 더 정밀하게 알기위해 permutation importance가 도입
