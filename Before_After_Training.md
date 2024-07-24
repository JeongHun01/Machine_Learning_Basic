## **학습 전 행해야 하는 것들**
**1. 데이터 개요 파악** : info(), descirbe(), display(), . . . 등으로 데이터의 전체적인 형태 파악 </br>
**2. Feature Engineering** </br>
**3. 데이터 셋 분업화** : training set과 test set으로 분류 , sklearn.model_selection </br>
train_test_split(feature, target, test_size(%), random_state = 11 (고정값, random_seed), shuffle = True (default), stratify = None)</br>
4. 하이퍼 파라미터 튜닝 : GridSearchCV, RandomizedSearch, Bayesian Optimization, 수동 튜닝 </br>
## **학습 후 행해야하는 것들**
#### **검증** 
만들어질 training set의 알고리즘 정확도를 자세히 알기 위해 KFold를 진행(+하이퍼 파라미터 튜닝), sklearn.model_selection</br>
KFold(), StratifiedKFold() : K개의 공간으로 나누어 검증 진행 </br>
cross_var_score(model, feature, target, scoring, cv) : StratifiedKFold 방식으로 cv의 결과를 순서대로 ndarray형태로 score 반환 </br>
#### **성능 평가**