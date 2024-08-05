## **학습 전 행해야 하는 것들**
**1. 데이터 개요 파악** : info(), descirbe(), hist(), 시각화, . . . 등으로 데이터의 전체적인 형태 파악 </br>
**2. Feature Engineering** </br>
**3. 데이터 셋 분업화** : training set과 test set으로 분류 , sklearn.model_selection </br>
train_test_split(feature, target, test_size(%), random_state = 11 (고정값, random_seed), shuffle = True (default), stratify = None)</br>
4. **하이퍼 파라미터 튜닝** : GridSearchCV(+검증), RandomizedSearch, Bayesian Optimization, 수동 튜닝 </br>
**5. 검증** 
만들어질 training set의 알고리즘 정확도를 자세히 알기 위해 KFold를 진행(+하이퍼 파라미터 튜닝), sklearn.model_selection</br>
KFold(), StratifiedKFold() : K개의 공간으로 나누어 검증 진행 (StartifiedKFold는 label로 분포를 결정하기에 인자로 필요)</br>
cross_var_score(model, feature, target, scoring, cv) : StratifiedKFold 방식으로 cv의 결과를 순서대로 ndarray형태로 score 반환 </br>
GridsearchCV(model, para_grid, cv, scoring, refit) : scoring 기준 최적 파라미터 탐색 + 학습(refit) -> fit(feature, label)로 검증 시작</br>
\- .cv_results_ = 결과 dict 반환 / .best_params_ = 최적 파라미터 dict 반환 / .best_score_ = 최고 scoring 반환 / .best_estimator_ = 학습 모델 반환
## **학습 후 행해야하는 것들**

#### **성능 평가**
여러 모델들로 성능 평가를 한 후, 서로간의 수치가 낮거나 이상치가 있으면 혹은 feature importance를 기반으로 feature engineering, coef_파악, parameter 수정 , . . . 등을 다시 수행 후 학습/예측한다 </br>
계속 반복하며 전체적 성능 올리기 -> 최종 모델 결정