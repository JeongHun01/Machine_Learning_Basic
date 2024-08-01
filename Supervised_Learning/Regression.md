# **Regression Algorithm**
회귀 분석은 데이터 값이 평균과 같은 일정한 값으로 돌아가려는 경향을 이용한 통계학 기법 </br>
여러 개의 독립변수와 한 개의 종속변수 간의 상관관계를 모델링하는 기법을 통칭</br>
머신러닝 회귀 예측의 핵심은 주어진 feature와 결정 값 데이터 기반에서 학습을 통해 회귀 계수(가중치)를 찾는 것 </br>
연속적인 결과 값을 예측

**단일 회귀 / 다중 회귀** : 독립 변수가 1개인지 다수인지 </br>
**선형 회귀 / 비선형 회귀** : 회귀 변수의 선형 유무</br></br>
대표적인 종류들</br>
**일반 선형 회귀(Linear Regression)** : 예측값과 실제값의 RSS를 최소화 할 수 있도록 회귀 계수를 최적화하며, 규제를 적용하지 않는 모델 </br>
**Ridge** : 선형 회귀에 L2 규제를 추가한 모델 </br>
**Lasso** : 선형 회귀에 L1 규제를 추가한 모델 </br>
**ElasticNet** : 선형 회귀에 L2, L1 규제를 추가한 모델 </br>
**Logistic Regression** (Regression이지만 사실 분류에 사용되는 모델)</br>
**Regression Tree** : tree 모델들을 이용해 회귀에 사용

## **RSS(Resiual Sum of Squares)**
오류 값의 제곱을 구해서 더하는 방식으로 회귀의 비용 함수(cost function), 손실 함수(loss function)이라고 불린다</br>
**RSS(W0,W1) = 1/N * Σ (y - ( w0 + w1 * x))^2** </br></br>
f(x) = W0 + W1 * x에서 회귀 계수인 W0,W1을 학습을 통해 찾는 것이 목적 </br>
학습 데이터로 입력되는 독립,종속변수는 RSS에서 모두 상수로 간주하며 여기에서 중심변수는 회귀 계수인 W이다</br>

#### **Gradient Descent(경사 하강법)**
많은 W 파라미터가 있는 경우에 사용 / 점진적으로 반복적인 계산을 통해 W 파라미터 값을 업데이트 하면서 오류 값이 최소가 되는 W 파라미터를 구하는 방식 </br></br>

비용 함수에 최초 w지점 부터 미분을 시작하였을 때, 더 이상 기울기가 감소하지 않는 지점을 바용 함수가 최소인 지점으로 보고 그때의 W를 반환한다</br>
**step1.** 비용 함수 식을 편미분시 각각 다음과 같이 나온다 </br>
RSS(W) / δW1 = -2/N * Σ x*(y - ( w0 + w1 * x)) = -2/N * Σ x*(실제값 - 예측값))</br>
RSS(W) / δW0 = -2/N * Σ (y - ( w0 + w1 * x)) = -2/N * Σ (실제값 - 예측값) </br></br>
**step2.** : 임의의 w0, w1을 생성한다 </br></br>
**step3.** 이후 아래의 식에 계속 대입하여 w를 업데이트하며 비용 함수가 최소가 되는 지점을 찾는다 (이때 편미분 값을 조절하기 위해 보정 계수인 학습률 η를 곱해서 사용)</br>
새로운 w1 = 이전 w1 - ( -η * RSS(W)/δW1 ) = 이전 w1 + η * 2/N * Σ x*(실제값 - 예측값) </br>
새로운 w0 = 이전 w0 - ( -η * RSS(W)/δW0 ) = 이전 w0 + η * 2/N * Σ (실제값 - 예측값) </br></br>
**만약 데이터가 너무 많다면 미니 배치 확률적 경사하강법을 사용한다 (전체에서 일부만 뽑아서 사용 - CLT개념 이용)**

## **성능 평가 지표**
-sklearn.metrics </br></br>
**MAE(Mean Absolute Error)** : mean_absolute_error(target, predict) - 실제값과 예측값 차이를 절댓값으로 변환해 평균한 것 </br>
#scoring = 'neg_mean_absolute_error' </br></br>
**MSE(Mean Squared Error)** : mean_squared_error(target, predict) - 실제값과 예측값 차이를 제곱해 평균한 것 </br>
#scroing = 'neg_mean_squared_error' </br></br>
**MSLE** : mean_square_log_error(target, predict) - MSE에 log를 적용한 것 (일부 큰 오류값으로 인해 전체 오류 커지는 것 방지) </br>
#scoring = 'neg_mean_squared_log_error'</br></br>
**RMSE** : MSE 파라미터 squared = False 설정 / np.sqrt() - MSE에 루트를 씌운 것 (제곱한 것을 다시 제곱근) - MAE에 비해 큰 오류값에 상대적인 패널티를 더 부여 </br>
#scoring = 'neg_root_mean_squared_error' </br></br>
**RMSLE** : MSLE 파라미터 squared = False 설정 / np.sqrt() - RMSE에 log를 적용한 것 </br></br>
**R2** : r2_score(target, predict) - R2 = 예측값 var / 실제값 var</br>
#scoring = 'r2' </br></br>

scoring 적용 시 유의 사항 : cross_val_score나 GridSearchCV는 score값이 가장 큰 것을 찾는데, 회귀 평가 지표중 대다수는 작을 수록 좋은 지표이기에 neg를 붙여 앞에 -1를 곱해준다 -> score은 -1이 붙여진 상태로 반환

## **Linear Regression**
예측값과 실제값의 RSS를 최소화하는 OLS(Ordinary Least Squares) 추정 방식으로 구현한 class - sklearn.learn_model / LinearRegression()</br>
주요 파라미터 : fit_intercept - 절편 값 계산의 유무 default = True </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
normalize - 입력 데이터 값 정규화 후 대입 여부 default = False (사용자가 직접 따로 스케일링 진행하는 것이 좋음)</br>
속성 : coef_ - fit() method로 학습이 되고나서 생성 된 회귀 계수가 저장된 곳 </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
intercept_ - intercept 값 </br></br>
**Multi-Collinearity(다중 공선성) 문제** </br>
feature 간의 상관관계가 매우 높으면 분산이 커져서 오류에 민감 </br>
일반적으로 상관관계가 높은 feature가 많은 경우 독립적인 중요 feature만 남기고 제거하거나 규제를 적용 </br>
무작정 지우기보단, 먼저 적용 후에 규제 선형이나 회귀 트리나 다른 모델들과 성능을 비교해는 것이 좋음

## **Polynomial Regression**
회귀식이 독립변수의 단항식이 아닌 다항식으로 표현되는 것을 지칭 </br></br>
PolynomialFeatures 클래스로 원본 단항 feature들을 다항 feature들로 변환한 후, 이 세트를 LinearRegression 객체에 적용하는 방식 - sklearn.processing</br>
(원래 PolynomialFeatures는 원본 피처 데이터 세트를 기반으로 degree차수에 따른 다항식을 적용하여 새로운 피처들을 생성하는 feature engineering 기법 </br>
 ex, [x1, x2] degree = 3 일시 fit(data)으로 (x1 + x2)^3 식 전개에 대응하여 transform(data)을 통해 [1, x1, x2, x1^2, x2^2, x1^3, x1^2*x2, x1*x2^2, x2^3] 생성) </br>
파라미터 - degree : 제곱수 / include_bias = 절편 유무 (default = True) 일반적으로 False일 시 성능이 좋아지는 경향 존재 </br>
**사이킷런에서는 일반적으로 Pipeline 클래스를 이용하여 PolynomialFeatures 변환과 LinearRegression 학습/예측을 결합하여 다항 회귀를 구현** </br>
다항 feature 변형 시, 회귀 계수가 늘어난다. 즉, 입력시 동일한 degree와 feature 개수의 데이터가 변환이 된 후에 들어가야지 본인 자리에 맞게 들어간다. 따라서 학습 데이터를 변형시켜 모델을 만들었다면, 이후 검증/테스트 데이터도 입력 전 변형시켜야한다. 이는 과정이 복잡하므로 pipeline을 이용해 한 번에 처리한다 </br></br>

**Pipeline** : 파라미터에 주어진 step들을 순차적으로 한 번에 실행 - sklearn.pipeline </br>
파라미터에 step 순으로 tuple로 이루어진 list 대입 - tuple('step_name', 모델_클래스/객체) -> 이후 fit(feature, target) 진행 </br>
개별 클래스 객체 접근 법 - pipeline_object.named_steps['step_name']시 해당 모델 객체 반환 </br></br>

**단점 : degree에 따라 overfitting하기가 쉽다** </br>
degree 높아지면 회귀 식이 정교해지고 이에 w0, w1, w2, . . .를 업데이트하며 식에 다가갈 때, 회귀 식의 함수 모양은 실제 함수(곡선 형태일 확률이 높은)와 유사해지지만 </br>
너무 높아지면 학습 데이터를 다 지나기 위해 회귀 계수가 커져 굉장히 난해하고 복잡한 형태의 함수 생성 </br>
반대로 degree가 너무 작아도 단순한 그래프가 나와 underfitting이 발생할 수 있어 복잡한 데이터 처리에 어려움 존재</br>
따라서 적정한 degree를 찾는 것이 중요

## **Regularized Linear Regression**
비용 함수에 alpha값으로 패널티를 부여해 과적합을 개선시키는 방식 ( 오류 최소화 + 회귀 계수 크기 제어 ) - sklearn.linear_model </br>
이를 위해, 비용 함수 목표 = Min( RSS(W) + alpha * ||W||^2(1) )로 변경 - alpha = 제어를 위한 튜닝 파라미터 </br></br>
alpha가 커지면 W가 작아져야 하기에 회귀 계수 크기를 줄이는 효과 / alpha가 작아지면 Min(RSS(W))에 가까워져 단순 RSS(W)를 최소화시켜 오류를 줄이는 효과 </br>
-> 회귀 계수 크기와 오류를 모두 줄이는 최적의 alpha 찾기 </br></br>

**규제 유형** </br>
**L2 규제** : W^2에 패널티를 부여 - Ridge, ElasticNet(계수 값 크기 조정) </br>
**L1 규제** : |W|에 패널티를 부여 - Lasso, ElasticNet(feature 개수 감소) </br></br>

#### **Ridge**
L2 규제식 이용 / alpha값을 이용하여 회귀 계수 크기를 조정- Ridge() </br></br>
비용 함수를 적게 하기 위해 미분을 할 시, RSS(W)/δW + 2alpha*W 형태가 된다 </br>
이때 선형 계수가 음수라면 비용 함수의 기울기를 늘리는 방향으로 양수라면 줄이는 방향으로 패널티를 부과하는데 </br>
경사 하강법을 진행 시, W 업데이트를 할 때 W가 클수록 큰 패널티를 부여하며 최종적으로 W가 0에 가까워지게 움직인다(0이 되진 않는다) </br>
이때 alpha값에 따라 더 큰 패널티(규제)를 부여할 수 있으며 Ridege에서는 이 alpha를 조절해 회귀 계수를 줄여 overfitting을 막는다 (너무 커질 시 underfitting 주의) </br></br>

#### **Lasso**
L1 규제식 이용 / 불필요한 회귀 계수를 0으로 만들고 제거 (feature_selection 특성) - Lasso() </br></br>
비용 함수를 미분을 할 시, RSS(W)/δW + alpha*( 1, 0 -1 중 1개)가 된다 </br>
경사 하강법을 진행 하게 되면 w 업데이트를 할 때 w를 0 또는 0에 가까워지 한다 </br>
0이 된다는 것은 모델에서 영향력이 적은 feature를 의미하며 자동적으로 제거가 되게 된다 </br>
이때 alpha값에 따라 더 큰 패널티(규제)를 부여할 수 있으며 더 많은 계수들이 0이된다 / overfitting과 underfitting을 유의해서 alpha를 결정한다 </br></br>

#### **ElasticNet**
비용 함수 = RSS(W) + L1 규제식(alpha_a) + L2 규제식(alpha_b)으로 L1 규제가 회귀 계수를 급격히 변동할 수 있어, 이를 완화하고자 L2 규제식을 추가한 것 - ElasticNet()</br></br>
비용 함수 미분시 = RSS(W)/δW + alpha_a*( 1, 0 -1 중 1개) + 2alpha_b*W 가 되며 </br>
alpha_a + alpha_b = alpha로 alpha_a 값을 조절해 L2가 L1를 완화시키는 역할을 해준다 </br></br>
파라미터 : alpha = alpha_a + alpha_b / l1_ratio = alpha_a의 비율

## **Logistic Regression**
선형 회귀 방식을 분류에 적용한 알고리즘 (주로 이진 분류) - sklearn.linear_model / LogisticRegression()</br>
회귀 최적 함수를 찾는 것이 아닌, Sigmoid 함수를 찾아 그 반환 값을 확률로 가정해 분류에 이용 </br></br>

장점 : 가볍고, 빠르며, 이진 분류 예측 성능 뛰어남. 특히 텍스트 분류에 유용 </br></br>

**Sigmoid function** = 1 / ( 1 + e^-x ), 치역 : 0\~1</br>
step1. 성공 확률 p에 대해, 실패 대비 성공 비율 함수 Odds(p) = p / ( 1 - p )로 정의하자 </br>
step2. Log 변환으로 Logit함수 생성후 선형 회귀식과 mapping. Log(Odds(p)) = W1*x + W0</br>
\#probability axioms에 의해 p의 범위는 0\~1이지만, 선형 회귀식은 -∞ \~ ∞이므로 log 변환으로 대응 </br>
step3. 이후 x에 대한 식을 구하기 위해, 역함수를 구한다 - 최종식, p(x) = 1 / ( 1 + e^-(W1x + W0)) </br>
-> 학습을 통해 Sigmoid 함수의 w를 최적화하여 예측하는 것
</br></br>
**주요 파라미터** : penalty = 규제 유형, C = 1 / alpha </br> 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
solver - lbfgs = default, 메모리 공간 절약 + CPU 코어 수가 많다면 최적화 병렬 수행 </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
liblinear = 다차원 작은 + 데이터 세트에서 효과적이지만 국소 최적화(Local Minimum) 이슈 + 병렬 최적화 불가 </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
newton-cg = 좀 더 정교화 최적화가 가능하지만, 대용량의 데이터에서 속도가 많이 저하 </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
sag = Stochastic Average Gradient로서 경사하강법 기반의 최적화 적용, 대용량 데이터에서 빠르게 최적화 </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
saga = sag와 유사한 최적화 방식이며 L1 정규화 가능 </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
추가적인 최적화 방식들은 구글링

## **Regression Tree**
CART(Classification and Regression Tree) 알고리즘 사용 - 기존 Classfier 동일 모듈 / DecisionTreeRegressor(), RandomForestRegressor(), XGBRegresor, LGBMRegresor() </br>
CART 회귀 트리는 분류와 유사하게 분할을 하며, 최종 분할이 완료된 후 각 분할 영역에 있는 데이터 결정값들의 평균값으로 학습/예측 </br>
(feature 각각 평균값 이어서 회귀 식 생성), tree 구조이기에 overfitting 유의</br>

## **예측 결과 혼합**
분류에서 Esemble기법과 유사하게(흉내만 낸 정도) 모델들의 결과에 가중치를 준 다음 더해 최종 성능을 올리는 기법 </br></br>
최종 predict = 가중치A * A모델_predict + 가중치B * B모델_predict + . . . ( ∑ 가중치 = 1) </br>
(가중치는 정해진 기준은 없으며, 여러 번 반복하여 최적 성능 탐색)

## **Stacking**
데이터 세트를 만드는 과정은 Classification과 동일, 학습만 회귀 모델로 </br>
회귀에서 stacking 기법은 다소 좋은 성능을 보임
