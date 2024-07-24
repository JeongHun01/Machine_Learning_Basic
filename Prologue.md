1. Supervised Learning vs Unsupervised Learning

Supervised : 답이 정해진 데이터 분석을 토대로 새로운 데이터가 들어오면 어느 종류인지 판단
            ex. 개와 고양이 사진 학습 -> 고양이 사진 제시 -> 고양이로 판단
            Problem : Classification / Regression

Unsupervised : 답이 존재하지 않는 데이터를 가지고 어떠한 기준을 가지고 데이터를 분류 후 새로운 데이터가 들어오면 이 기준에 따라 판단
            ex. 데이터들을 보고 비슷한 동물들 끼리 모음 -> 새로운 동물 제시 -> 기존에 세운 분류에 따라 판단
            Problem : Clustering / Network analysis / Anomaly detection

가장 큰 차이 : 데이터의 정답 유무 -> 사용하는 알고리즘에 차이가 발생

2. Data Processing

Training(Learning) : Data들을 토대로 알고리즘을 만드는 과정

Dataset = Traning data + Test data
 Traning data = Traning data set + Validation data set (여러 구간으로 나눠질 수 있음)
 1) Traning data set -> Error를 최소화 하는 방향으로 Parameter 결정 -> Objective function
 2) Vadlidation data set -> Test처럼 이용하되, Error가 너무 크다면 Training data set으로 돌아가 Parameter 수정
 3) Test data set -> 만들어진 모델을 평가

목적 : in sample error와 out of error가 모두 작아야함 -> 적절한 복잡도를 가진 알고리즘 모델이 필요

3. R vs Python

R : 통계 전용 프로그램 언어로 다양하고 많은 통계 패키지를 보유
Python : 개발 언어로 다양한 라이브러리 활용 가능 - 뛰어난 확장성, 연계, 호환성, 딥러닝 프레임워크 대부분이 파이썬 기준

    파이썬 패키지들
    1. scikit-learn : 머신러닝 패키지
    2. Numpy / Scipy : 배열, 선형대수, 통계 패키지   (주로 다차원)
    3. Pandas : 데이터 핸들링                       (주로 2차원)
    4. matplotlib / Seaborn : 시각화
    5. Juypter : 대화형 파이썬 툴

4. Overfitting / Underfitting

Overfitting - 과대적합 : 분산이 높아지고 편향이 낮아지는 경향
Underfitting - 과소적합 : 분산이 낮아지고 편향이 높아지는 경향

둘다 total error를 증가