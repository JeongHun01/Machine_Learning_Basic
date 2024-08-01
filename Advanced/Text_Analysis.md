# **Text Analysis**
**NLP(Natural Language Processing)** : 인간의 언어를 이해하고 해석하는데 더 중점을 두고 기술이 발전해옴 </br>
**텍스트 분석** : 머신러닝, 언이 이해, 통계 등을 활용해 모델을 수립하고 정보를 추출해 분석 작업을 주로 수행</br>
\# 딥러닝의 LLM이 발전하며 둘 사이의 경계는 약해지고 함께 연구되며 활용</br></br>

### 주요 영역
**Text Classification** : 문서가 특정 분류 또는 카테고리에 속하는 것을 예측하는 기법</br>
**Sentiment Analysis** : 텍스트에서 나타나는 감정/판단/믿음/의견/기분 등의 주관적인 요소를 분석하는 기법</br>
**Summarization** : 텍스트 내에서 중요한 주제나 중심 사상을 추출하는 기법</br>
**텍스트 군집화 및 유사도 측정** : 비슷한 유형의 문서에 대해 군집화를 수행 하는 / 유사도를 측정해 비슷한 문서끼리 모으는 기법 </br>

### 수행 과정
1\. Text 문서 데이터 사전 가공 </br>
2\. Feature Vectorization </br> 
3\. Feature 기반의 데이터 set 제공 후 ML 학습/예측/평가 </br>
### 패키지
1\. NLTK(National Language Toolkit for Python) : 대표적인 NLP 패키지로 방대한 데이터 세트와 모듈 존재, 그러나 수행 속도가 다소 느려 대용량 데이터에 부적합 - 딥러닝 등장 후, 사용 저조</br>
2\. Gensim : Topic Modeling에 가장 많이 사용, Word2vec 구현 등 다양한 기능 존재 </br>
3\. SpaCy : 뛰어난 수행 성능의 NLP 패키지

## **Text Preprocessing**
기본적으로 class/method는 패키지마다 다르니 구글링 후 사용 </br></br>
**클렌징(Cleansing)** : 텍스트에서 분석에 방해가 되는 불필요한 문자, 기호 등을 사전에 제거하는 작업, ex. HTML, XML 태그 혹은 기호 </br></br>
**토큰화(Tokenization)** : 문장 토큰화, 단어 토큰화 - list type으로 반환</br>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; 
n - gram = 문맥적인 의미를 보존하기 위해 연속된 n개의 단어를 하나의 토큰화 단위로 분리</br>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp;
vectorization class에서 동시에 처리 가능</br></br>
**Filtering / Remove StopWords / Modify Spelling** : 불필요한 단어나 분석에 필요없는 단어(a, the, is, will 등) 그리고 잘못된 철자 수정 </br></br>
**Stemming / Lemmatization** : 어근 추출 (Lemmatization이 품사를 파라미터로 가져, Stemming보다 정교하고 의미론적인 단어 원형을 찾아줌)

## **Featrue Vectorization**
### **Bag of Words**  
문서의 모든 단어들을 문맥이나 순서를 무시하고 feature로 만든 다음 단어가 의미하는 것에 빈도 관련 특정 기반으로 정수 값을 부여 </br>- 주로 count / 정규화 변환 횟수</br>
->Document Term Matrix : 부여한 값들을 행렬로 표현 - m개의 문서(문장) x n개의 단어 = m x n matrix</br></br>
**순서** </br>
step1. 모든 문장에 있는 단어들에서 중복을 제거하고 단어를 column명으로 나열</br>
step2. 각 단어에 고유한 index(위치) 부여 </br>
step3. 문장에서 해당 단어가 나타타는 횟수(Occurrence)를 행렬 상 해당하는 위치에 기재 </br></br>
장점 : 쉽고 빠른 구축 + 문서의 특징을 잘 나타냄</br>
단점 : 문맥 의미 반영 문제 + 희소 행렬 문제
#### **피쳐 벡터화 유형**
**단순 카운트 기반 벡터화** : 단어 feature에 이 단어가 나타는 횟수를 값으로 Count를 부여하는 경우 - 값이 높을 수록 중요한 단어로 인식 </br>
\- sklearn.feature_extraction.text / CountVectorizer() </br>
문제점 - 언어의 특성상 자주 사용될 수 밖에 없는 단어도 높은 값 부여 </br></br>
**TF-IDF 벡터화** : 개별 문서에서 자주 나타나는 단어들에 가중치를 주되, 모든 문서에서 전반적으로 자주 나타나면 패널티를 부과</br>
-주로 문서의 크기가 크고 많을 때 사용, sklearn.feature_extraction.text / TfidfVectorizer() </br>
TF(Term Frequency) = 해당 문서에서 해당 단어가 얼마나 나왔는지를 나타내는 지표 </br>
DF(Document Frequency) = 해당 단어가 몇 개의 문서에서 나왔는지를 나타내는 지표 </br>
IDF(Inverse Document Frequency) = DF의 역수로서 (전체 문서수 / DF)</br>
-> TF-IDF = TF * log IDF (DF가 높다면 패널티를 부과) </br></br>
fit(), transform()으로 적용 희소 행렬 반환 </br></br>

**파라미터** : make_df = 일정 이상의 높은 빈도의 단어는 문법적인 특성의 반복적인 단어라고 판단하고 제거하기 위해 상한 설정 </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
min_df = 일정 이하의 낮은 빈도의 단어는 중요도가 낮거나 garbage성 단어라고 판단하고 제거하기 위해 하한 설정 </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
\- int형 ~ 문서 개수 / 부동소수형 ~ 상위(하위)% </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
max_features = 최대 추출하는 상위 feature 개수 </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
stop_words = stop word로 지정할 단어 입력 </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
ngram_range = tuple 형태 (범위 최소값, 범위 최대값)으로 n-gram 범위 설정 (범위내 모든 n값으로 n-gram 실행후 포함) </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
analyzer = feature 추출을 수행할 단위 (default = 'word')</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
token_pattern = 토큰화를 수행하는 정규 표현식 지정 (default = '\b\w\w+\b') </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
\#어근 추출 시 외부 함수 사용할 경우 해당 함수를 인자로 사용</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
lower_case = 모든 문자를 소문자로 변경하는지 (default = True) </br></br>
**속성** : .vocabulary_ = 각각의 단어와 index를 mapping한 것을 dict 형식으로 반환

#### **CSR Matrix(희소 행렬)**
BOW의 행렬처럼 너무 많은 0값이 메모리 공간에 할당 되어 있으면, 많은 저장 공간과 연산 시에 많은 시간이 소모 되므로 필요한 데이터만 사용 </br>
\- scipy / sparse</br></br>
**COO 형식** : coordinate 방식이며, 0이 아닌 데이터만 별도의 배열에 저장하고 그 데이터를 가리키는 행과 열의 위치를 배열로 지정 - coo_matrix()</br>
**CSR 형식** : coo형식의 좌표값 중복 문제를 해결한 형식(행과 열의 좌표 값에 또 다시 index를 부여하는 방식) - csr_matrix()




### **Word Embedding(Word2Vec)** 
개별 단어를 문맥을 가지는 N차원 공간에 벡터화 시키는 것 (딥러닝에 주로 사용)</br></br>