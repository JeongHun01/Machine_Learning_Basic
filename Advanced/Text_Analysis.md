# **Text Analysis**
**NLP(Natural Language Processing)** : 인간의 언어를 이해하고 해석하는데 더 중점을 두고 기술이 발전해옴 </br>
**텍스트 분석** : 머신러닝, 언이 이해, 통계 등을 활용해 모델을 수립하고 정보를 추출해 분석 작업을 주로 수행</br>
\# 딥러닝의 LLM이 발전하며 둘 사이의 경계는 약해지고 함께 연구되며 활용</br></br>

### 주요 영역
**Text Classification** : 문서가 특정 분류 또는 카테고리에 속하는 것을 예측하는 기법</br>
**Sentiment Analysis** : 텍스트에서 나타나는 감정/판단/믿음/의견/기분 등의 주관적인 요소를 분석하는 기법</br>
**Summarization** : 텍스트 내에서 중요한 주제나 중심 사상을 추출하는 기법 - Topic Modeling</br>
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
기본적으로 class/method는 패키지마다 다르니 구글링 후 사용 </br>
(정규 표현식 익히는 것이 좋음) </br></br>
**클렌징(Cleansing)** : 텍스트에서 분석에 방해가 되는 불필요한 문자, 기호 등을 사전에 제거하는 작업, ex. HTML, XML 태그 혹은 기호 </br></br>
**토큰화(Tokenization)** : 문장 토큰화, 단어 토큰화 - list type으로 반환</br>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; 
n - gram = 문맥적인 의미를 보존하기 위해 연속된 n개의 단어를 하나의 토큰화 단위로 분리</br>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp;
vectorization class에서 동시에 처리 가능</br></br>
**Filtering / Remove StopWords / Modify Spelling** : 불필요한 단어나 분석에 필요없는 단어(a, the, is, will 등) 그리고 잘못된 철자 수정 </br>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp;
vectorization class에서 동시에 처리 가능</br></br>
**Stemming / Lemmatization** : 어근 추출 (Lemmatization이 품사를 파라미터로 가져, Stemming보다 정교하고 의미론적인 단어 원형을 찾아줌) </br>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp;&nbsp; &nbsp; &nbsp; &nbsp;
벡터화 파라미터 중 tokenizer에 함수 형식으로 넣어 어근 추출 동시에 가능


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
fit(), transform()으로 적용 및 희소 행렬 반환 </br></br>

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

## **Text Classification**
문서를 카테고리나 종류에 맞게 분류 - 지도학습

 step1. 데이터 전처리 </br>
 step2. 벡터화 적용 </br>
 step3. 분류 모델(지도 학습) 적용 및 예측 </br></br>
 pipeline 이용시 더 간단히 이용 가능

 ## **Sentiment Analysis**
 주어진 text의 주관적인 성격 분석 - 지도 학습 / 감성 어휘 사전(비지도 학습) </br></br>
 **지도학습 기반** : 데이터와 레이블을 기반으로 학습한 후 다른 데이터의 결과를 예측</br>
step1. 데이터 전처리 </br>
step2. 벡터화 </br>
step3. 분류 모델 적용 및 예측 </br></br> 
 **감성 어휘 사전 기반(비지도 학습)** : 감성 분석을 위한 용어와 문맥에 대한 정보를 가진 사전을 이용해 감성 수치를 계산하고 긍정/부정 판단 </br>
 지도 학습에 비해 성능이 떨어지기에 class값이 없을 때 주로 사용 </br>
 - SentiWordNet : Synset 별로 3가지 감정 점수를 할당(주관적-(긍정, 부정) , 객관성) / 단어들의 감정 부정 감성지수 합산하여 최종 결정 
 - VADER : 주로 소셜 미디어의 텍스트를 위한 패키지. 뛰어난 감성 분석 결과 제공하며, 빠른 수행으로 대용량에 적합
 - Pattern : 예측 성능 측면 뛰어남. 파이썬 2.X버전에서만 동작</br>

#### **SentiWordNet**
Synset 객체가 긍정 지수, 부정 지수, 객관성 지수를 분석하여 저장함 </br>
nltk.corpus / sentimentwordnet() - 과정이 좀 복잡하니 필요시 구글링 필요 </br></br>
step1. 문서를 문장 단위로 분해 </br>
step2. 다시 문장을 단어 단위로 토큰화하고 품사 태깅(POS) </br>
step3. 품사 태깅된 단어 기반으로 synset 객체와 senti_synset 객체를 생성 </br>
step4. senti_synset에서 긍정/부정 감성 지수를 구하고 이를 모두 합산해 특정 임계치를 기준으로 긍정, 부정 결정 

#### **VADER**
소셜 미디어의 감성 분석 용도로 만들어진 룰 기반의 Lexicon - nltk.sentiment.vader / SentimentIntensityAnalyzer()</br></br>
step1. SentimentIntensityAnalyzer() 클래스로 객체를 생성   </br>
step2. polarity_scores() method로 감성 점수 dict 반환 </br>
step3. 직접 임계치 기준 compound 값으로 긍정 부정 결정 </br></br>
'neg' = 부정 감정 지수 / 'neu' = 중립 감성 지수 / 'pos' = 긍정 감성 지수 / compound = 감성 지수들을 조합해 -1\~1 사이의 값으로 표현한 것 </br>
compound 값이 최종 감성 여부 결정 -> 임계값 조절으로 예측 성능 조절 (일반적으론 0.1 기준 높으면 긍정 처리)

## **Topic Modeling**
문서들에 잠재되어 있는 공통된 토픽(주제)들을 추출하는 기법 </br>
이는 유사성 도출과 더불어 문서들이 가지는 주요 토픽의 분포도와 개별 토픽이 어떤 의미인지를 제공하는 특징 보유 </br></br>
모델을 통해 토픽들을 나눔 + 개별 토픽들의 단어 분포를 얻음 -> 결과를 보고 직접 우리가 토픽들에 적정한 이름을 부여 </br></br>
**유형**</br>
**LSA(Latent Semantic Analysis), NMF(Non Negative Factorization)** : 행렬 분해 기반 (SVD, NMF)</br>
**pLSA, LDA(Latent Drichlet Allocation)** : 확률 기반</br></br>
1\. 개별 문서는 혼합된 여러 개의 주제로 구성되어 있다 </br>
2\. 개별 주제는 여러 개의 단어로 구성되어 있다 </br>
위 2가지의 가정이 전제되어 있어야 함
</br></br>

### LDA
관찰된 문서 내 단어들을 이용하여 베이즈 추론을 통해 잠재된 문서 내 토픽 분포와 토픽별 단어 분포를 추론하는 방식</br>
이때, 베이즈 추론의 사전 확률 분포로 사용되는 것은 디리클레 분포(Dirichlet Dirstribution) </br>
베이즈 추론 : 초기에 사전 확률을 구한 후, 이후 데이터가 관측되면 사후 확률을 계산하고 이를 사전확률로 두고 다시 반복하여 업데이트 하는 방식 </br>
\- sklearn.decomposition / LatentDirochletAllocation()</br></br>
**내부 진행 순서**</br>
step0. 벡터화 - 단순 Count기반 문서-단어 행렬 생성 (확률 기반이기에 TF-IDF불가)</br>
step1. 토픽의 개수를 먼저 설정 </br>
step2. 각 단어들을 임의의 주제로 최초로 할당한 후 문서별 토픽 분포와 토픽별 단어 분포가 결정</br>
step3. 특정 단어 하나를 추출하고 해당 단어를 제외한 문서의 토픽 분포와 토픽별 단어 분포를 재계산 / 추출된 단어는 새롭게 토픽 할당 분포 계산 </br>
step4. 다른 단어를 추출하고 step3 다시 수행, 모든 단어들의 토픽 할당 분포 재 계산 </br>
step5. 지정된 반복 횟수 만큼 또는 모든 단어들의 토픽 할당 분포가 변경되지 않고 수렴될 때까지 3\~4를 수행 </br></br>

단점 : 추출된 토픽은 다시 사람의 주관적인 해석 필요 + 초기화 파라미터 및 Document-Term 행렬의 단어 필터링 최적화 어려움 존재 </br></br>

**파라미터** : n_components = 토픽의 개수 </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
doc_topic_prior = α (문서의 토픽 분포 θ의 초기 값) </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
topic_word_prior = β (토픽의 단어 분포 φ의 초기 값)</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
max_iter = 반복 횟수</br></br>
**속성** : components_ = topic별로 개별 단어들의 연관도(횟수)를 정규화한 수 2d ndarray ( shape - topic 수 X feature 단어 수)</br></br>
transform()시 개별 문서별 토픽 분포 반환 -> 직접 분포도에 비율을 보고 어떠한 토픽들로 구성되어 있는 지 판단

## **Document Clustering**
Text Classification과 유사하지만 비지도 학습으로, 비슷한 텍스트 구성의 문서를 군집화 하는 기법 </br></br>
 step1. 데이터 전처리 </br>
 step2. 벡터화 적용 </br>
 step3. 군집화 모델(비지도 학습) 적용 및 예측 </br></br>
 주로 K-Means 모델을 많이 사용</br></br>

 **군집별 핵심 단어 추출** </br>
 모델.cluster_centers_ : 각 cluster에 해당하는 feature 단어들의 비율 반환 (shape - cluster개수 X feature 단어 수) </br>
 centroid에 가까울 수록 값이 1에 가까워짐 </br>
 -> 이를 보고 직접 중요 단어 판단 및 추출

## **Document Similarity**
문서와 문서간의 유사도를 측정 ( ex\. 뉴스 기사 아래에 관련된 기사 목록 표시) </br></br>
**문서 유사도 측정 지표** : Cosine Similarity, Jaccard Similarity, Manhattan Distance, Euclidean Distance 

### **Cosine Similarity**
cos(0) = 1, cos(𝝿/2) = 0 을 이용해 방향성을 기준으로 유사도를 측정 - sklearn.metrics.pairwise / cosine_similarity()</br>
1\. 벡터화 한 행렬 상 각각의 Document들은 feature들로 이루어져 있음</br>
2\. 이를 각각을 벡터로 보고 문서간의 cos값으로 유사도를 측정 </br>
3\. 0\~1까지의 값을 가지며 1에 가까울 수록 높은 유사도를 가짐 (벡터화 행렬은 음수가 존재 안함)</br>
-> simliarity = cosθ = A·B / ||A||||B|| = ∑ AB / (∑ A^2)^(1/2) * (∑ B^2)^(1/2) </br></br>
cosine_similarity(X,X)시 2d ndarray로 D0~DN에 관한 서로간의 유사도를 나타낸 행렬 반환 </br>
(1행 지정,X)시 지정된 행 기준으로 다른 행들과의 비교값을 행렬로 반환

## **한글 NLP**
띄워 쓰기, 다양한 조사, 주어/목적어 생략 가능, 의성어/의태어, 높임말, . . . 등의 이유로 한글은 NLP를 어렵게 만드는 요인이 많다</br>
형태소 분석 : 말뭉치를 형태소 어근 단위로 쪼개고 각 형태소에 품사 태깅(POS tagging)을 부착하는 작업을 지칭
### **KoNLPy**
C/C++, Java로 만들어진 한글 형태소 엔진을 파이선 Wrapper 기반으로 재작성된 패키지 - konlpy.tag / 사용할class이름()</br>
품사 태깅 클래스 : Kkma, Komoran, Hannanum, Okt, Mecab(리눅스 전용) </br>
pandas에서 한글 문서 불러올 시, 인코딩을 cp949로 설정 / text파일은 UTF-8 </br></br>
**순서**</br>
step1. (class.morphs(text) : 인자로 들어온 text를 형태소 단어로 토큰화하여 list로 반환)의 함수를 설정</br>
step2. 벡터화시 파라미터인 tokenizer에 step1에서 만든 함수 대입후 벡터화 실시 </br>
step3. 이후 목적에 맞는 지도/비지도 모델 선정 후 학습 및 예측