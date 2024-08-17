## **Basic**
인간이 가장 이해하기 쉬운 데이터 구조인 2차원을 효율적으로 가공/처리 가능

**DataFrame** : Column 과 Row로 구성된 2차원 data set </br>
**Series** : 1개의 Column값으로만 구성된 1차원 data set </br>
**Index** : DataFrame/Series의 고유한 key값 객체 (Row에 해당)

#### **기본 API**
pd.read_csv(file, sep) : (csv)파일을 불러와 DataFrame으로 저장 </br>
\#default sep = ',' / sep설정시 csv외 다른 파일 불러오기 가능 </br>
객체['column'] : 해당 columns들의 Series/DataFrame만 반환 (axis1 - 열) / 문자열인 경우 Series = 객체.column가능</br>
\#단, dtype이 index인 인자도 []안에 입력 가능 (.index 이용) </br> 
display(DataFrame) : DataFrame 출력 - 셀 상에서 마지막 줄인 경우 display()생략가능  (Series는 print만 가능한 듯)</br>
head(), tail() : 입력된 수만큼의 앞/뒤 데이터만 출력 - default = 5</br>
.rename(columns = {기존 : 변경}, inplace) : column 명칭 변경 </br>
pd.set_option() : DataFrame에 보여지는 데이터 조절 </br>
shape, values, columns, index, index.values, dtypes : 각각을 반환 </br>
info() : 여러 정보를 제공(Column명, 데이터 타입, Null의 수, . . .) </br>
describe() : 평균, 표준편차, 4분위 분포 제공 - 단 숫자형에 국한</br>
.nunique() : column내 몇 건의 고유값이 있는 지 확인 (value_counts는 고유값이 각각 몇 건인지) - numpy도 np.으로 가능</br>
df.copy() : 복사본 생성 / deep = True시 deep copy (default = False)</br>
merge(df1, df2, on) = on(column 교집합)기준으로 DataFrame 합병</br></br>
[column].value_counts() : 해당 column(Series)에서 동일한 데이터가 각각 몇 개 있는 지 출력(Null 제외) - DataFrame에도 사용 가능 </br>
\# 이때 column의 데이터들이 index로 표시 + value_counts의 column명은 count로 표시 + Series type 반환</br>
\# default dropna = True / False인 경우 NaN도 출력 </br></br>

**기본적으로 여러 개의 데이터가 입력될 시 list형태로 넣는다**</br>
**단일 데이터도 list형태로 넣어도 무방하다. 단, series반환 시에는 list를 사용할 시 aixs1= 1인 DataFrame이 반환된다**</br></br>
**함수가 무엇을 반환한다면 기본적으로 inplace = False라고 판단**

## **Interconversion**
**list -> DataFrame** : pd.DataFrame(list, columns = ['col_name]')</br>
\# 1차원 : 1개의 column 1열 / 2차원 : numpy상 행렬을 따름 (axis0, axis1) </br>
**ndarray -> DataFrame** : pd.DataFrame(array, columns = ['col_name'])</br>
**dict -> DataFrame** : pd.DataFrame(dict, cloumns, index)</br>
**DataFrame -> ndarray** : values 속성 이용 - .values</br>
**DataFrame -> list** : ndarray 변환 후 tolist() </br>
**DataFrame -> dict** : to_dict()

## **New Column / Edit Column**
**생성 및 수정** : ['cloumn_name'] = 간단한 수식 / 데이터 list </br>
**Lambda.version**-[column].apply(lambda statement) : 데이터 가공, column의 값들이 순서대로 lambda식에 들어간 후 반환 - 주로 복잡한 데이터를 가공할 경우 사용(if문)</br></br>
**삭제** : drop(labels, axis, index, columns, level, inplace, erros) </br>
&nbsp; &nbsp; &nbsp; &nbsp;
#1 labels : 해당 row, column 이름</br>
&nbsp; &nbsp; &nbsp; &nbsp;
#2 axis : row, column 선택</br>
&nbsp; &nbsp; &nbsp; &nbsp;
#3 inplace : 원본 적용 유무 / False - 원본 유지 + 변경 사항 새 객체 적용 , True - 원본 변경 + 반환 None

## **Index**
DataFrame, Series의 row 데이터를 고유하게 식별하는 객체 </br>
오직 식별용이며(차원에 영향 X), DataFrame, Series 객체에 포함되지만 연산에는 제외 </br>
immutable이므로 고유 함수 사용 이외 수정 불가</br>
cf) 객체[index]는 error지만 , 객체[slicing]은 반환이 된다. -> 사용은 지양</br></br>
.index : range index / .index.values : 1D ndarray </br>
reset_index(drop, inplace) : 변경 사항 - 기존 index를 column형태로 추가(default - drop = False) + index는 0~N-1로 초기화 </br>

#### **DataFrame Indexing / Filtering**
**1. []** : column기반 필터링 혹은 boolean indexing 필터링 제공</br>
**2. loc[], iloc[]** : 명칭/위치 기반 인덱싱 제공</br></br>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;
명칭 기반 - loc[index, column_name] : column의 명칭을 기반으로 열 위치 지정 + 행 위치는 index 이용</br>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;
#slicing, fancy, boolean 가능 (단, slicing은 : 뒤도 포함) </br></br>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;
위치 기반 - iloc[row, column] : 좌표 기반의 행과 열 위치를 기반으로 데이터 지정 </br>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;
#일반적 slicing, fancy 가능/ boolean 불가</br>

**3. Boolean Indexing** : 조건식에 따른 필터링 제공 - 객체[조건식]</br>
**4. column 명칭 + index 인덱스** : df[column_name][index]시 호출 가능</br> 


## **Aggregation / Group by**
**Aggreagation** : 집합 연산
1) count() : DataFrame의 Column들의 건수 (NaN 제외) </br>
2) mean(), sum(), min(), max(), . . . </br>
4) agg() : 2가지 이상의 Aggregation을 사용할 시 (column이 여러 개이며, 각각 다른 Aggregation을 사용한다면 dict / NamedAgg이용) - str형식 입력</br>

**Group by** : 그룹화 - groupby() </br>
해당 column에서 같은 데이터끼리 그룹화하여 나눈다 </br>
이 함수로 나온 객체는 DataFrameGroupBy type을 가지며 Group by method와 일부 DataFrame method를 사용 가능하다 </br></br>
**Group by로 원하는 데이터들끼리 그룹화 후, Aggregation으로 원하는 값을 뽑은 뒤 하나의 DataFrame type으로 바꾸어 display함** </br>
#DataFrame -> + groupby -> DataFrameGroupBy -> + Aggregation -> DataFrame </br>
cf) 이미지적으로 떠올리면 굉장히 간단 </br></br>

## **Extra**
**Missing Data** </br>
1) isna() : 주어진 column 값들이 NaN인가에 대해 True/False 반환 </br>
2) fillna() : Missing data를 인자로 주어진 값으로 대체 (inplace = False) </br>

**Sorting** </br>
.sort_values(by, ascending) : 정렬 / default - ascending = True(오름차순) - False : 내림차순 </br>

**Replace** </br>
[column].replace() : 원본 값을 특정 값으로 대체 (2개 이상 변경 시, dict이용)</br>

concat([df1,df2], axis) : DataFrame 합치기 - default axis = 0

