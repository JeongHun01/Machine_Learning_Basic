## **Basic**
**ndarray** : N 차원(Dimension) 배열(Array)의 객체</br>
사용하는 이유? -> 파이썬의 기존 list는 대용량의 데이터를 다루기엔 다소 비약한 부분이 있으며 numpy는 이를 보완함</br></br>
**Dimension** : 차원을 뜻하며 주로 vector가 만드는 공간 </br>
**Shape** : 튜플 형식으로 각각의 크기를 표시(데이터의 수 / axis 관점) - 1차원인 경우 (3,)  ',' 표시</br></br>
**ndarray내의 데이터 타입은 그 연산 특성상 같은 데이터 타입만 가능 - 가장 겉 []기준** </br>
->[1, 0.9]는 데이터 타입이 더 큰 float형으로 형변환이 일어나 [1.0,0.9] </br>
강제 형변환 (대용량 메모리 절약시 사용) - astype() </br>
type() - numpy.ndarray / dtype() - 데이터 요소의 타입</br></br>
**Axis** : 배열의 축 -> shape 기준 (axis0, axis1, axis2, . . . , axisN) </br>
축이므로 차원에서 방향성을 가짐 (각각의 axis에 해당하는 데이터 배열)
</br></br>

**Extra method**</br>
np.round(ndarray, num) : 내부 요소들 num자리까지 반올림

## **Create**
**ndarray 생성**</br>
arange(n || start, end, step): 0~n-1 요소의 리스트 생성 (range()와 동일)</br>
zeros(shape, dtype) : shape와 datatype을 매개변수로 가지며, 0으로 초기화된 배열 생성</br>
ones(shape, dtype) : zeros와 동일하지만 1로 초기화 </br>
\#default dtype = 'float64'</br>
reshape() : shape 변환 - tuple에 -1이 있는 경우, 데이터 수에 따라 다른 axis를 기준으로 자동으로 할당

## **Indexing**
**1. 단일값 추출** : array[axis0, axis1, . . .] - [shape] (Dimension을 1개 줄인 후 출력) </br>
**2. Slicing** : array[a:b(axis0) , c:d(axis1), . . .] - [n1:n2(shape)] (Dimension 유지) - Continuous</br>
**3. Fancy Indexing** : array[anything] - axis순으로 list, ndarray, slicing, 단일값 등 원하는 범위를 지정 - Continuous + Discontinous </br>
**4. Boolean Indexing** : array[conditional statement] - 각 index의 true/false를 판별하여 true의 index만 저장

## **Sorting**
np.sort(ndarray, axis) : ndarray는 유지하되, 정렬된 배열을 새롭게 반환 - inplace = False</br>
ndarray.sort(axis) : ndarray를 정렬 - inplace = True</br>
np.argsort(ndarray, axis) : ndarray 정렬 후, 정렬 전 인덱스 값 기준 정렬해서 반환 </br>
\# 내림차순 : +[::-1]

## **Linear Method**
np.dot(A, B) : 내적 </br>
np.transpose() : 전치