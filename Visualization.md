# **Visualization**
여러 툴을 사용하여 데이터의 시각화를 통해 데이터 처리의 효율성으로 머신러닝 성능을 더 높이는데 목적을 둔다 </br></br>

**-Python 시각화 Library-** </br>
**Matplotlib** : 전반적인 모든 분야 시각화</br>
**Seaborn** : 주로 통계적인(Statistical) 시각화</br>
**Plotly** : 주로 업무 분석(Business Analysis) 시각화 </br>
**Pandas**</br></br>

**-차트의 유형들-** </br> 
**Histogram** : 연속형 값에 도수 분포를 표현</br>
**Violin Plot** : 어떤 값의 분포를 특정 값 별로 기하학적으로 표현 (히스토그램 변형 버전)</br> 
**Bar Plot** : 주로 이산적인 데이터 x축에 대한 y축 값을 표현 (Breakdown -> Grouped Bar)</br> 
**Scatter Plot** : 점들로 분포를 찍어내는 것으로 주로 x축 값 y축 값 모두 연속값이며, Outlier 탐색에 용이</br> 
**Line Plot** : 선으로 이어지는 그래프로, 주로 시계열에 사용 ( ex. 주식 )</br>
**Bot Plo** : 분위를 나타낼 때 사용 </br>
**Heatmap** : 정보를 일정한 이미지위에 열분포 형태의 그래픽으로 시각화 - 주로 상관도 측정에 이용 </br></br> 

## **Matplotlib**
Python Graph Visualization으로 가장 많이 사용되는 라이브러리 </br>
3차원 이상의 입체 시각화도 다양하게 지원</br>
문제점 : 직관적이지 못한 API + 현대적인 감각이 떨어지는 Visual (따로 설정을 통해 개선 필요) </br></br>
import matplotlib.pyplot as plt </br>
pyplot은 MATLAB 스타일의 Interface를 가지며, MATLAB 사용자들이 보다 쉽게 python에 적응할 수 있게 설계 됨</br></br>

**Figure** </br>
그림을 그리기 위한 Canvas의 역할 - 그림판의 크기 등을 조절</br>
예시\) plt.show()는 내부적으로 Figure.show()를 호출하여 그림을 나타냄
</br></br>
**Axes** </br>
실제 그림을 그리는 method들을 가짐 + x축, y축, title등의 속성 설정 </br>
예시\) plt.plot() - 기본으로 설정 된 Axes에서 Axes.plot()을 호출하여 그림을 그림 / plt.title()은 내부적으로 Axes.set_title()를 호출하여 title 설정</br></br>
-> Figure로 그림판을 만들고, Axes로 다 그리며 이름 등을 설정한 후, Figure로 그림판을 호출하여 시각화</br></br>

### **API**
plt.figure(figsize = (가로, 세로), 여러 설정) : figsize 크기에 맞춰 Figure 객체를 설정하고 반환 / 여러 설정을 추가해 그림판 꾸밈 가능</br>
plt.bar/plot(axis별 좌표 list/ndarray, 그래프를 꾸미는 여러 설정) : 각 좌표를 매칭시켜 그림을 그림 / 좌표 개수는 동일해야함 (보간법)</br>
\# 여러 설정 예시 - color, marker, linestyle, linewidth, markersize, label, . . .</br>
plt.title(str) : title을 설정 </br>
plt.show() : 그림판 호출</br>
fig, (ax1, ax2, . . .)/ax([]로 접근) = plt.subplots(nrows, ncols, figsize) = 여러 개의 Axes를 가지는 Figure 객체 설정 및 반환</br>
plt.xlabel/ylabel() : x축/y축에 축명을 텍스트로 할당</br>
plt.xticks/yticks(ticks, rotation) : x축/y축 단위/문자열 rotation만큼 회전 + 아래 눈금을 ticks = np.arange(범위)로 나눔 - 주로 문자열 있을 때 사용 </br>
plt.xlim/ylim((최소,최대)) : x축/y축에 나타나는 값 범위 제한 </br>
plt.legend() : label이 붙은 plot들을 범례 표시 </br></br>
**주의사항** : plt.axes()로 객체를 반환한 후 객체.method로 작업할 때, 일부 method명 변경 ex) set_xlabel, set_title</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
특히 subplot을 사용할 때, 여러 Axes 객체를 반환 받으므로 이때 사용</br></br>

## **Seaborn**
Matplotlib 기반으로 작성 되었으며, 기존 Matplotlib의 문제점들을 개선한 라이브러리 </br>
단, 사용을 위해선 Matplotlib을 잘 알고 있어야함</br></br>
import seaborn as sns</br>
기본적으로 plt.figure() 필요 X / 하지만 figure size를 정하고 싶거나 subplot을 부를 땐 호출</br></br>
**Axes level** : 기존 Matplotlib과 유사하게 개별 Axes가 plot에 대한 주도적인 역할 수행 </br>
**Fiqure level** : Seaborn의 FacetGrid 클래스에서 개별 Axes기반의 plot을 그릴 수 있는 기능을 통제 </br>
버전이 업그레이드 되면서 Seaborn은 Axes level -> Figure level 방향성을 가짐</br>
장점 : 여러 개의 subplot을 쉽게 생성 + 자동 인식 + 여러 plot 들을 복합하여 쉽게 시각화</br>
단점 : 새로운 API + 커스터마이징 변경 적용 어려움 존재 </br>
-> 이번 학습에선 Axes level 위주로 하고, 나중에 필요시 Figure level은 따로 구글링</br></br>
**seaborn에서 subplot 사용법**</br>
Axes level function에 인자인 ax에 객체[]를 넣어준다. ex) ax = axes[1]</br>
이후 method 사용은 Axes[].set_~ (단 legend는 set 사용 X)</br></br>
**Rotation 이용법**</br>
주로 축 상 너무 촘촘하여 글자가 안보일 때 사용 </br>
Axes[].set_x/yticklabels(Axes[].get_x/yticklabels(), rotation)</br></br>

### **Histogram**
**Matplotlib** : plt.hist()</br>
**Pandas** : Series.hist() </br></br>
**Seaborn** </br>
sns.histplot() - Axes level / distplot() - Figure level</br></br>
파라미터 : kde = pdf형 그래프 유무 (default = True) - y축에 Axes level은 count / Figure level은 density</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
bins = bin의 개수 지정</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
rug = x축 상에 밀도 표현 (default = False)</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
x = 원하는 column 명 / data = DataFrame -> DataFrame에서 column을 찾아 표현 (Axes level만 가능)</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
ax = subplot 사용 시 Axes 설정</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
label = label 설정</br></br>
sns.countplot(x, data) : 해당 column의 value_counts를 시각화</br></br>

### **Bar plot**
sns.barplot()</br></br>
파라미터 : x/y = 각각의 x축 y축에 넣을 column명</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
data = DataFrame</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
errorbar = 오차구간 표시 (default = ('ci',95) )</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
color = 색상 설정 (조건식 이용 가능)</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
estimator = x축(문자열 축)기준 y축 표현 지표 (default = 'mean')</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
hue = 들어오는 column명 기준으로 breakdown (stacked bar는 지원 X)</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
order = x축(문자열 축) 순서 설정</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
orient = 'h'시 x축 y축 변경(90도 회전)</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
palette = bar들의 색상 구별 설정 </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
ax = subplot 사용 시 Axes 설정</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
label = label 설정</br></br>

### **Violin Plot**
sns.violinplot()</br>
x축 값 별로 y축 값의 연속분포 곡선을 알 수 있음</br></br>
파라미터 : x/y = 각각의 x축 y축에 넣을 column명</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
data = DataFrame</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
ax = subplot 사용 시 Axes 설정</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
label = label 설정 </br></br>

### **Scatter Plot**
sns.scatterplot()</br>
산포도로서 x축과 y축에 연속형 값을 시각화</br></br>
파라미터 : x/y = 각각의 x축 y축에 넣을 column명</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
data = DataFrame</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
hue = breakdown</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
style = 해당 column 정보를 산포도 점에 마커로 표시 (새로 찍어내는 것 아님 - hue와 다른점)</br></br>

### **Box plot**
sns.boxplot()</br>
4분위를 박스 형태로 표현(25%~75%) -> x축값에 이산값을 부여하면 이산 값에 따른 시각화</br></br>
파라미터 : x/y = 각각의 x축 y축에 넣을 column명</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
data = DataFrame</br></br>

### **Heatmap**
corr = df.corr() -> sns.heatmap(corr)</br>
column간의 상관도를 heatmap 형태로 표현</br></br>
주로 연속형 값끼리 상관도를 봄 (feature와 label이 모두 연속형 - Regression)</br>
카테고리성은 상관도 측정에 약간 애매한 부분이 있지만 충분히 해석에 이용은 가능</br></br>
파라미터 : annot = 칸에 상관도 값을 표시 (default = False)</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
fmt = formatting (주로 소수점 자리 설정)</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
cbar = 상관도 색상 수치도 표시 (default = True)</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
linewidth = 칸 사이에 라인 크기 설정 (default = 0)</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
cmap = 디자인 설정
