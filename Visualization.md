# **Visualization**
여러 툴을 사용하여 데이터의 시각화를 통해 데이터 처리의 효율성으로 머신러닝 성능을 더 높이는데 목적을 둔다 </br></br>

**-Python 시각화 Library-** </br>
**Matplotlib** : 전반적인 모든 분야 시각화</br>
**Seaborn** : 주로 통계적인(Statistical) 시각화</br>
**Plotly** : 주로 업무 분석(Business Analysis) 시각화 </br>
**Pandas**</br></br></br>


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
plt.plot(axis별 좌표 리스트) : 각 좌표를 매칭시켜 그림을 그림 </br>
plt.title(str) : title을 설정 </br>
plt.show() : 그림판 호출</br>
fig, (ax1, ax2, . . .) = plt.subplots(nrows, ncols, figsize) = 여러 개의 Axes를 가지는 Figure 객체 설정 및 반환






## **Seaborn**
Matplotlib 기반으로 작성 되었으며, 기존 Matplotlib의 문제점들을 개선한 라이브러리 </br>
단, 사용을 위해선 Matplotlib을 잘 알고 있어야함