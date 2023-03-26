---
layout: post
title: "Logistic Regression"
Abstract: "로지스틱 회귀 개념 정리"
date: 2023-03-27 00:00:00 +0800
author: daeun
image: 
image_caption: 'Overview of Proposed Methods'
tags: [MachineLearinig, statistics]
---
# Logistic Regression

본 포스팅은 고려대학교 강필성 교수님의 강의 자료를 참고하여 작성하였습니다.

### Logistic Regression : 수식

### **Review : Multiple Linear Regression**

로지스틱 회귀 분석에 대해 알아보기 전 다중 선형 회귀 (Multiple Linear Regression) 에 대해 설명하도록 하겠습니다.

다중 선형 회귀의 목표는 수치형 설명 변수 X 와 연속형 데이터로 이루어진 종속변수 Y 간의 관계를 선형으로 정의하고 이를 가장 잘 표현할 수 있는 회귀 계수를 추정하는 것입니다.

![Untitled](/images/daeun/Untitled.png)

즉, 선형 결합 계수인 Beta hat 을 구하는 것이 학습의 목표가 됩니다.

예시를 살펴보겠습니다

![Untitled](/images/daeun/Untitled%201.png)

위의 그림은 나이와 혈압에 대한 데이터입니다. 

다음과 같은 연속형 데이터의 경우 나이가 증가함에 따라 혈압이 1.222 만큼 증가한다는 것을 알 수 있게 됩니다.

그렇다면 연속형 데이터 대신 범주형 데이터로 문제를 바꾸게 된다면 어떻게 될까요?  

![Untitled](/images/daeun/Untitled%202.png)

그래프를 보면 이 선은 발병(1) 정상(0) 의 범주를 제대로 표현하고 있지 않습니다. 이처럼 Y 가 범주형일 때 다중 선형 회귀 모델을 그대로 적용할 수 없습니다.

### Logistic Regression : Background

- 시그모이드 함수

$$
f(x) = \frac{1}{1 + e^{-x}}
$$

X 값으로는 어떤 값이든 받을 수 있지만 출력 값은 0에서 1 사이의 값을 출력하게 됩니다.

![Untitled](/images/daeun/Untitled%203.png)

확률밀도함수의 조건을 충족하는 함수이다.

1. 모든 x 값에 대응되는 f(x) 가 0 이상일 것
2. 특정 구간에서의 확률 밀도는 그 구간의 범위를 적분한 값
3. 전체 구간에서의 확률 밀도 함수는 1

- Odds

로지스틱 함수의 기본적인 개념이 되는 승산에 대해 알아보도록 하겠습니다.

승산이란 성공 확률을 p 로 정의할 때에 실패 대비 성공 확률의 비율을 나타냅니다.

$$
Odds = \frac{p}{1 - p}
$$

$(p =1;odd=infinite,p=0;odd=0)$

![Untitled](/images/daeun/Untitled%204.png)

- logit function

음의 무한대부터 양의 무한대까지의 실수 값을 0 부터 1 사이의 실수값으로 1 대 1 대응시키는 시그모이드 함수입니다.

위의 odds 함수에 log 를 취하게 되면

$$
z = logit(Odds) = log(\frac{p}{1 - p})
$$

logit function 의 값은 로그 변환에 의해서 음의 무한대부터 양의 무한대까지의 값을 가질 수 있게 됩니다..

![Untitled](/images/daeun/Untitled%205.png)

p 가 0.5 일 경우 log(p/(1-p) = 1) = log1 = 0

p가 0.5 보다 낮을 때 -infinite, p 가 0.5 보다 큰 경우 infinite 으로 수렴하게 됩니다.

 

- logistic function

logit function 의 역함수로 음의 무한대부터 양의 무한대의 값을 가지는 입력 변수를 0부터 1 사이의 값을 가지는 출력변수로 변환한 것

$$
logistic(z) = \frac{1}{1+exp(-z)}
$$

![Untitled](/images/daeun/Untitled%206.png)

p = exp(y) / (1+exp(y))

분모 분자에 exp(-y) 를 곱해주면

p = 1 / (1 + exp(-y)) 로 시그모이드 형태와 같아 짐

![Untitled](/images/daeun/Untitled%203.png)

### Logistic Regression

다중 선형 회귀에서 사용했던 식을 그대로 들고와

y 를 log(Odds) 즉, 어떠한 확률의 값으로 바꾼다면 ?

- 다중 선형 방정식

![Untitled](/images/daeun/Untitled.png)

- odds 에 대한 선형 방정식

![Untitled](/images/daeun/Untitled%207.png)

- 각 항에 지수 함수를 취하여 log 를 없앰

![Untitled](/images/daeun/Untitled%208.png)

- 위의 로지스틱 함수에 대한 정리와 동일

![Untitled](/images/daeun/Untitled%209.png)

추정된 회귀 계수 B 로 부터 사후 확률 P 를 추정하는 공식이 됨

### Logistic Regression : 학습

로지스틱 회귀 분석의 모수 w 는 최대 가능도 (Maximum Likelihood Estimation MLE) 방법으로 추정할 수 있습니다.

우선 베르누이 시행에 대해 정의하겠습니다.

베르누이 시행은 결과가 성공 또는 실패의 두가지 중 하나로만 나오는 실험입니다.

베르누이 시행의 결과를 확률 변수 X 로 나타내는 경우 X = 1 을 성공, X = 0 을 실패라고 둘 수 있습니다. 불연속적인 두 가지의 경우의수를 가지므로 X 는 이산확률변수가 됩니다.

X = 1 일 확률을 성공 확률이라고 부르고, 이때 확률 변수 X 가 모수의 베르누이 분포를 따른다고 합니다.

베르누이 분포는 다음과 같이 표현합니다.

![Untitled](/images/daeun/Untitled%2010.png)

베르누이 분포에 로지스틱 함수를 적용하면

![Untitled](/images/daeun/Untitled%2011.png)

![Untitled](/images/daeun/Untitled%2012.png)

![Untitled](/images/daeun/Untitled%2013.png)

다음과 같이 정리할 수 있습니다.

데이터의 표본이 여러개 있는 경우 전체 데이터에 대해 로그 가능도를 구해보겠습니다.

![Untitled](/images/daeun/Untitled%2014.png)

각 베르누의 확률 분포를 곱해준 것에 log 를 취해주게 됩니다. 

여기서 likeli hood 를 계산하기 위해 각 데이터 샘플에서 분포에 대한 likeli hood 를 더하지 않고 모두 다 곱한 이유는 이 모든 데이터들의 sampling 이 독립적으로 연달아 일어나는 사건이기 때문입니다.

여기에 자연로그를 취하게 되면 각항이 덧셈으로 이루어지게 됩니다.

선형회귀에서 오차를 최대화 하는 w 의 값을 구했던 것 처럼

로지스틱 회귀에서는 로그 가능도를 최대화 하는 w 의 값을 구해야합니다.

따라서 로그 라이클리 후드를 모수로 미분합니다.

![PNG 이미지 (1).png](/images/daeun/PNG_%25EC%259D%25B4%25EB%25AF%25B8%25EC%25A7%2580_(1).png)

** 부록 **

2번 도출 과정 풀이

μ(xi;w) = 1 / (1 + exp(-wT * xi))

우선 w로 편미분하는 과정에서 체인룰을 사용하게 됩니다

∂μ(xi;w)/∂w = ∂μ(xi;w)/∂(wT*xi) * ∂(wT*xi)/∂w

여기서 ∂(wT*xi)/∂w = xi 입니다.

∂μ(xi;w)/∂(wT*xi)을 찾기 위해

미분의 체인룰을 사용하면 다음과 같이 나타낼 수 있습니다.

∂μ(xi;w)/∂(wT*xi) = ∂(1/(1+exp(-wT*xi)))/∂(wT*xi)

시그모이드 함수의 미분이므로 다음과 같이 표현할 수 있습니다

∂μ(xi;w)/∂(wT*xi) = μ(xi;w) * (1 - μ(xi;w))

따라서, 다시 ∂μ(xi;w)/∂w 식으로 돌아가면,

∂μ(xi;w)/∂w = ∂μ(xi;w)/∂(wT*xi) * ∂(wT*xi)/∂w = μ(xi;w) * (1 - μ(xi;w)) * xi

이렇게 편미분 값을 구할 수 있습니다.

### 수치적 최적화

LOG LIKELIHOOD 를 최대화 하는 것은 다음 목적함수를 최소화 하는 것과 같습니다.

$$
J = -LL
$$

gradient vector 는

$$
gk = \frac{d}{dw}(-LL)
$$

다음과 같고, gk 에 learning rate 를 곱한 것 만큼 이동하게 됩니다.

따라서 정리해보면

![Untitled](/images/daeun/Untitled%2015.png)

기울기의 업데이트는 다음과 같이 진행되게 됩니다.