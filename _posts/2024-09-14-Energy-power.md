---
title: 전기 신호에서 에너지와 파워
date: 2024-09-14 11:50:00 +0900
categories: [Signal, Hyukppenheim]
tags: [euler equation, taylor series]     # TAG names should always be lowercase
author: moon
math: true
toc: true
---

[[혁펜하임 강의]](https://www.youtube.com/watch?v=olZzJ_WAoTA&list=PL_iJu012NOxcDuKgSjTKJZJd3bQtkAyZU&index=3)

<br>

에너지와 파워는 물리학과 공학에서 중요한 개념입니다. 전구가 빛을 내거나, 음악 신호가 스피커를 통해 소리로 전달될 때 모두 에너지가 사용됩니다. 이번 글에서는 에너지와 파워가 어떻게 정의되고, 그 공식을 왜 사용하는지 이해하기 쉽게 설명하겠습니다.

## 1. 에너지란 무엇인가?

에너지는 물체가 일을 할 수 있는 능력을 의미합니다. 물체가 움직이거나 힘을 받을 때, 그 물체에 에너지가 전달됩니다. 예를 들어, 우리가 쇼핑카트를 밀면 카트가 앞으로 움직입니다. 이때 우리가 카트에 일을 했고, 그만큼 카트에 에너지가 전달된 것입니다.

<br>

### 에너지의 단위
에너지의 단위는 줄(J, Joule)입니다. 1줄(J)은 1뉴턴(N)의 힘으로 물체를 1미터(m) 이동시킬 때 필요한 에너지를 의미합니다. 여기서 힘의 단위인 뉴턴은 $1 \, \text{kg}$ 의 물체에 $1 \, \text{m/s}^2$ 의 가속도를 가할 때 필요한 힘입니다. 

<br>

## 2. 일과 에너지의 관계
일(work)은 힘을 가한 방향으로 물체가 이동한 거리에 비례해서 발생합니다. 물리학에서 일은 다음과 같이 정의됩니다:

$$
W = F \cos \theta \, dx
$$

여기서:
- $F$ 는 물체에 가해진 힘입니다.
- $dx$ 는 물체가 이동한 거리입니다.
- $\theta$ 는 힘이 물체의 이동 방향과 이루는 각도입니다.

![alt text](/assets/img/signal-and-system/force-work.png)

이 식에서 $\cos \theta$ 는 물체가 이동하는 방향과 힘이 이루는 각도에 따른 기여도를 나타냅니다. 예를 들어, 힘이 이동 방향과 일치하면 $\cos \theta = 1$ 이 되어 일은 $F \cdot dx$ 가 됩니다. 반대로, 힘이 이동 방향과 수직이면 $\cos \theta = 0$ 이 되어 물체가 이동해도 일을 하지 않은 것으로 계산됩니다.

<br>

## 3. 신호 에너지는 어떻게 구할까?

신호 에너지로 넘어가기 전에, 물리적 에너지와 신호 에너지가 어떻게 다른지 이해해야 합니다. 물리적 에너지는 힘과 거리, 그리고 각도 같은 공간적 요소를 포함하지만, 신호 에너지는 시간에 따라 변하는 **진폭**에만 집중합니다. 이때, 진폭의 제곱을 사용해 에너지를 계산합니다. 신호는 음성, 전기 신호, 또는 디지털 데이터를 의미할 수 있으며, 신호의 에너지는 시간에 따른 진폭의 크기를 기반으로 계산합니다.

신호 에너지를 구하는 공식은 다음과 같습니다:

$$
E = \int_{t_1}^{t_2} |x(t)|^2 dt
$$

여기서:
- $\|x(t)\|^2$ 는 신호 $x(t)$ 의 크기(진폭)의 제곱을 의미합니다.
- $\int$ 는 시간 구간 동안 신호의 에너지를 계산하는 적분입니다.

<br>

## 4. 왜 신호 에너지를 크기의 제곱으로 계산할까?

신호 에너지를 계산할 때 각도가 사라지고, 대신 진폭의 제곱을 사용하는 이유는 신호 에너지가 **공간적 관계**가 아니라 **시간적 관계**에만 의존하기 때문입니다. 물리적 에너지는 힘이 어느 방향으로 작용하느냐에 따라 달라지기 때문에 각도 $\theta$ 를 고려해야 합니다. 그러나 *신호 에너지는 그 신호의 크기(진폭)에 의해 결정됩니다.* 신호가 양수일 때나 음수일 때 모두 에너지를 포함하고 있기 때문에, 진폭의 절대값을 계산하고, 그 크기를 제곱해 에너지를 양수로 유지합니다.

예를 들어, *음악 신호에서 소리가 크게 날 때(진폭이 클 때) 더 많은 에너지가 소비되지만, 그 소리가 음수 영역에 있더라도 에너지는 여전히 양수입니다.* 음의 진폭을 고려하지 않고 에너지를 계산하면 마치 에너지가 상쇄되는 것처럼 보일 수 있기 때문에, 진폭의 제곱을 사용해 에너지가 정확히 계산되도록 합니다.

<br>

## 5. 파워란 무엇인가?

파워(Power) 또는 일률은 단위 시간당 얼마나 많은 일을 했는지를 의미합니다. 쉽게 말해, 전구가 1초에 얼마나 많은 에너지를 소비하는지를 나타내는 것과 같습니다. 파워는 다음과 같이 정의됩니다:

$$
P = \frac{\text{일}}{\text{시간}} = \frac{J}{s} = W
$$

여기서 $W$ 는 와트(Watt)로, 파워의 단위입니다. 예를 들어, 60W 전구는 1초에 60줄(J)의 에너지를 소비합니다. 즉, 더 높은 파워를 가진 전구일수록 더 많은 에너지를 소비합니다.

<br>

## 6. 평균 파워

평균 파워는 일정 시간 동안 발생한 파워의 평균값입니다. 평균 파워를 구하는 식은 다음과 같습니다:

$$
\frac{1}{t_2 - t_1}\int_{t_1}^{t_2} |x(t)|^2 dt
$$

이 공식은 일정 시간 동안 신호가 소비한 에너지를 시간에 대해 평균화한 값입니다. 예를 들어, 전기기기가 특정 시간 동안 사용한 평균 전력을 계산할 때 이 공식을 사용할 수 있습니다.

<br>

## 7. 일상생활에서의 예시

전구가 전기를 사용해 빛을 내는 것을 생각해 봅시다. 전구의 전력이 60W라면, 이는 전구가 1초당 60줄의 에너지를 소비한다는 뜻입니다. 전구를 오래 켜둘수록 더 많은 에너지를 소비하게 됩니다. 이때 소비한 에너지를 적분으로 계산할 수 있습니다. 전구의 에너지가 크면(와트가 높으면) 더 많은 빛을 내기 위해 더 많은 전기 에너지를 사용하게 됩니다.

<br>

## 정리

에너지는 물체에 일을 가하거나 신호가 전달될 때, 그 힘이나 크기에 비례해 계산할 수 있습니다. 물리적 에너지는 물체의 힘, 이동 거리, 각도 간의 관계를 고려해야 하지만, 신호 에너지는 시간에 따른 진폭의 제곱을 기반으로 계산합니다. 크기의 제곱을 사용함으로써 신호의 양수, 음수에 관계없이 에너지가 항상 양수로 유지됩니다. 파워는 단위 시간당 얼마나 많은 에너지를 소비했는지를 의미하며, 평균 파워를 구하면 일정 시간 동안의 소비량을 평균화할 수 있습니다.