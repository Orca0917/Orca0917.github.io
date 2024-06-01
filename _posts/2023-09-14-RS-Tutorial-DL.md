---
title: Deep Learning for Recommender Systems
date: 2023-09-14 12:52:00 +0900
categories: [Recommender system, Survey]
tags: [tutorial]     # TAG names should always be lowercase
author: moon
math: true
toc: true
---

[Deep Learning for Recommender Systems](https://dl.acm.org/doi/10.1145/3109859.3109933)
- ACM 2017
- Alexandros Karatzoglou, Balázs Hidasi

<br>

## Abstract

딥러닝은 추천시스템에서 앞으로 해야할 큰 일 중 하나이다. 생각해보면 현재 컴퓨터비전, 자연어처리, 그리고 음성인식까지 딥러닝이 적용되어서 아주 뛰어난 성과들을 보여주고 있는 상태다. 이에 맞추어 추천시스템 분야는 2016년이 되어서야 본격적으로 딥러닝이 적용되기 시작했다. 따라서, 본 튜토리얼 논문을 통해 음악추천, 뉴스추천, 세션기반 추천 뿐만 아니라 전반적인 task에 걸쳐서 딥러닝을 적용시키는 다양한 연구가 진행되기를 기대한다.

<br/>

## 1. Introduction

저자가 본 논문을 통해서 소개하고 싶은 것은 2가지 이다.

1. RNN이나 CNN처럼 추천시스템에 사용되고 있는 딥러닝 기술에 대한 소개
2. 추천을 위한 SOTA 모델에 대한 소개

<br/>

현재는 matrix factorization이나 tensor factorization기법을 딥러닝을 통해 학습하고 있다. 딥러닝을 활용한다면, 특징들을 추출하고 유저 데이터나 아이템 데이터 속에 존재하는 숨겨진 패턴을 찾아 사용자들에게 좋은 추천을 제공할 수 있을 것이다. 이전의 기법과 달리 딥러닝의 특징은 데이터 속에 존재하는 특징을 아주 잘 추출해낸다는 것이다.

<br/>

또 다른 장점으로, 현재 사용중인 협업필터링 기법중 matrix factorization은 시간적 정보를 활용할 수가 없는데, RNN 구조를 사용하게 된다면 시간적 정보 (상품을 구매하는 순서)들을 사용할 수 있게되므로 더 나은 추천을 하는데 큰 도움을 줄 수 있게 된다.

<br/>

## 2. Deep Learning Techniques for Recommender Systems

딥러닝이라는 분야 자체는 2006년 Hinton 교수가 한 논문을 발표한 이후로, 엄청난 연구가 시작되었다. 이미지를 처리하는 컴퓨터 비전, 텍스트를 처리하는 자연어처리, 그리고 음성처리까지 딥러닝을 통해 엄청난 발전을 이루었지만, 추천시스템 분야에는 그렇게 빨리 적용되지는 않았다. 하지만 딥러닝이 지속적으로 발전하고 추천시스템도 동시에 발전함에 따라 딥러닝을 적용시키는 연구가 점차 활발해지기 시작했다.

<br/>

이제부터 아래에서는 최근 3~4년 동안 연구된 딥러닝기반 추천시스템을 여러 카테고리로 나누어서 설명하려고 한다. 논문의 저자인 Alexandros Karatzoglou는 크게 4가지의 카테고리로 분류하였다.

<br/>

![RS-Tutorial-4-Classification](/assets/img/RS-Tutorial-DL/tutorial-classification.png){: style="display:block; margin:auto;" w="90%"}

<br/>

### 2.1. Embedding methods

자연어처리 분야에서 발표된 word2vec을 바탕으로, 추천시스템에서 유저의 정보 또는 아이템 정보 심지어는 프로필의 텍스트 정보까지 모두 임베딩해서 추천에 사용되는 방법이다. 유저 임베딩과 아이템 임베딩을 내적하여 추천의 결과를 직접적으로 제공하는 것도 가능하지만, 다른 모델의 입력을 위해 임베딩을 사용하기도 한다.

<br/>

어떻게 보면 협업필터링의 matrix factorization도 임베딩처럼 바라볼 수는 있지만, 임베딩 벡터가 훨씬 확장성이 좋아 최근에는 벡터형식이 많이 사용되고 있다고 한다.

<br/>

### 2.2. Feedforward networks and autoencoders for collaborative filtering

이 방법은 FFN과 AE를 모두 사용하는 방법으로, 유저-아이템 상호작용 정보를 직접 모델링하게 된다. Deep Factorization Method 중 하나라고 볼 수 있으며, 고전적으로 사용되던 협업필터링 기법보다 대체적으로 좋은 성능을 보여준다.

<br/>

### 2.3. Deep feature extracting methods

딥러닝을 특징 추출에 사용한 모델로서, 보통 아이템의 특징을 추출해 추천에 활용하고 있다. 더 나아가, 상품의 이미지 또는 음악추천에 사용되는 음성파일도 모두 feature extraction하여 여러 정보를 결합한 추천인 hybrid recommendation에 사용되고 있다. 최근에는 상품 설명정보와 같은 텍스트에서 특징추출을 해 추천에 사용하는 연구도 이루어진다.

<br/>

### 2.4. Session based recommendation with recurrental neural networks

![Tutorial-Session-based](/assets/img/RS-Tutorial-DL/tutorial-session-based.png){: style="display:block; margin:auto;" w="90%"}

이 방법은 RNN을 사용한 추천으로, 상품 구매 또는 열람의 시간적 정보를 모두 고려할 수 있다. 유저별로 세션을 구성하는 것이 아니라, 세션으로 나눠진 데이터들은 모두 익명으로 처리되기 때문에 새로운 세션 데이터에 대해 더 잘 예측할 수 있게 된다. 성능을 평가했을 때, 기존 협업필터링 방식보다 큰 격차를 벌리고 높은 추천 정확도를 보여주었다.

<br/>

## 3. Future

Future에서는 딥러닝 연구가 지속적으로 이루어짐에 따라 저자가 기대하는 앞으로의 방향성을 제시한다. 먼저 논문이 발표될 당시 연구되고 있던 Adversarial Learning, Siamese Network, One-shot learning들이 추천시스템에 적용이 될 수 있을지는 모르겠지만, 결국에는 적용될 수 있을 것이라고 한다. 

<br/>

최근에는 수집하는 데이터의 품질도 좋아지고, feature extraction의 품질도 상당히 좋아짐에 따라 딥러닝 모델을 추천시스템에 많이 적용할 것이라고 보고있다. 한편으로는 대화형 인공지능 서비스가 등장하면서 대화형 추천시스템도 결국 만들어질 것이라 예상한다.

<br/>

현재로 보았을 때, 아직 추천시스템에 적용되는 모든 기법을 알고있지는 못하지만, 분명 adversarial learning은 사용되고 있고 성능도 계속해서 증가하고 있는 것에는 틀림없다. 실제로 Chat-GPT가 등장하면서 대화형 서비스가 큰 성공을 거두었는데, 추천시스템과 대화형 서비스가 결합되면 어떤 서비스가 될지 기대가 되는 바이다.

<br/>

## References

[1] Alexandros Karatzoglou and Balázs Hidasi. 2017. Deep Learning for Recommender Systems. In Proceedings of the Eleventh ACM Conference on Recommender Systems (RecSys '17). Association for Computing Machinery, New York, NY, USA, 396–397. https://doi.org/10.1145/3109859.3109933

[2] Nisha Muktawar, Session based recommender systems, cloudera blog, https://blog.cloudera.com/session-based-recommender-systems/


