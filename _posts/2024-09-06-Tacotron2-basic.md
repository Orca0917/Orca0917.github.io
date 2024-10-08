---
title: Tacotron 2 (기본)
date: 2024-09-06 01:53:00 +0900
categories: [Audio, Speech Synthesis]
tags: [tacotron2]     # TAG names should always be lowercase
author: moon
math: true
toc: true
---

컴퓨터나 스마트폰에 텍스트를 입력하면, 바로 자연스럽고 인간처럼 들리는 음성이 나오기를 기대해본 적이 있나요? 단조롭고 기계적인 음성이 아닌, 부드럽고 자연스러운 목소리 말이죠. 이런 기술이 바로 Tacotron 2가 실현하고 있는 미래입니다. Tacotron 2는 텍스트를 음성으로 변환하는(text-to-speech, TTS) 신경망 아키텍처로, 인공지능이 점점 더 인간다운 음성을 생성할 수 있게 해주는 혁신적인 시스템입니다. 이번 글에서는 Tacotron 2가 어떻게 작동하는지, 왜 중요한지, 그리고 이 기술이 음성 합성의 미래를 어떻게 변화시키고 있는지 알아보겠습니다.

<br>

## 1. Tacotron 2란 무엇인가?

Tacotron 2는 기본적으로 텍스트를 자연스러운 음성으로 변환하는 시스템입니다. 이것이 쉬운 일처럼 들리겠지만, 사실 음성을 자연스럽게 생성하는 일은 많은 난관이 있습니다. 수십 년 동안 연구자들은 컴퓨터가 사람처럼 말할 수 있게 만드는 방법을 고민해왔습니다. 초기에는 연결합성(Concatenative synthesis) 방식이 많이 사용되었는데, 이 방식은 미리 녹음된 음성 조각들을 이어붙여 새로운 문장을 만드는 방법입니다. 이 방법은 정교한 기술이 필요하지만, 결과적으로 경계에서 소리의 부자연스러움이 자주 발생하고, 이는 듣는 이로 하여금 기계적인 느낌을 받게 만듭니다.

이 문제를 해결하기 위해 등장한 것이 통계적 파라메트릭 음성 합성(Statistical Parametric Speech Synthesis)입니다. 이 방식은 사전에 녹음된 음성을 조각내는 것이 아니라, 음성의 특징을 기반으로 소리를 직접 생성합니다. 그러나 이 방식도 여전히 소리가 부자연스럽고 뭉개지는 문제를 가지고 있었습니다. 예를 들어, 마치 사람이 말을 할 때 소리를 조금 억누른 듯한 느낌을 줬죠.

<br>

## 2. WaveNet과의 만남: 혁신의 시작

2016년, WaveNet이라는 기술이 등장하면서 음성 합성에 새로운 바람이 불기 시작했습니다. WaveNet은 시간 도메인의 파형(waveform) 자체를 생성하는 모델로, 실제 사람의 목소리와 거의 구별할 수 없을 정도로 자연스러운 음성을 만들어냈습니다. 이 기술은 음성 합성 분야에서 큰 돌파구를 마련했지만, 한 가지 문제점이 있었습니다. WaveNet을 활용하기 위해서는 음성의 언어적 특징(단어, 음소 등)과 기본 주파수($F_0$) 등의 세부적인 정보가 필요했습니다. 이 정보를 만들어내는 과정이 매우 복잡하고, 관련 지식이 필요했죠.

여기에서 Tacotron이 등장했습니다. Tacotron은 시퀀스-투-시퀀스(Sequence-to-Sequence) 아키텍처로, 입력된 텍스트로부터 스펙트로그램을 생성하는 모델입니다. 스펙트로그램이란 음성의 주파수 구성을 시각적으로 표현한 것인데, Tacotron은 이를 활용해 WaveNet보다 더 간단한 방식으로 자연스러운 음성을 만들 수 있었습니다. 그러나 이때까지 Tacotron은 Griffin-Lim 알고리즘을 사용해 음성을 생성했는데, 이 방법은 오디오 품질이 WaveNet보다 떨어진다는 단점이 있었습니다.

<br>

## 3. Tacotron 2: 완전한 신경망 기반 음성 합성

Tacotron 2는 이러한 배경을 바탕으로 만들어졌습니다. *Tacotron의 시퀀스-투-시퀀스 모델을 기반으로 멜 스펙트로그램(mel spectrogram)을 생성하고, 이를 WaveNet이 받아서 시간 도메인 파형을 생성하는 방식*이죠. 쉽게 말해, Tacotron 2는 텍스트를 받아 사람처럼 자연스러운 음성으로 변환할 수 있습니다.

그렇다면, 여기서 멜 스펙트로그램이란 무엇일까요? 멜 스펙트로그램은 일반적인 스펙트로그램과 달리, 인간의 청각 시스템을 모방해 주파수 축을 비선형적으로 변환한 것입니다. 주파수가 높을수록 세밀한 차이를 덜 강조하고, 저주파수에서는 더 세밀한 차이를 강조합니다. 이는 음성의 명료도를 높이는 데 중요한 역할을 합니다.

Tacotron 2는 이 멜 스펙트로그램을 사용해, 기존의 복잡한 특징 엔지니어링 과정을 모두 제거했습니다. 그 결과, 더욱 간단하고 효율적인 방법으로 높은 품질의 음성을 생성할 수 있게 되었습니다.

<br>

## 4. Tacotron 2의 구조: 어떻게 작동할까?

Tacotron 2는 크게 두 가지 부분으로 나뉩니다. 첫 번째는 특징 예측 네트워크이고, 두 번째는 WaveNet 보코더입니다. 

1. 특징 예측 네트워크는 입력된 텍스트를 멜 스펙트로그램으로 변환합니다. 이 과정은 인코더와 디코더 구조로 이루어져 있습니다. 인코더는 텍스트를 숨겨진 표현으로 변환하고, 디코더는 이를 받아 멜 스펙트로그램을 예측합니다. 이때 어텐션 메커니즘이 사용되어, 디코더가 인코더의 출력을 효과적으로 요약하고 문맥을 반영할 수 있도록 도와줍니다.

2. WaveNet 보코더는 예측된 멜 스펙트로그램을 받아 시간 도메인 파형을 생성합니다. 즉, WaveNet이 보코더 역할을 하여, 우리가 들을 수 있는 음성 신호를 만들어내는 것입니다. 여기서 중요한 점은, Tacotron 2가 사용하는 WaveNet은 기존 WaveNet보다 더 간결하게 설계되었다는 것입니다. Tacotron 2는 12개의 층만 사용해도 고품질의 음성을 생성할 수 있는데, 이는 멜 스펙트로그램이 이미 오디오 신호의 저수준 표현을 제공하기 때문입니다. 따라서 더 적은 층을 사용해도 충분한 성능을 발휘할 수 있습니다.

<br>

## 5. Tacotron 2의 학습 과정

Tacotron 2는 두 단계로 학습됩니다. 먼저, 특징 예측 네트워크가 학습된 후, 이 네트워크의 출력(즉, 멜 스펙트로그램)을 바탕으로 WaveNet을 학습시키는 방식입니다. 

이때 teacher-forcing 기법이 사용됩니다. 이는 학습 중에 디코더가 이전 예측값이 아닌 실제값을 입력으로 사용하는 방법입니다. 이를 통해 모델이 더 빠르게 학습하고, 정확한 예측을 할 수 있게 합니다. 반면, 추론 단계에서는 이전 단계의 예측값을 입력으로 사용해 음성을 생성합니다.

Tacotron 2는 단일 여성 화자의 24.6시간 분량의 음성 데이터를 바탕으로 학습되었습니다. 중요한 점은, 이 데이터가 모두 정규화된 텍스트로 이루어졌다는 것입니다. 예를 들어, 숫자 "16"은 "sixteen"으로 풀어서 학습됩니다. 이를 통해 모델이 다양한 텍스트 입력에 대해 더 정확하게 음성을 생성할 수 있게 됩니다.

<br>

## 6. 실험 결과: Tacotron 2의 성능

Tacotron 2의 성능을 평가하기 위해 여러 가지 실험이 진행되었습니다. 그중 하나는 평균 의견 점수(MOS) 평가입니다. 이 평가에서는 사람들이 생성된 음성을 듣고, 1점에서 5점 사이로 음질을 평가합니다.

Tacotron 2는 4.53의 MOS를 기록했는데, 이는 전문가가 녹음한 실제 음성(4.58점)과 거의 동일한 수준입니다. 또한, 기존의 Tacotron이나 다른 음성 합성 시스템보다 훨씬 높은 점수를 기록했습니다.

하지만 Tacotron 2도 완벽하지는 않습니다. 평가자들은 때때로 발음 오류가 발생하거나, 문장의 운율이 부자연스러운 경우가 있다고 언급했습니다. 특히 이름이나 고유명사를 처리할 때 이러한 문제가 자주 나타났습니다.

<br>

## 7. 개선의 여지: 앞으로의 과제

Tacotron 2는 높은 성능을 자랑하지만, 여전히 개선의 여지가 남아 있습니다. 특히, 모델이 운율을 더 자연스럽게 처리할 수 있도록 개선할 필요가 있습니다. 또한, 발음 오류를 줄이기 위한 추가적인 연구가 필요합니다. 예를 들어, 이름이나 고유명사에 대해 더 잘 대응할 수 있도록 모델을 개선할 수 있을 것입니다.

또한, Tacotron 2는 도메인 외 텍스트에 대한 처리 성능도 평가되었습니다. 도메인 외 텍스트란, 학습 데이터에서 다루지 않았던 새로운 유형의 텍스트를 의미합니다. 이 실험에서 Tacotron 2는 4.148의 MOS를 기록했으며, 이는 WaveNet과 거의 동일한 수준이었습니다. 그러나 이 과정에서도 발음 문제가 발생할 수 있었습니다. 이는 Tacotron 2와 같은 종단간(end-to-end) 방식이 더 다양한 데이터를 학습해야 한다는 점을 시사합니다.

<br>

## 8. 결론: Tacotron 2가 가져온 혁신

Tacotron 2는 음성 합성 기술에서 중요한 돌파구를 마련한 시스템입니다. 이 시스템은 복잡한 음성 합성 과정에서 많은 부분을 신경망으로 대체함으로써, 더욱 간결하고 효과적으로 텍스트를 음성으로 변환할 수 있습니다. 특히 멜 스펙트로그램을 중간 표현으로 사용하여, WaveNet이 더 작은 규모로도 고품질 음성을 생성할 수 있게 했습니다.

Tacotron 2는 아직 완벽하지 않지만, 인간처럼 자연스러운 음성을 생성하는 데 한 걸음 더 가까워졌습니다. 앞으로 이 기술이 더욱 발전하여, 사람과의 대화나 음성 기반 서비스에서 큰 변화를 가져올 것입니다.

Tacotron 2는 우리가 텍스트에서 음성을 생성하는 방식을 혁신적으로 변화시키고 있으며, 인공지능이 점점 더 인간다운 목소리를 낼 수 있는 미래를 기대하게 만듭니다.