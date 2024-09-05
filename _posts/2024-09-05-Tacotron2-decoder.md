---
title: PyTorch로 구현하는 Tacotron2 - Decoder 텍스트와 음성의 연결
date: 2024-09-05 18:11:00 +0900
categories: [Audio, Implementation]
tags: [tacotron2]     # TAG names should always be lowercase
author: moon
math: true
toc: true
---

Tacotron2의 디코더는 텍스트 정보를 기반으로 자연스러운 음성을 생성하는 핵심적인 역할을 합니다. 이 디코더는 *인코더에서 텍스트를 처리한 결과를 받아, 음성의 특징을 연속적으로 예측하고 생성하는 과정을 반복적으로 수행합니다.* 이를 위해 여러 단계의 모듈들이 협력하여 텍스트와 음성을 매끄럽게 연결합니다. 디코더의 목적은 단순히 음성을 예측하는 것을 넘어, 각 음성 프레임이 텍스트의 어느 부분과 관련이 있는지를 학습하고, 시간의 흐름에 따라 일관된 음성을 출력하는 것입니다.

<br>

## 1. 디코더의 입력: Prenet

디코더의 첫 단계는 `Prenet`입니다. 이 모듈은 정답 멜 스펙트로그램(Mel-Spectrogram)을 디코더가 처리할 수 있는 형태로 변환하는 역할을 합니다. 디코더는 단순히 텍스트와 음성을 바로 매칭하는 것이 아니라, 일련의 프레임을 예측하면서 생성된 음성 프레임을 사용하는 반복적인 구조를 가지고 있습니다. 이를 위해 `Prenet`은 입력 데이터를 두 개의 선형 레이어를 통해 압축된 표현으로 변환합니다. 각 레이어는 ReLU 활성화 함수와 dropout을 통해 모델이 학습 과정에서 과적합되지 않도록 하고, 더 일반화된 표현을 얻습니다.

![PreNet](/assets/img/tacotron2-decoder/prenet.svg)

```python
class _Prenet(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
        return x
```

이 단계에서 중요한 점은, `Prenet`이 입력된 음성 프레임을 더 저차원의 정보로 변환해, 디코더가 학습할 수 있는 형식으로 만들어준다는 것입니다. 이는 디코더가 이전에 예측한 음성 프레임을 다음 예측에 효과적으로 사용할 수 있도록 돕습니다.

<br>

## 2. Attention 메커니즘: 텍스트와 음성의 연결고리

음성 합성의 핵심은 디코더가 각 음성 프레임이 텍스트의 어느 부분과 관련이 있는지를 정확하게 찾는 것입니다. 이를 위해 사용되는 것이 `Attention` 메커니즘입니다. `Attention`은 현재의 디코더 상태와 인코더에서 전달된 전체 텍스트 정보를 비교하여, **텍스트 중 어떤 부분이 현재 생성 중인 음성 프레임과 가장 관련이 있는지 파악**합니다. 이 과정에서 `Attention RNN`은 과거의 디코더 상태를 기억하고, 그 정보를 활용해 새로운 음성 프레임을 예측합니다.

![attention rnn input](/assets/img/tacotron2-decoder/attention-rnn-input.svg)

```python
# _Decoder class의 decode() 중 일부
cell_input = torch.cat((decoder_input, attention_context), -1)
attention_hidden, attention_cell = self.attention_rnn(cell_input, (attention_hidden, attention_cell))
```

![attention rnn](/assets/img/tacotron2-decoder/attention-rnn.svg)

```python
class _Attention(nn.Module):
    def forward(
        self,
        attention_hidden_state: Tensor, 
        memory: Tensor, 
        processed_memory: Tensor, 
        attention_weights_cat: Tensor, 
        mask: Tensor
    ) -> Tuple[Tensor, Tensor]:

        alignment = self._get_alignment_energies(attention_hidden_state, processed_memory, attention_weights_cat)
        alignment = alignment.masked_fill(mask, self.score_mask_value)
        attention_weights = F.softmax(alignment, dim=1)

        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory).squeeze(1)
        return attention_context, attention_weights
```

이 과정에서 디코더는 **현재의 음성 프레임을 만들기 위해 텍스트의 어느 부분을 "집중"해야 하는지를 학습**하게 됩니다. `Attention`은 음성 생성 과정에서의 중심적인 역할을 하며, 텍스트 시퀀스 중에서 가장 중요한 부분을 선택하는 능력을 제공합니다. 이를 통해 음성과 텍스트가 자연스럽게 정렬되며, 텍스트의 어떤 부분이 음성의 어느 부분과 대응되는지 배울 수 있습니다.

<br>

## 3. Decoder RNN: 시간의 흐름을 유지하는 핵심

디코더의 가장 중요한 부분 중 하나는 `Decoder RNN`입니다. 이 RNN은 `Attention`에서 선택된 텍스트 정보를 바탕으로 음성 프레임을 순차적으로 예측합니다. 이때 LSTM(Long Short-Term Memory) 구조를 사용하여 이전의 예측과 현재의 상태를 기반으로 새로운 음성 프레임을 생성합니다. `Decoder RNN`은 시간의 흐름에 따라 연속적인 음성을 생성하는 데 중요한 역할을 합니다.

```python
class _Decoder(nn.Module):
    def decode(
        self,
        decoder_input: Tensor,
        attention_hidden: Tensor,
        attention_cell: Tensor,
        decoder_hidden: Tensor,
        decoder_cell: Tensor,
        attention_weights: Tensor,
        attention_weights_cum: Tensor,
        attention_context: Tensor,
        memory: Tensor,
        processed_memory: Tensor,
        mask: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        
        cell_input = torch.cat((decoder_input, attention_context), -1)
        attention_hidden, attention_cell = self.attention_rnn(cell_input, (attention_hidden, attention_cell))
        attention_hidden = F.dropout(attention_hidden, self.attention_dropout, self.training)

        attention_weights_cat = torch.cat((attention_weights.unsqueeze(1), attention_weights_cum.unsqueeze(1)), dim=1)
        attention_context, attention_weights = self.attention_layer(
            attention_hidden, memory, processed_memory, attention_weights_cat, mask
        )

        decoder_input = torch.cat((attention_hidden, attention_context), -1)
        decoder_hidden, decoder_cell = self.decoder_rnn(decoder_input, (decoder_hidden, decoder_cell))
        decoder_hidden = F.dropout(decoder_hidden, self.decoder_dropout, self.training)

        decoder_hidden_attention_context = torch.cat((decoder_hidden, attention_context), dim=1)
        decoder_output = self.linear_projection(decoder_hidden_attention_context)

        gate_prediction = self.gate_layer(decoder_hidden_attention_context)

        return decoder_output, gate_prediction, attention_hidden, attention_cell, decoder_hidden, decoder_cell, attention_weights, attention_weights_cum, attention_context
```

`Decoder RNN`은 입력된 텍스트와 과거 디코더 상태를 기반으로 다음 timestep에서의 음성을 생성하며, 이전에 생성된 음성 프레임과 텍스트 사이의 관계를 유지하면서 연속적인 음성 프레임을 만들어냅니다.

<br>

## 4. Linear Projection과 Gate Layer: 최종 음성 프레임 생성

`Decoder RNN`이 예측한 결과는 `Linear Projection`을 통해 최종 음성 프레임으로 변환됩니다. 이 단계는 디코더가 예측한 정보를 실제로 사람이 들을 수 있는 음성으로 변환하는 과정입니다. 각 timestep에서의 RNN 출력과 `Attention Context`를 결합하여 선형 변환을 수행하고, 이를 통해 멜 스펙트로그램의 프레임이 생성됩니다.

```python
decoder_hidden_attention_context = torch.cat((decoder_hidden, attention_context), dim=1)
decoder_output = self.linear_projection(decoder_hidden_attention_context)
```

또한, 음성 프레임을 언제 멈출지 결정하는 `Gate Layer`도 존재합니다. `Gate Layer`는 디코더가 음성 생성을 멈출 지점을 예측하여 디코딩 과정을 제어합니다. 이 단계는 고정된 길이의 음성을 예측하지 않고, 문장의 끝에 도달했을 때 멈추도록 돕습니다.

<br>

## 5. 결론

Tacotron2의 디코더는 복잡한 과정을 통해 자연스러운 음성을 생성하는 데 집중합니다. `Prenet`을 통해 입력을 준비하고, `Attention`을 사용해 텍스트와 음성을 연결하며, `Decoder RNN`으로 시간의 흐름을 유지하면서 연속적인 음성 프레임을 생성합니다. 최종적으로 `Linear Projection`을 통해 음성 프레임이 출력되고, `Gate Layer`로 디코딩이 언제 끝날지를 결정합니다.

이 모든 과정이 협력하여 텍스트로부터 자연스럽고 연속적인 음성을 합성하는 것이 디코더의 핵심 역할입니다.