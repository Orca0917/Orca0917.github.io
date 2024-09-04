---
title: PyTorch로 구현하는 Tacotron2 - Encoder 텍스트를 음성으로 변환하는 첫 단계
date: 2024-09-05 00:14:00 +0900
categories: [Audio, Implementation]
tags: [tacotron2]     # TAG names should always be lowercase
author: moon
math: true
toc: true
---

이전 글에서 우리는 **Tacotron2** 모델을 위한 데이터셋 처리 과정에 대해 살펴보았습니다. 이제 데이터를 준비했으니, 모델의 **인코더(Encoder)** 부분을 깊이 있게 다루어 보겠습니다. Tacotron2는 텍스트 데이터를 음성으로 변환하는 모델로, 이 과정에서 인코더는 매우 중요한 역할을 담당합니다. 인코더는 입력된 텍스트를 숨겨진 특징(hidden features)으로 변환하여 디코더가 이를 음성 스펙트로그램으로 변환할 수 있게 해줍니다.

<br>

> Tacotron2의 Encoder 구현코드 원본 살펴보러 가기 [[링크]](https://pytorch.org/audio/2.2.0/_modules/torchaudio/models/tacotron2.html#Tacotron2)

<br>

## 1. Tacotron2 인코더 개요

Tacotron2의 인코더는 *문자 시퀀스를 숨겨진 특징 벡터로 변환하는 모듈*입니다. 인코더는 텍스트를 처리하고, 이를 디코더에서 사용할 수 있는 형태로 변환하는 데 필요한 모든 기능을 갖추고 있습니다. 인코더는 세 가지 주요 구성 요소로 나눌 수 있습니다:

1. **문자 임베딩**: 텍스트 데이터를 숫자로 표현하여 모델이 처리할 수 있는 형태로 변환합니다.
2. **컨볼루션 층**: 여러 개의 1D 컨볼루션 필터를 사용하여 텍스트에서 더 긴 맥락을 학습합니다.
3. **양방향 LSTM**: 마지막으로 컨볼루션 층의 출력을 받아 시퀀스의 양방향(앞에서 뒤, 뒤에서 앞) 정보를 모두 학습합니다.

이 과정을 통해 인코더는 텍스트에서 중요한 정보를 추출하여 디코더가 사용할 수 있는 특징 벡터를 생성합니다.

<br>

## 2. 문자 임베딩 (Character Embedding)

텍스트 데이터를 모델이 처리할 수 있는 숫자 벡터로 변환하는 것이 **문자 임베딩(Character Embedding)**입니다. 문자 임베딩은 각 문자를 고정된 차원의 벡터로 매핑하여, 텍스트 데이터를 숫자로 표현하게 합니다. Tacotron2에서는 이를 통해 각 문자가 고유한 벡터로 변환되며, 인코더는 이 벡터를 바탕으로 특징을 학습합니다.

![alt text](/assets/img/tacotron2-dataset/phoneme_sequence.svg){: style="display:block; margin:auto;" w="500"}

음소 시퀀스에 대해 padding을 더해주고 각 시퀀스의 값 하나하나를 256차원으로 확장하면 아래와 같은 3차원 텐서의 모습을 띄게 됩니다. 아래의 그림에서는 간결성을 위해 14차원을 모두 표현하진 않고, 시퀀스의 최대 길이를 8로 재설정하였습니다.

![Character embedding](/assets/img/tacotron2-encoder/chracter_embedding.svg){: style="display:block; margin:auto;" w="400"}

```python
# __init__ 내에서 선언
self.embedding = nn.Embedding(n_symbol, symbol_embedding_dim)

# forward 내에서 선언 - Convolution 연산을 위해 미리 transpose !!
embedded_inputs = self.embedding(tokens).transpose(1, 2)
```

- **`nn.Embedding`**: PyTorch에서 제공하는 임베딩 레이어로, 각 문자를 고정된 차원의 벡터로 변환합니다.
  - `n_symbol`: 사용할 문자(심볼)의 개수. 이는 모델이 처리할 수 있는 총 문자 수를 의미합니다.
  - `symbol_embedding_dim`: 각 문자를 매핑할 벡터의 차원입니다. Tacotron2에서는 이 임베딩 차원이 **512**로 설정되어 있습니다. 즉, 각 문자는 512차원 벡터로 변환됩니다.
  
- **`tokens`**: 입력 문자를 토큰으로 변환한 것입니다. 즉, 각 문자는 고유한 숫자(토큰)로 매핑되며, 이 숫자는 임베딩 레이어를 통해 벡터로 변환됩니다.

- **`transpose(1, 2)`**: 임베딩된 입력 벡터의 차원을 변경하여 [배치 크기, 시퀀스 길이, 임베딩 차원] 형태로 맞춥니다. 이는 이후에 컨볼루션 층에서 처리하기 위한 형식으로 변환하는 과정입니다.

Tacotron2의 문자 임베딩은 문자 데이터를 벡터로 표현하여, 텍스트의 의미를 벡터 공간에서 표현할 수 있게 합니다. 이 벡터 표현은 인코더의 다음 처리 단계에서 중요한 역할을 하며, 모델이 텍스트 정보를 효과적으로 학습할 수 있도록 돕습니다.

<br>

## 3. 인코더 구성 요소 살펴보기

```python
class _Encoder(nn.Module):
    """
    인코더는 문자 시퀀스를 숨겨진 특징 표현으로 변환한다.
    """
    
    def __init__(
        self,
        encoder_embedding_dim: int,
        encoder_n_convolution: int,
        encoder_kernel_size: int,
    ) -> None:
        super().__init__()

        self.convolutions = nn.ModuleList()
        for _ in range(encoder_n_convolution):
            conv_layer = nn.Sequential(
                _get_conv1d_layer(
                    encoder_embedding_dim,
                    encoder_embedding_dim,
                    kernel_size=encoder_kernel_size,
                    stride=1,
                    padding=int((encoder_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="relu",
                ),
                nn.BatchNorm1d(encoder_embedding_dim),
            )
            self.convolutions.append(conv_layer)

        self.lstm = nn.LSTM(
            encoder_embedding_dim,
            int(encoder_embedding_dim / 2),
            1,
            batch_first=True,
            bidirectional=True,
        )

        self.lstm.flatten_parameters()

    def forward(self, x: Tensor, input_lengths: Tensor) -> Tensor:
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        input_lengths = input_lengths.cpu()
        x = nn.utils.rnn.pack_padded_sequence(x, input_lengths, batch_first=True, enforce_sorted=False)

        outputs, _ = self.lstm(x)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        return outputs
```

<br>

### **컨볼루션 층 (Convolutional Layers)**
텍스트 데이터는 긴 맥락을 고려하여 처리해야 합니다. 컨볼루션 층은 이런 역할을 담당하며, 각 컨볼루션 필터는 **5개의 문자를 아우르는** 형태(아래 그림에서는 3개의 문자를 아우르는 형태로 묘사)로 설계됩니다. 즉, 입력된 텍스트 시퀀스를 일정 크기의 필터로 나누어 더 긴 맥락 정보를 학습합니다.

이 코드는 3개의 컨볼루션 층을 사용하여 텍스트 시퀀스를 처리하며, 각 층은 512개의 필터를 가지고 있습니다. 또한, 각 컨볼루션 층 뒤에는 **배치 정규화(Batch Normalization)**와 **ReLU 활성화 함수**가 적용됩니다.

![convolution](/assets/img/tacotron2-encoder/convolution.svg)

(위) 배치 중 **1개의 데이터**에 대해서만 컨볼루션 연산의 진행과정을 나타낸 것  
(아래) 배치 데이터에 대해 컨볼루션 연산의 진행과정을 나타낸 것

```python
for _ in range(encoder_n_convolution):
    conv_layer = nn.Sequential(
        _get_conv1d_layer(
            encoder_embedding_dim,
            encoder_embedding_dim,
            kernel_size=encoder_kernel_size,
            stride=1,
            padding=int((encoder_kernel_size - 1) / 2),
            dilation=1,
            w_init_gain="relu",
        ),
        nn.BatchNorm1d(encoder_embedding_dim),
    )
    self.convolutions.append(conv_layer)
```

<br>

### **양방향 LSTM (Bidirectional LSTM)**

컨볼루션 층을 통과한 텍스트 데이터는 이제 **양방향 LSTM**을 거칩니다. LSTM은 순차적 데이터를 처리하는데 뛰어난 성능을 보이는 구조로, 시퀀스의 양방향(순방향과 역방향)을 동시에 처리할 수 있습니다. 

![bidirectional lstm](/assets/img/tacotron2-encoder/bilstm.svg)

```python
self.lstm = nn.LSTM(
    encoder_embedding_dim,
    int(encoder_embedding_dim / 2),
    1,
    batch_first=True,
    bidirectional=True,
)
```

이 LSTM 층은 각 방향에 256개의 유닛을 가지고 있으며, 양방향으로 총 512개의 유닛을 통해 텍스트 시퀀스를 인코딩합니다. 이를 통해 텍스트의 앞뒤 문맥을 모두 반영한 특징 벡터를 추출할 수 있습니다.

LSTM의 성능을 최적화하기 위해 `self.lstm.flatten_parameters()`를 호출하여 파라미터들이 메모리에서 인접하게 배치되도록 합니다. 이는 학습 속도를 높이는 데 도움이 됩니다. [[참고]](/posts/LSTM_flatten_parameters)

<br>

### **순전파 과정 (Forward Process)**

순전파 과정에서는 입력된 텍스트 데이터를 차례로 컨볼루션 층과 LSTM 층을 통과시켜 최종 숨겨진 특징 벡터를 생성합니다.

```python
def forward(self, x: Tensor, input_lengths: Tensor) -> Tensor:
    for conv in self.convolutions:
        x = F.dropout(F.relu(conv(x)), 0.5, self.training)

    x = x.transpose(1, 2)
    
    input_lengths = input_lengths.cpu()
    x = nn.utils.rnn.pack_padded_sequence(x, input_lengths, batch_first=True, enforce_sorted=False)

    outputs, _ = self.lstm(x)
    outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

    return outputs
```

- **컨볼루션 층 통과**: 입력된 텍스트는 각 컨볼루션 층을 거치며 **ReLU 활성화 함수**와 **드롭아웃**(0.5 확률) 적용을 통해 활성화됩니다.
- **양방향 LSTM 통과**: 컨볼루션 층을 통과한 특징 벡터는 양방향 LSTM에 전달되어 시퀀스의 양쪽 맥락을 학습합니다.
- **출력**: 마지막으로 LSTM의 출력은 **패딩된 상태**로 복원되고, 숨겨진 특징 벡터를 반환합니다. [[참고]](/posts/pack_padded_sequence)

<br>

## 4. \_get\_conv1d\_layer: 컨볼루션 층 생성

컨볼루션 층을 생성하는 `_get_conv1d_layer` 함수는 주어진 하이퍼파라미터를 사용해 1D 컨볼루션 필터를 설정합니다. 이때 필터의 크기, 스트라이드, 패딩 등 다양한 설정을 할 수 있으며, 초기 가중치는 **Xavier 초기화**를 통해 설정됩니다.

```python
def _get_conv1d_layer(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 1,
    stride: int = 1,
    padding: Optional[Union[str, int, Tuple[int]]] = None,
    dilation: int = 1,
    bias: bool = True,
    w_init_gain: str = "linear"
) -> torch.nn.Conv1d:
    
    if padding is None:
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd")
        padding = int(dilation * (kernel_size - 1) / 2)

    conv1d = torch.nn.Conv1d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias
    )

    torch.nn.init.xavier_uniform_(conv1d.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    return conv1d
```

<br>

## 5. 결론

이 글에서는 **Tacotron2의 인코더**에 대해 자세히 알아보았습니다. 인코더는 텍스트 데이터를 모델이 처리할 수 있는 숨겨진 특징 벡터로 변환하는 중요한 역할을 합니다. *컨볼루션 층을 통해 텍스트의 긴 맥락을 학습하고, 양방향 LSTM을 통해 시퀀스의 앞뒤 문맥을 모두 고려한 특징 벡터를 생성합니다.*

Tacotron2 모델의 인코더는 텍스트 데이터를 효과적으로 처리하여 디코더가 이를 기반으로 음성 스펙트로그램을 예측할 수 있도록 돕습니다. 이 인코더 구조는 텍스트에서 음성으로 변환하는 작업에서 매우 중요한 역할을 합니다. 다음 글에서는 이 인코더에서 추출된 특징을 기반으로 **디코더가 어떻게 음성 스펙트로그램을 생성하는지** 살펴보겠습니다.

<br>

이 코드와 설명을 통해 Tacotron2의 인코더가 어떻게 텍스트를 처리하고, 숨겨진 특징 벡터를 생성하는지 이해할 수 있었길 바랍니다! 😊