---
title: PyTorch로 구현하는 Tacotron2 - 데이터셋 처리
date: 2024-09-04 02:39:00 +0900
categories: [Audio, Implementation]
tags: [tacotron2]     # TAG names should always be lowercase
author: moon
math: true
toc: true
---

음성 합성 분야에서 Tacotron2는 매우 인기 있는 모델입니다. 이 모델은 텍스트 데이터를 받아 음성으로 변환하는데, 그 과정에서 텍스트를 음소(phoneme)로 변환하고, 음성 데이터를 멜 스펙트로그램(mel spectrogram)으로 변환하는 등 다양한 전처리 과정이 필요합니다. 이번 글에서는 이러한 과정을 PyTorch 코드로 구현한 예제를 단계별로 설명하겠습니다.

<br>

![alt text](/assets/img/tacotron2-dataset/raw-dataset.svg){: style="display:block; margin:auto;" w="650"}

<br>

## 1. **텍스트를 음소로 변환하는 함수 - `phonemizer`**
음성 합성을 위해서는 텍스트 데이터를 음소로 변환해야 합니다. 이는 텍스트가 음성으로 어떻게 발음될지 알려주는 과정입니다.

![alt text](/assets/img/tacotron2-dataset/phoneme_sequence.svg){: style="display:block; margin:auto;" w="500"}

```python
def phonemizer(
    text: str,
    processor: Tacotron2TTSBundle.TextProcessor
) -> Tuple[Tensor, Tensor]:

    processed, lengths = processor(text)
    return processed, lengths
```

- **입력**: 텍스트(`text`)와 이를 음소로 변환할 전처리기(`processor`)를 받습니다.
- **출력**: 텍스트를 음소로 변환한 결과(`processed`)와 그 길이(`lengths`)를 반환합니다.

즉, 이 함수는 텍스트를 음소로 변환하여 모델이 사용할 수 있도록 준비해줍니다.

<br>

## 2. **오디오를 멜 스펙트로그램으로 변환하는 함수 - `wav2mel`**
오디오 데이터를 멜 스펙트로그램으로 변환하는 작업도 중요한데, 멜 스펙트로그램은 음성 신호의 주파수 특성을 시각적으로 표현한 형태로, 모델에서 자주 사용되는 형식입니다.

![alt text](/assets/img/tacotron2-dataset/mel.svg){: style="display:block; margin:auto;" w="300"}

```python
def wav2mel(
        audio: Tensor = None,
        sample_rate: int = 22050,
) -> Tensor:
    frame_size = int(0.05 * sample_rate)
    frame_hop = int(0.0125 * sample_rate)

    mel_specgram_transform = transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_mels=80,
        n_fft=frame_size,
        hop_length=frame_hop,
        f_min=125,
        f_max=7600,
        window_fn=torch.hann_window
    )

    mel_specgram = mel_specgram_transform(audio)
    mel_specgram = torch.clamp(mel_specgram, min=0.01)
    mel_specgram = torch.log(mel_specgram)

    return mel_specgram, torch.LongTensor([mel_specgram.size(-1)])
```

- **입력**: 오디오 데이터(`audio`)와 샘플링 레이트(`sample_rate`).
- **출력**: 멜 스펙트로그램(`mel_specgram`)과 그 길이.
  
이 함수는 오디오 데이터를 멜 스펙트로그램으로 변환하고, 그 데이터를 모델이 사용할 수 있도록 만들어 줍니다.

<br>

## 3. **데이터셋을 구성하는 클래스 - `Tacotron2Dataset`**
`Tacotron2Dataset` 클래스는 PyTorch의 `Dataset` 클래스를 상속받아, 데이터를 모델이 학습할 수 있도록 준비하는 역할을 합니다. 

```python
class Tacotron2Dataset(Dataset):
    def __init__(self, dataset: Dataset = None, text_processor: Tacotron2TTSBundle.TextProcessor = None) -> None:
        self.dataset = dataset
        self.processor = text_processor

    def __getitem__(self, index):
        processed_text, processed_text_length = phonemizer(self.dataset[index][3], self.processor)
        processed_mel_specgram, processed_mel_specgram_length = wav2mel(self.dataset[index][0].squeeze(), self.dataset[index][1])

        return (
            processed_text.squeeze(),
            processed_text_length,
            processed_mel_specgram,
            processed_mel_specgram_length
        )

    def __len__(self):
        return len(self.dataset)
```

<br>

## 4. **배치를 구성하는 클래스 - `Tacotron2Collate`**
*모델 학습 시 데이터를 배치(batch) 단위로 처리하는데, 각 데이터 샘플의 길이가 다를 수 있습니다. 이때 길이가 다른 샘플들을 맞춰주기 위해 패딩(padding)을 사용해야 합니다.* `Tacotron2Collate` 클래스는 이런 패딩 작업을 담당합니다.

![alt text](/assets/img/tacotron2-dataset/padded_phoneme.svg){: style="display:block; margin:auto;" w="500"}

![alt text](/assets/img/tacotron2-dataset/padded_mel.svg){: style="display:block; margin:auto;" w="600"}

```python
class Tacotron2Collate():
    def __init__(self):
        ...

    def __call__(self, batch: List[Tensor]):
        B = len(batch)

        text_lengths = torch.LongTensor([text_len for _, text_len, _, _ in batch])
        text_lengths_sorted, ids_sorted_decreasing = torch.sort(text_lengths, descending=True, dim=0)
        max_length = text_lengths_sorted[0]

        text_padded = torch.LongTensor(B, max_length)
        text_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        n_mels = batch[0][2].size(0)
        max_melspec_len = max([melspec.size(1) for _, _, melspec, _ in batch])
        
        mel_padded = torch.Tensor(B, n_mels, max_melspec_len)
        mel_padded.zero_()
        gate_padded = torch.Tensor(B, max_melspec_len)
        gate_padded.zero_()
        mel_lengths = torch.LongTensor(B)

        for i in range(len(ids_sorted_decreasing)):
            melspec = batch[ids_sorted_decreasing[i]][2]
            mel_padded[i, :, :melspec.size(1)] = melspec
            gate_padded[i, melspec.size(1) - 1:] = 1
            mel_lengths[i] = melspec.size(1)
        
        return (
            text_padded,
            text_lengths,
            mel_padded,
            mel_lengths,
            gate_padded
        )
```

- **입력**: 여러 샘플이 포함된 배치.
- **출력**: 길이가 맞춰진(패딩된) 텍스트, 텍스트 길이, 멜 스펙트로그램, 멜 스펙트로그램 길이, 종료 토큰.


<br>

## 전체 흐름 정리

1. **텍스트를 음소로 변환**: `phonemizer` 함수가 텍스트를 음소로 변환합니다.
2. **오디오를 멜 스펙트로그램으로 변환**: `wav2mel` 함수가 오디오 데이터를 멜 스펙트로그램으로 변환합니다.
3. **데이터셋 클래스**: `Tacotron2Dataset` 클래스는 텍스트와 오디오 데이터를 변환하고, 이를 모델에 사용할 수 있게 처리합니다.
4. **배치 구성**: `Tacotron2Collate` 클래스는 모델 학습을 위해 배치 단위로 데이터를 정리하고, 패딩 작업을 통해 길이를 맞춰줍니다.

<br>

## 결론

이번 글에서는 음성 합성 모델인 Tacotron2에서 데이터를 처리하는 핵심적인 부분을 다루어 보았습니다. 텍스트를 음소로 변환하고, 오디오를 멜 스펙트로그램으로 변환한 뒤, 이를 학습 가능한 배치로 만들어주는 일련의 과정은 음성 합성 모델 학습에 필수적이며 다른 곳에서도 충분히 적용해볼 수 있습니다. 이 글을 통해 Tacotron2 모델을 위한 데이터 전처리 작업을 쉽게 이해할 수 있기를 바랍니다. ✨