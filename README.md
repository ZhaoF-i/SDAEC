# SDAEC: Signal Decoupling for Advancing Acoustic Echo Cancellation
## Announcement

This GitHub account was created to comply with INTERSPEECH's double-blind regulations, and it will be relocated to a different address once the acceptance results are published.

## Introduction
<div align="center">
<img src="https://github.com/ZhaoF-i/SDAEC/blob/main/pictures/LAEC_2.png" alt="LAEC" width="660" height="300">
</div>

  The diagram of the ideal single-channel AEC system is illustrated in the figure above. The microphone signal $d(n)$ is a mixture of echo signal $y(n)$ and near-end signal $s(n)$.
If the environmental noise is not considered, the microphone signal in the time domain can be formulated as follows:

$$
    d(n) = y(n) + s(n) \tag{1}
$$

The acoustic echo $y(n)$ is the convolution of the source signal $x(n)$ with the room impulse response (RIR) $h(n)$. 
However, in real-world scenarios, individuals often adjust the speaker volume based on the environmental conditions, causing the far-end signal $x(n)$ to be modified to $x^\prime(n)$ through the speaker output. Simultaneously, due to the inherent limitations of the speaker itself, nonlinear distortion results in the final output of the speaker being $x^\prime_{nl}(n)$. Then formula (1) can be written as:

$$
    d(n) = x^\prime_{nl}(n) * h(n) + s(n) \tag{2}
$$

where $*$ represents the convolution operation. We reformulate formula (2) into the time-frequency domain by applying the short-time Fourier transform (STFT) as:

$$
    D[t, f] =  \sum_k^K H[k,f]X^\prime_{nl}[t-k, f] + S[t, f]
$$

where $S[t, f]$, $D[t, f]$ and $X^\prime_{nl}[t, f]$ represent the near-end signal, microphone signal, and far-end signal played by speaker at the frame $t$ and frequency $f$, respectively, and $H[k, f]$ is the echo path. Here, $K$ stands for the number of taps.

<div align="center">
<img src="https://github.com/ZhaoF-i/SDAEC/blob/main/pictures/alpha_efficiency3_00.png" alt="alpha" width="660" height="480">
</div>

  We propose a signal decoupling-based approach. When employing a neural network for AEC tasks, the network implicitly learns the echo path from the input far-end reference signal and microphone signal to achieve echo elimination. Our proposed method decouples the energy component in the echo path from the input reference signal and microphone signal, allowing for separate predictions. We denote this energy component as the energy scaling factor $alpha$. The original far-end signal is replaced by the product of multiplying the far-end reference signal with the energy scaling factor, and this product, along with the microphone signal, is used as input to the echo cancellation network. The picture above depicts a comparison between the unprocessed signal (mix) and the outcomes obtained by employing the traditional adaptive filtering method frequency domain Kalman filter (FDKF), as well as the reference signal multiplied by the energy scaling factor $alpha$ and subsequently processed through FDKF. The variants incorporating $alpha$ showcased in the figure demonstrate further enhancement in the performance of FDKF, thereby initially validating the necessity and feasibility of our proposed method. Additionally, this observation supports our utilization of signal decoupling in deep learning methods. 

## Methodology
### Overview

| ![Image 1](https://github.com/ZhaoF-i/SDAEC/blob/main/pictures/overview.png) | ![Image 2](https://github.com/ZhaoF-i/SDAEC/blob/main/pictures/SDM.png) |
|:---:|:---:|



  As shown above on the left, the proposed approach comprises two main components: a signal decoupling module and an AEC module. First, the far-end speech signal and the microphone signal are fed into the signal decoupling module to estimate an energy scaling factor. This energy scaling factor is multiplied with the far-end speech signal to obtain a modified input. Concurrently, the microphone signal is also provided as input to the AEC module, along with the modified far-end input. The AEC module then predicts the near-end speech signal by suppressing the echo components.

### Signal decoupling module
  The implementation details of the signal decoupling module are shown in the picture on the right above. We apply a time-unfolding operation to the input far-end signal X and microphone signal D in the time-frequency domain. This operation incorporates information from the preceding (n-1) frames across the time dimension, enabling a more accurate prediction of the energy scaling factor. Subsequently, we apply the absolute value operation and the square operation separately to the time-unfolded signals, obtaining their respective energy information. These two signals are then stacked together, and an addition operation is performed along the frequency dimension to concentrate the energy in the time dimension. The concentrated energy representation is passed through two linear layers, which map the information from n frames to a single frame and analyze the energy scaling factors from the two signals. Another absolute value operation is then applied to the output, ensuring that the resulting signal remains non-negative, thereby computing the final energy scaling factor $alpha$. Ultimately, $X$ is multiplied by $alpha$ to obtain the modified signal $X^\prime$.

### AEC module
The proposed signal decoupling method is employed in conjunction with the following AEC models:

 * **ICCRN** : A neural network for monaural speech enhancement operating on the time-frequency cepstral domain. It is implemented by incorporating a cepstral frequency block into an in-place convolutional recurrent network. We change the input format of ICCRN to match the AEC task.
 * **ICRN*** : This model employs in-place convolution and channel-wise temporal modeling to preserve the near-end signal information. The asterisk $*$ indicates that the multi-task learning strategy employed in ICRN has been removed from this model.
 * **CRN*** : This model employs convolutional recurrent networks (CRN) and long short-term memory (LSTM) networks to separate the near-end speech from the microphone signal. The asterisk $*$ denotes that a convolution kernel of size (1, 3) is employed in this model.

## Result  

|              |          |           |          |           |           |          |           |          |           |           |          |           |          |           |           |
| :----------: | :------: | :-------: | :------: | :-------: | :-------: | :------: | :-------: | :------: | :-------: | :-------: | :------: | :-------: | :------: | :-------: | :-------: |
|     SER      |   -10    |           |          |           |           |    0     |           |          |           |           |    10    |           |          |           |           |
|    Scenes    |    DT    |           |  ST NE   |           |   ST FE   |    DT    |           |  ST NE   |           |   ST FE   |    DT    |           |  ST NE   |           |   ST FE   |
| Model/Metric |   PESQ   |    SDR    |   PESQ   |    SDR    |   ERLE    |   PESQ   |    SDR    |   PESQ   |    SDR    |   ERLE    |   PESQ   |    SDR    |   PESQ   |    SDR    |   ERLE    |
|     mix      |   1.57   |    -10    |   4.5    | $+\infty$ |    --     |   2.02   |     0     |   4.5    | $+\infty$ |    --     |   2.62   |    10     |   4.5    | $+\infty$ |    --     |
|    ICCRN     |   2.3    |   9.57    | **4.5**  |   20.94   |   42.63   |   2.98   |   14.63   | **4.5**  |   20.94   |   43.08   |   3.51   |   18.25   | **4.5**  |   20.94   |   41.32   |
| **ICCRN+SD** | **2.36** | **10.18** | **4.5**  | **24.25** | **44.16** | **3.05** | **15.57** | **4.5**  | **24.25** | **43.57** | **3.55** | **19.94** | **4.5**  | **24.25** | **41.97** |
|     ICRN     |   2.03   |   0.07    |   3.39   |   0.08    | **63.12** |   2.62   |   0.08    |   3.39   |   0.08    | **52.95** |   3.02   |   0.08    |   3.39   |   0.08    | **42.83** |
| **ICRN+SD**  | **2.17** | **0.14**  | **3.88** | **0.17**  |   60.37   | **2.81** | **0.17**  | **3.88** | **0.17**  |   50.36   | **3.27** | **0.18**  | **3.88** | **0.17**  |   40.33   |
|     CRN      |   1.79   | **3.06**  | **4.42** |   5.62    |   30.47   |   2.51   |   4.42    | **4.42** |   5.62    |   20.45   |   3.13   |   4.47    | **4.42** |   5.62    |   10.46   |
|  **CRN+SD**  | **1.81** |   2.92    |   4.4    | **8.32**  | **33.48** | **2.56** | **5.21**  |   4.4    | **8.32**  | **23.54** | **3.17** | **5.98**  |   4.4    | **8.32**  | **13.56** |


The effectiveness of the proposed signal decoupling approach is evaluated on three AEC models. As illustrated in the above table, experiments are conducted under signal-to-echo ratios (SERs) of -10, 0, and 10 dB, encompassing simulation scenarios of double-talk, near-end single-talk, and far-end single-talk conditions. 

In summary, the signal decoupling modules (+SD) consistently improve the performance of the AEC models in terms of PESQ (speech quality), SDR (signal-to-distortion ratio), and ERLE (echo suppression) metrics, especially in double-talk and near-end single talk scenarios. The improvements are more pronounced for the ICCRN and CRN models compared to the ICRN model.

## Examples


|   |80 DT|41 DT|62 DT|0 DT|
|:---:|:---:|:---:|:---:|:---:|
| mic | ![Image 1](https://github.com/ZhaoF-i/SDAEC/blob/main/wav_samples/80DT_mic.png) | ![Image 2](https://github.com/ZhaoF-i/SDAEC/blob/main/wav_samples/41DT_mic.png) |![Image 2](https://github.com/ZhaoF-i/SDAEC/blob/main/wav_samples/62DT_mic.png) |![Image 2](https://github.com/ZhaoF-i/SDAEC/blob/main/wav_samples/0DT_mic.png) | 
| near | ![Image 1](https://github.com/ZhaoF-i/SDAEC/blob/main/wav_samples/80DT_near.png) | ![Image 2](https://github.com/ZhaoF-i/SDAEC/blob/main/wav_samples/41DT_near.png) |![Image 2](https://github.com/ZhaoF-i/SDAEC/blob/main/wav_samples/62DT_near.png) |![Image 2](https://github.com/ZhaoF-i/SDAEC/blob/main/wav_samples/0DT_near.png) | 
| far | ![Image 1](https://github.com/ZhaoF-i/SDAEC/blob/main/wav_samples/80DT_far.png) | ![Image 2](https://github.com/ZhaoF-i/SDAEC/blob/main/wav_samples/41DT_far.png) |![Image 2](https://github.com/ZhaoF-i/SDAEC/blob/main/wav_samples/62DT_far.png) |![Image 2](https://github.com/ZhaoF-i/SDAEC/blob/main/wav_samples/0DT_far.png) | 
| far*alpha | ![Image 1](https://github.com/ZhaoF-i/SDAEC/blob/main/wav_samples/80DT_farAlpha.png) | ![Image 2](https://github.com/ZhaoF-i/SDAEC/blob/main/wav_samples/41DT_farAlpha.png) |![Image 2](https://github.com/ZhaoF-i/SDAEC/blob/main/wav_samples/62DT_farAlpha.png) |![Image 2](https://github.com/ZhaoF-i/SDAEC/blob/main/wav_samples/0DT_farAlpha.png) | 
| ICCRN | ![Image 1](https://github.com/ZhaoF-i/SDAEC/blob/main/wav_samples/80DT_ICCRN_est.png) | ![Image 2](https://github.com/ZhaoF-i/SDAEC/blob/main/wav_samples/41DT_ICCRN_est.png) |![Image 2](https://github.com/ZhaoF-i/SDAEC/blob/main/wav_samples/62DT_ICCRN_est.png) |![Image 2](https://github.com/ZhaoF-i/SDAEC/blob/main/wav_samples/0DT_ICCRN_est.png) | 
| ICCRN+SD | ![Image 1](https://github.com/ZhaoF-i/SDAEC/blob/main/wav_samples/80DT_ICCRN+SD_est.png) | ![Image 2](https://github.com/ZhaoF-i/SDAEC/blob/main/wav_samples/41DT_ICCRN+SD_est.png) |![Image 2](https://github.com/ZhaoF-i/SDAEC/blob/main/wav_samples/62DT_ICCRN+SD_est.png) |![Image 2](https://github.com/ZhaoF-i/SDAEC/blob/main/wav_samples/0DT_ICCRN+SD_est.png) | 



