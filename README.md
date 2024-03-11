# SDAEC: Signal Decoupling for Advancing Acoustic Echo Cancellation
## Announcement

This GitHub account was created to comply with INTERSPEECH's double-blind regulations, and it will be relocated to a different address once the acceptance results are published.

## Introduction

The diagram of the ideal single-channel AEC system is illustrated in ![LAEC](pictures/LAEC_2.png). The microphone signal $d(n)$ is a mixture of echo signal $y(n)$ and near-end signal $s(n)$.
If the environmental noise is not considered, the microphone signal in the time domain can be formulated as follows:
\begin{equation}
    d(n) = y(n) + s(n)
    \label{1}
\end{equation}
The acoustic echo y(n) is the convolution of the source signal x(n) with the room impulse response (RIR) \cite{habets2006room} h(n). 
However, in real-world scenarios, individuals often adjust the speaker volume based on the environmental conditions, causing the far-end signal $x(n)$ to be modified to $x^\prime(n)$ through the speaker output. Simultaneously, due to the inherent limitations of the speaker itself, nonlinear distortion results in the final output of the speaker being $x^\prime_{nl}(n)$. Then \autoref{1} can be written as:
\begin{equation}
    d(n) = x^\prime_{nl}(n) * h(n) + s(n)
    \label{2}
\end{equation}
where $*$ represents the convolution operation. We reformulate \autoref{2} into the time-frequency domain by applying the short-time Fourier transform (STFT) as:
\begin{equation}
    D[t, f] =  \sum_k^K H[k,f]X^\prime_{nl}[t-k, f] + S[t, f]
    \label{3}
\end{equation}
where $S[t, f]$, $D[t, f]$ and $X^\prime_{nl}[t, f]$ represent the near-end signal, microphone signal, and far-end signal played by speaker at the frame $t$ and frequency $f$, respectively, and $H[k, f]$ is the echo path. Here, $K$ stands for the number of taps.

we propose a signal decoupling-based approach. When employing a neural network for AEC tasks, the network implicitly learns the echo path from the input far-end reference signal and microphone signal to achieve echo elimination. Our proposed method decouples the energy component in the echo path from the input reference signal and microphone signal, allowing for separate predictions. We denote this energy component as the energy scaling factor $alpha$. The original far-end signal is replaced by the product of multiplying the far-end reference signal with the energy scaling factor, and this product, along with the microphone signal, is used as input to the echo cancellation network. \autoref{fig:alpha_effiency} depicts a comparison between the unprocessed signal (mix) and the outcomes obtained by employing the traditional adaptive filtering method frequency domain Kalman filter (FDKF), as well as the reference signal multiplied by the energy scaling factor $alpha$ and subsequently processed through FDKF. The variants incorporating $alpha$ showcased in the figure demonstrate further enhancement in the performance of FDKF, thereby initially validating the necessity and feasibility of our proposed method. Additionally, this observation supports our utilization of signal decoupling in deep learning methods. 
