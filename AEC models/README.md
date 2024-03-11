The proposed signal decoupling method is employed in conjunction with the following AEC models:

 * **ICCRN** : A neural network for monaural speech enhancement operating on the time-frequency cepstral domain. It is implemented by incorporating a cepstral frequency block into an in-place convolutional recurrent network. We change the input format of ICCRN to match the AEC task.
 * **ICRN*** : This model employs in-place convolution and channel-wise temporal modeling to preserve the near-end signal information. The asterisk $*$ indicates that the multi-task learning strategy employed in ICRN has been removed from this model.
 * **CRN*** : This model employs convolutional recurrent networks (CRN) and long short-term memory (LSTM) networks to separate the near-end speech from the microphone signal. The asterisk $*$ denotes that a convolution kernel of size (1, 3) is employed in this model.
