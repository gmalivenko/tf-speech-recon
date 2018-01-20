# tf-speech-recognition-challenge
This repository is based on Tensorflow Speech Commands framework (https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/speech_commands)

Presented code was used to get into top 17% in Kaggle Speech Recognition Challenge (https://www.kaggle.com/c/tensorflow-speech-recognition-challenge). Architectures that we tried and their corresponding private leaderboard scores are presented in the table below.

| Architecture | Private score |
| :----------- | -----------: |
| wave-net<sup id="1">[[1]](#1)</sup> | 0.87137         |
| ds-cnn<sup id="2">[[2]](#2)</sup>   | 0.84600         |
| cnn on raw audio<sup id="3">[[3]](#3)</sup> | 0.83895 |
| lace<sup id="4">[[4]](#4)</sup> | 0.83319 |
| crnn<sup id="2">[[2]](#2)</sup> | 0.83143 |
| gru-rnn<sup id="2">[[2]](#2)</sup> | 0.82978 |

To replicate best result modify pathes and run launch_training_locally.sh.


References:

<b id="[1]">1 - </b> Oord A. et al. Wavenet: A generative model for raw audio //arXiv preprint arXiv:1609.03499. – 2016. [↩](#1)

<b id="[2]">2 - </b> Zhang Y. et al. Hello Edge: Keyword Spotting on Microcontrollers //arXiv preprint arXiv:1711.07128. – 2017.  [↩](#2)

<b id="[3]">3 - </b> Dai W. et al. Very deep convolutional neural networks for raw waveforms //Acoustics, Speech and Signal Processing (ICASSP), 2017 IEEE International Conference on. – IEEE, 2017. – С. 421-425. [↩](#3)

<b id="[4]">4 - </b> Xiong W. et al. The Microsoft Conversational Speech Recognition System //Acoustics, Speech and Signal Processing (ICASSP), 2017 IEEE International Conference on. – IEEE, 2017. [↩](#4)



