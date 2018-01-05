# tf-speech-recon
The architectures, that we tried their leaderboard scores are presented in the table below. Scores are presented in decreasing order:

| Architecture | Public score |
| :----------- | -----------: |
| wave-net<sup id="1">[[1]](#1)</sup> | 0.86         |
| ds-cnn<sup id="2">[[2]](#2)</sup>   | 0.83         |
| cnn on raw audio<sup id="3">[[3]](#3)</sup> | 0.83 |
| crnn<sup id="2">[[2]](#2)</sup> | 0.83 |
| gru-rnn<sup id="2">[[2]](#2)</sup> | 0.83 |
| lace | 0.82 |

References:

<b id="[1]">1 - </b> Oord A. et al. Wavenet: A generative model for raw audio //arXiv preprint arXiv:1609.03499. – 2016. [↩](#1)

<b id="[2]">2 - </b> Zhang Y. et al. Hello Edge: Keyword Spotting on Microcontrollers //arXiv preprint arXiv:1711.07128. – 2017.  [↩](#2)

<b id="[3]">3 - </b> Dai W. et al. Very deep convolutional neural networks for raw waveforms //Acoustics, Speech and Signal Processing (ICASSP), 2017 IEEE International Conference on. – IEEE, 2017. – С. 421-425. [↩](#3)

Speech recognition Kaggle repo

Some ideas:

1. seq2seq attention
2. feature combination (raw + STFT + MFCC)
3. adversarial learning
4. depth-wise convolution
5. CTC (connectionist temporal classification)
6. model combination
7. manual labling for generating new testdata (~500 samples)
