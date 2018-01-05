# Wave-net branch of the tf-speech-recon framework

The architectures, that we tried their leaderboard scores are presented in the table below:

| Architecture | Public score |
| :----------- | -----------: |
| wave-net<sup id="1">[[1]](#1)</sup> | 0.86         |

<b id="[1]">1</b> Footnote content here. [↩](#1)

Implementation of the dilated convolutional neural network for the kaggle speech commands recognition challenge.
Current best model achieves 96.3% on the Google's test dataset and 86% on the leaderboard.

Use model_configs/best_wave_net.config to replicate the results.

Idea of the architecture is based on the original wave-net paper: https://arxiv.org/pdf/1609.03499.pdf

Implementation follows the logic of speech-to-text-wave-net from https://github.com/buriburisuri/speech-to-text-wavenet

Some ideas for improvement:
1. Investigate main causes of errors on LB by looking into some random utterances and comparing their labels to the outputs of the model
2. Incorporate more data augmentation like stretching, etc.
3. Adapt adversarial training to this model
4. Substitute normal convolutions with depthwise separable convolutions for efficiency
5. Try to lighten the model by decreasing number of channels and dilated blocks

