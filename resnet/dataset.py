from scipy import signal
from scipy.io import wavfile
from collections import defaultdict

from glob import glob
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
import torch
import numpy as np
import random

import sys
sys.path.append('../python_speech_features')
from python_speech_features import mfcc


def to_one_hot(x, len):
    vec = np.zeros(len)
    vec[x] = 1
    return vec


class WavDataset(Dataset):
    """Speech dataset."""

    def __init__(self, dataset_root, lst, augment=False):
        """
        Args:
            lst (attr_dict): txt file with params
            augment (bool): use augmentation
        """
        self.dataset_root = dataset_root
        self.augment = augment

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.train_samples = defaultdict(list)
        self.samples = []

        self.noises = \
            [c for c in glob(self.dataset_root + '_background_noise_mono_/*.wav')]

        self.labels = [
            'yes', 'no', 'up',
            'down', 'left', 'right',
            'on', 'off', 'stop', 'go']

        for line in open(lst):
            cat, sample = line.rstrip().split('/')
            try:
                if cat in self.labels:
                    self.train_samples[cat].append((cat, sample))
                else:
                    self.train_samples['unknown'].append((cat, sample))
                    self.samples.append(('unknown', sample))
            except KeyError:
                pass

        self.labels += ['unknown', 'silence']
        self.categories_map = {c: i for (i, c) in enumerate(self.labels)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cat = random.choice(list(self.train_samples.keys()))

        if cat is not 'silence':
            directory, sample = random.choice(self.train_samples[cat])
            (rate, sig) = wavfile.read(self.dataset_root + directory + '/' + sample)
        else:
            sig = wavfile.read(np.random.choice(self.noises))

        pad_sig = np.zeros(rate)

        dx = 0
        if sig.shape[0] - rate > 0:
            dx = np.random.randint(0, sig.shape[0] - rate)
        pad_sig[dx:dx + sig.shape[0]] = sig

        if self.augment:
            (_, noise_sig) = \
                wavfile.read(np.random.choice(self.noises))

            pad_noise = np.zeros(rate)
            dx = np.random.randint(0, noise_sig.shape[0] - rate)
            pad_noise[:rate] = noise_sig[dx:dx + rate]

            if random.random() > 0.5:
                pad_sig = pad_sig + pad_noise
            elif random.random() > 0.5:
                pad_sig = np.maximum(pad_sig, pad_noise)
            else:
                pad_sig = pad_sig

        f, t, spectogram = signal.spectrogram(pad_sig, rate)
        mf = mfcc(pad_sig, numcep=13, nfilt=26)

        # import matplotlib.pyplot as plt
        # plt.imshow(spectogram)
        # plt.show()

        # print(spectogram.shape)
        # print(mf.shape)
        y_true = \
            torch.FloatTensor(
                to_one_hot(self.categories_map[cat], len(self.labels))
            )

        return torch.FloatTensor(spectogram).unsqueeze(0),\
               torch.FloatTensor(mf).unsqueeze(0),\
               y_true


if __name__ == '__main__':
    ds = WavDataset(lst='../testing_list.txt')
    # print(ds[2])
    ds[1]
