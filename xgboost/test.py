from glob import glob
import os
import scipy.io.wavfile as wav
from catboost import CatBoostClassifier
from collections import defaultdict
import numpy as np
import sys

sys.path.append('../python_speech_features')
from python_speech_features import *


def get_features(file, pad_to=(100, 13)):
    (rate, sig) = wav.read(file)
    mfcc_feat = np.array(mfcc(sig, rate))
    # d_mfcc_feat = delta(mfcc_feat, 2)
    # fbank_feat = logfbank(sig, rate)
    # mfcc_feat = np.reshape(mfcc_feat, -1)
    result = np.zeros(pad_to, dtype=float)
    result[:mfcc_feat.shape[0], :mfcc_feat.shape[1]] = mfcc_feat
    # print(result.shape)
    return result


def to_one_hot(x, len):
    vec = np.zeros(len)
    vec[x] = 1
    return vec


categories = [os.path.basename(c) for c in glob("../data/train/audio/*")]

train_samples = defaultdict(list)
for line in open('../data/train/testing_list.txt'):
    cat, sample = line.rstrip().split('/')
    train_samples[cat].append(sample)

test_samples = defaultdict(list)
for line in open('../data/train/validation_list.txt'):
    cat, sample = line.rstrip().split('/')
    test_samples[cat].append(sample)

print(categories, len(categories))
print(train_samples, len(train_samples))

train_samples_x = []
train_samples_y = []

test_samples_x = []
test_samples_y = []

dims = len(categories)

for i, c in enumerate(train_samples.keys()):
    print(i, c, len(train_samples[c]))
    cat_samples = np.random.choice(train_samples[c], 256)
    for s in cat_samples:
        train_samples_x.append(np.reshape(get_features('../data/train/audio/{0}/{1}'.format(c, s)), -1))
        # train_samples_y.append(to_one_hot(i, dims))
        train_samples_y.append(i)


for i, c in enumerate(train_samples.keys()):
    print(i, c, len(train_samples[c]))
    test_samples_idx = np.random.choice(test_samples[c], 64)
    for s in test_samples_idx:
        test_samples_x.append(np.reshape(get_features('../data/train/audio/{0}/{1}'.format(c, s)), -1))
        # test_samples_y.append(to_one_hot(i, dims))
        test_samples_y.append(i)

train_samples_x = np.array(train_samples_x)
train_samples_y = np.array(train_samples_y)

test_samples_x = np.array(test_samples_x)
test_samples_y = np.array(test_samples_y)

model = CatBoostClassifier(
    device_type='CPU', iterations=3000, depth=7,
    learning_rate=0.5, loss_function='MultiClass', verbose=True)

# Train the model
model.fit(train_samples_x, train_samples_y, verbose=True)

# Estimate accuracy
print(model.predict(test_samples_x))
pred = model.predict(test_samples_x)
pred = np.reshape(pred, -1)
error_rate = np.sum(pred != test_samples_y) / float(test_samples_y.shape[0])
print('Test error using softmax = {}'.format(error_rate))
model.save_model('model.cat')
