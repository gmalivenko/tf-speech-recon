import argparse
import numpy as np
from attrdict import AttrDict
from sklearn.metrics import accuracy_score, average_precision_score

import torch
import torch.utils.data as data_utils
from torch.autograd import Variable

from model import resnet18
from dataset import WavDataset

use_cuda = False


def train(args):
    ds = WavDataset(lst=args.dataset_root + 'train/validation_list.txt', augment=True)
    test_ds = WavDataset(lst=args.dataset_root + 'train/testing_list.txt')

    # model = RNNModel(num_classes=len(ds.labels))
    model = resnet18(pretrained=False, num_classes=12)

    if use_cuda:
        model = model.cuda()

    criterion = torch.nn.BCELoss()
    model_optimizer = torch.optim.Adadelta(model.parameters(), lr=1e-2, eps=1e-6)

    train_loader = data_utils.DataLoader(
        ds, batch_size=32, shuffle=True,
        num_workers=0)

    test_loader = data_utils.DataLoader(
        test_ds, batch_size=128, shuffle=False,
        num_workers=0)

    best_AP = 0

    for epoch in range(1000):
        losses = []
        for (spectrogram, mfcc, classes) in train_loader:
            spectrogram = Variable(spectrogram)
            mfcc = Variable(mfcc)
            classes = Variable(classes)

            if use_cuda:
                spectrogram = spectrogram.cuda()
                mfcc = mfcc.cuda()
                classes = classes.cuda()
            output = model(spectrogram, mfcc)
            loss = criterion(output, classes)  # ones = true

            model.zero_grad()
            loss.backward()
            model_optimizer.step()
            losses.append(loss.data[0])

        print('Epoch: {0}, L: {1}'.format(
            epoch, np.average(np.array(losses))))
        AP = []
        ACC = []
        for (images, classes) in test_loader:
            images = Variable(images)
            if use_cuda:
                images = images.cuda()
            output = model(images)

            y_true = classes.numpy()
            y_pred = output.cpu().data.numpy()

            current = average_precision_score(np.reshape(y_true, -1), np.reshape(y_pred, -1))

            # print(current)
            AP.append(current)
            ACC.append(
                np.sum(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1)) / float(y_true.shape[0])
            )

        print('AP: {0}'.format(np.average(np.array(AP))))
        print('ACC: {0}'.format(np.average(np.array(ACC))))
        if np.average(np.array(AP)) > best_AP:
            best_AP = np.average(np.array(AP))
            torch.save(model.state_dict(), args.checkpoint_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Training script.'
    )
    parser.add_argument('--dataset-root', required=True,
                        help='a path to the dataset')
    parser.add_argument('--checkpoint-path', required=True,
                        help='a path to the model')
    args = parser.parse_args()
    train(args)
