import numpy as np
import time
import matplotlib.pyplot as plt

import torch

import torchvision.transforms as transforms

from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader

from trainer import CheXpertTrainer
from model import DenseNet121
# from model import Resnet101
from dataset import CheXpertDataSet

use_gpu = torch.cuda.is_available()

# Paths to the files with training, and validation sets.
# Each file contains pairs (path to image, output vector)
train_data_csv = './CheXpert-v1.0-small/train.csv'
test_data_csv = './CheXpert-v1.0-small/valid.csv'

# Training settings: batch size, maximum number of epochs
batch_size = 16
max_epoch = 1

# Class names
class_names = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
               'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
               'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
OUTPUT_CLASS_COUNT = len(class_names)
NORMALIZE_FACTOR = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

# normalization
normalize = transforms.Normalize(NORMALIZE_FACTOR[0], NORMALIZE_FACTOR[1])
transformList = [transforms.ToTensor(), normalize]
transformSequence = transforms.Compose(transformList)

# load data files
dataset = CheXpertDataSet(train_data_csv, transformSequence, policy="ones")
# split the train and validation data
dataset_valid, dataset_train = random_split(dataset, [500, len(dataset) - 500])
datasetTest = CheXpertDataSet(test_data_csv, transformSequence)

train_data = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
validation_data = \
    DataLoader(dataset=dataset_valid, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
test_data = DataLoader(dataset=datasetTest, num_workers=8, pin_memory=True)

# initialize and load the model
# Replace with Resetnet to switch model to train
if use_gpu:
    model = DenseNet121(OUTPUT_CLASS_COUNT).cuda()
    model = torch.nn.DataParallel(model).cuda()
else:
    model = DenseNet121(OUTPUT_CLASS_COUNT)
    model = torch.nn.DataParallel(model)


def main():
    timestamp_run = time.strftime("%d%m%Y") + '-' + time.strftime("%H%M%S")

    batch, losst, losse = CheXpertTrainer.train(model, train_data, validation_data, OUTPUT_CLASS_COUNT, max_epoch,
                                                timestamp_run, checkpoint=None)

    print("Length of training loss all: %d" % len(losst))
    print("Training Loss (All):\n", losst)

    # take the mean so that we have same length of data for train loss and evaluation loss
    losstn = []
    j = 0
    for i in range(0, len(losse)):
        losstn.append(np.mean(losst[j:j+len(losst) // len(losse)]))
        j += len(losst) // len(losse)

    print("Training Loss:\n", losstn)
    print("Evaluation Loss:\n", losse)

    batch = [i*(len(losst) // len(losse)) for i in range(len(losse))]

    plt.plot(batch, losstn, label="train")
    plt.plot(batch, losse, label="eval")
    plt.xlabel("# of batches (batch_size = %d)" % batch_size)
    plt.ylabel("BCE Loss")
    plt.title("BCE Loss Graph")
    plt.legend()

    plt.savefig("bce_loss.png", dpi=1000)
    plt.show()


if __name__ == "__main__":
    main()
