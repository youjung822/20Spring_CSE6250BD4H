import numpy as np
import time
import matplotlib.pyplot as plt

import torch

import torchvision.transforms as transforms

from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader

from trainer import CheXpertTrainer
from model import DenseNet121
from model import Resnet101
from dataset import CheXpertDataSet

use_gpu = torch.cuda.is_available()

# Paths to the files with training, and validation sets.
# Each file contains pairs (path to image, output vector)
pathFileTrain = './CheXpert-v1.0-small/train.csv'
pathFileValid = './CheXpert-v1.0-small/valid.csv'

# Training settings: batch size, maximum number of epochs
batch_size = 16
max_epoch = 1

# Class names
class_names = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
               'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
               'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
nnClassCount = len(class_names)                   # dimension of the output

# TRANSFORM DATA

normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
transformList = [transforms.ToTensor(), normalize]
transformSequence = transforms.Compose(transformList)

# LOAD DATASET

dataset = CheXpertDataSet(pathFileTrain, transformSequence, policy="ones")
datasetValid, datasetTrain = random_split(dataset, [500, len(dataset) - 500])
datasetTest = CheXpertDataSet(pathFileValid, transformSequence)

dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
dataLoaderVal = DataLoader(dataset=datasetValid, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
dataLoaderTest = DataLoader(dataset=datasetTest, num_workers=8, pin_memory=True)

# initialize and load the model
if use_gpu:
    model = Resnet101(nnClassCount).cuda()
    model = torch.nn.DataParallel(model).cuda()
else:
    model = Resnet101(nnClassCount)
    model = torch.nn.DataParallel(model)

def main():

    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%d%m%Y")
    timestampLaunch = timestampDate + '-' + timestampTime

    batch, losst, losse = CheXpertTrainer.train(model, dataLoaderTrain, dataLoaderVal, nnClassCount, max_epoch,
                                                timestampLaunch, checkpoint=None)
    print("Model trained")

    print("Length of training loss all: %d" % len(losst))
    print("Training Loss (All):\n", losst)

    losstn = []
    j = 0
    for i in range(0, len(losse)):
        losstn.append(np.mean(losst[j:j+len(losst) // len(losse)]))
        j += len(losst) // len(losse)

    print("Training Loss:\n", losstn)
    print("Evaluation Loss:\n", losse)


    lt = losstn
    le = losse
    batch = [i*(len(losst) // len(losse)) for i in range(len(le))]

    plt.plot(batch, lt, label="train")
    plt.plot(batch, le, label="eval")
    plt.xlabel("# of batches (batch_size = 32)")
    plt.ylabel("BCE Loss")
    plt.title("BCE Loss Graph")
    plt.legend()

    plt.savefig("bce_loss.png", dpi=1000)
    plt.show()


if __name__ == "__main__":
    main()
