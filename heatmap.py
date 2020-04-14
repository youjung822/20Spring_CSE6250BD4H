import torch
from model import DenseNet121
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from train import nnClassCount
from train import class_names

use_gpu = torch.cuda.is_available()


class HeatmapGenerator:

    # ---- Initialize heatmap generator
    # ---- pathModel - path to the trained densenet model
    # ---- nnArchitecture - architecture name DENSE-NET121, DENSE-NET169, DENSE-NET201
    # ---- nnClassCount - class count, 14 for chxray-14

    def __init__(self, pathModel, nnClassCount, resize_dim):

        # ---- Initialize the network
        model = DenseNet121(nnClassCount).cuda()

        if use_gpu:
            model = torch.nn.DataParallel(model).cuda()
        else:
            model = torch.nn.DataParallel(model)

        modelCheckpoint = torch.load(pathModel)
        model.load_state_dict(modelCheckpoint['state_dict'])

        self.model = model
        self.resize_dim = resize_dim
        self.model.eval()

        # ---- Initialize the weights
        self.weights = list(self.model.module.densenet121.features.parameters())[-2]

        # ---- Initialize the image transform
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transformList = []
        transformList.append(transforms.Resize(self.resize_dim))
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)
        self.transformSequence = transforms.Compose(transformList)

    # --------------------------------------------------------------------------------

    def generate(self, pathImageFile, pathOutputFile):

        # ---- Load image, transform, convert
        with torch.no_grad():

            imageData = Image.open(pathImageFile).convert('RGB')
            imageData = self.transformSequence(imageData)
            imageData = imageData.unsqueeze_(0)
            if use_gpu:
                imageData = imageData.cuda()
            l = self.model(imageData)
            output = self.model.module.densenet121.features(imageData)
            label = class_names[torch.max(l, 1)[1]]
            # ---- Generate heatmap
            heatmap = None
            for i in range(0, len(self.weights)):
                map = output[0, i, :, :]
                if i == 0:
                    heatmap = self.weights[i] * map
                else:
                    heatmap += self.weights[i] * map
                npHeatmap = heatmap.cpu().data.numpy()

        # ---- Blend original and heatmap

        imgOriginal = cv2.imread(pathImageFile, 1)
        imgOriginal = cv2.resize(imgOriginal, self.resize_dim)

        cam = npHeatmap / np.max(npHeatmap)
        cam = cv2.resize(cam, self.resize_dim)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

        img = cv2.addWeighted(imgOriginal, 1, heatmap, 0.35, 0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.title(label)
        plt.imshow(img)
        plt.plot()
        plt.axis('off')
        plt.savefig(pathOutputFile)
        plt.show()


pathInputImage = './CheXpert-v1.0-small/valid/patient64543/study1/view1_frontal.jpg'
pathOutputImage = 'heatmap_view1_frontal.png'
pathModel = "model_ones_densenet_preprocessed.pth.tar"


h = HeatmapGenerator(pathModel, nnClassCount, (320, 320))

h.generate(pathInputImage, pathOutputImage)
