import torch
from model import DenseNet121
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from train import OUTPUT_CLASS_COUNT
from train import class_names
from train import NORMALIZE_FACTOR


class HeatmapGenerator:

    def __init__(self, path_to_model, output_class_count, resize_dim):
        self.use_gpu = torch.cuda.is_available()
        model = DenseNet121(output_class_count).cuda()

        if self.use_gpu:
            model = torch.nn.DataParallel(model).cuda()
        else:
            model = torch.nn.DataParallel(model)
        # load the saved model state_dict
        model.load_state_dict(torch.load(path_to_model)['state_dict'])

        self.model = model
        self.resize_dim = resize_dim
        self.model.eval()
        self.weights = list(self.model.module.densenet121.features.parameters())[-2]

        normalize = transforms.Normalize(NORMALIZE_FACTOR[0], NORMALIZE_FACTOR[1])
        transform_list = [transforms.Resize(self.resize_dim), transforms.ToTensor(), normalize]
        self.transformSequence = transforms.Compose(transform_list)

    def generate(self, input_file, output_file):

        # load the image and pass to the model
        with torch.no_grad():
            image = Image.open(input_file).convert('RGB')
            image = self.transformSequence(image)
            image = image.unsqueeze_(0)
            if self.use_gpu:
                image = image.cuda()
            label = self.model(image)
            output = self.model.module.densenet121.features(image)
            label = class_names[torch.max(label, 1)[1]]

            # generate heatmap
            for i in range(0, len(self.weights)):
                cur_output = output[0, i, :, :]
                if i == 0:
                    ith_heatmap = self.weights[i] * cur_output
                else:
                    ith_heatmap += self.weights[i] * cur_output
                np_heatmap = ith_heatmap.cpu().data.numpy()

        # blend image and heatmap
        orig_image = cv2.resize(cv2.imread(input_file, 1), self.resize_dim)
        cam = np_heatmap / np.max(np_heatmap)
        cam = cv2.resize(cam, self.resize_dim)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

        img = cv2.addWeighted(orig_image, 1, heatmap, 0.35, 0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.title(label)
        plt.imshow(img)
        plt.plot()
        plt.axis('off')
        plt.savefig(output_file)
        plt.show()


def main():
    input_image = './CheXpert-v1.0-small/valid/patient64543/study1/view1_frontal.jpg'
    output_image = 'heatmap_view1_frontal.png'
    path_to_model = "model_ones_densenet_preprocessed.pth.tar"

    HeatmapGenerator(path_to_model, OUTPUT_CLASS_COUNT, (320, 320)).generate(input_image, output_image)
