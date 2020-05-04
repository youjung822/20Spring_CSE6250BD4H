from torch.utils.data import Dataset
from PIL import Image
import torch
import csv
from os import path


class CheXpertDataSet(Dataset):
    # policy is how we treat unknown labels
    # "ones" means we treat them as positive, "zeros" means we treat them as negative
    def __init__(self, image_file_list, transform=None, policy="ones"):
        image_names = []
        labels = []
        from train import OUTPUT_CLASS_COUNT
        with open(image_file_list, "r") as f:
            csv_reader = csv.reader(f)
            next(csv_reader, None)
            k = 0
            for line in csv_reader:
                k += 1
                image_name = line[0]
                label = line[5:]

                for i in range(OUTPUT_CLASS_COUNT):
                    if label[i]:
                        a = float(label[i])
                        if a == 1:
                            label[i] = 1
                        elif a == -1:
                            if policy == "ones":
                                label[i] = 1
                            elif policy == "zeroes":
                                label[i] = 0
                            else:
                                label[i] = 0
                        else:
                            label[i] = 0
                    else:
                        label[i] = 0

                image_name = './' + image_name
                dir_list = image_name.split('/')
                dir_list[-1] = 'temp' + dir_list[-1]
                transformed_image_name = '/'.join(dir_list)
                if path.exists(transformed_image_name):
                    image_names.append(transformed_image_name)
                    labels.append(label)

        self.image_names = image_names
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        """Take the index of item and returns the image and its labels"""

        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.image_names)
