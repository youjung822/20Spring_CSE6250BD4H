import torch.nn as nn
import torchvision

class DenseNet121(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x


class Resnet101(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard Resnet101
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, out_size):
        super(Resnet101, self).__init__()
        self.resnet101 = torchvision.models.resnet101(pretrained=False, num_classes=out_size)

    def forward(self, x):
        x = self.resnet101(x)
        return x
