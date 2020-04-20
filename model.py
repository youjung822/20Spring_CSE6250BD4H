import torch.nn as nn
import torchvision


# Model definition of the densenet121 architecture
# We use the pre-trained densenet121 network to extract features and replace the classifier in the original model with a
# linear layer and a sigmoid activation function
class DenseNet121(nn.Module):
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(self.densenet121.classifier.in_features, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x


# Model definition of the resent101 architecture
# We use the non pre-trained resnet101 architecture and specified the output size of the number of classes we have
class Resnet101(nn.Module):
    def __init__(self, out_size):
        super(Resnet101, self).__init__()
        self.resnet101 = torchvision.models.resnet101(pretrained=False, num_classes=out_size)

    def forward(self, x):
        x = self.resnet101(x)
        return x
