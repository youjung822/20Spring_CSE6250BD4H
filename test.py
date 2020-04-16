from trainer import CheXpertTrainer
from train import nnClassCount
from train import dataLoaderTest
from model import DenseNet121
from model import Resnet101
import torch
from train import class_names
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

model = Resnet101(nnClassCount).cuda()
model = torch.nn.DataParallel(model).cuda()
out_gt_resnet, out_pred_resnet = \
    CheXpertTrainer.test(model, dataLoaderTest, nnClassCount, "model_ones_resnet.tar", class_names)
model = DenseNet121(nnClassCount).cuda()
model = torch.nn.DataParallel(model).cuda()
out_gt_densenet, out_pred_densenet = \
    CheXpertTrainer.test(model, dataLoaderTest, nnClassCount, "model_ones_densenet_preprocessed.pth.tar", class_names)

for i in range(nnClassCount):
    fpr, tpr, threshold = metrics.roc_curve(out_gt_resnet.cpu()[:, i], out_pred_resnet.cpu()[:, i])
    roc_auc = metrics.auc(fpr, tpr)
    f = plt.subplot(2, 7, i+1)
    fpr2, tpr2, threshold2 = metrics.roc_curve(out_gt_densenet.cpu()[:, i], out_pred_densenet.cpu()[:, i])
    roc_auc2 = metrics.auc(fpr2, tpr2)

    plt.title('ROC for: ' + class_names[i])
    plt.plot(fpr, tpr, label='Resnet: AUC = %0.2f' % roc_auc)
    plt.plot(fpr2, tpr2, label='Densenet: AUC = %0.2f' % roc_auc2)

    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 30
fig_size[1] = 10
plt.rcParams["figure.figsize"] = fig_size

plt.savefig("test_result_auc.png", dpi=1000)
plt.show()
