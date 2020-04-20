from trainer import CheXpertTrainer
from train import OUTPUT_CLASS_COUNT
from train import dataLoaderTest
from model import DenseNet121
from model import Resnet101
import torch
from train import class_names
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

model1 = Resnet101(OUTPUT_CLASS_COUNT).cuda()
model1 = torch.nn.DataParallel(model1).cuda()
out_gt_resnet, out_pred_resnet = \
    CheXpertTrainer.test(model1, dataLoaderTest, OUTPUT_CLASS_COUNT, "model_ones_resnet.tar", class_names)
model2 = DenseNet121(OUTPUT_CLASS_COUNT).cuda()
model2 = torch.nn.DataParallel(model2).cuda()
out_gt_densenet, out_pred_densenet = CheXpertTrainer.test(model2, dataLoaderTest, OUTPUT_CLASS_COUNT,
                                                          "model_ones_densenet_preprocessed.pth.tar", class_names)

for i in range(OUTPUT_CLASS_COUNT):
    fp_rate, tp_rate, _ = metrics.roc_curve(out_gt_resnet.cpu()[:, i], out_pred_resnet.cpu()[:, i])
    roc_auc = metrics.auc(fp_rate, tp_rate)
    f = plt.subplot(2, 7, i+1)
    fp_rate2, tp_rate2, _ = metrics.roc_curve(out_gt_densenet.cpu()[:, i], out_pred_densenet.cpu()[:, i])
    roc_auc2 = metrics.auc(fp_rate2, tp_rate2)

    plt.title('ROC for: ' + class_names[i])
    plt.plot(fp_rate, tp_rate, label='Resnet: AUC = %0.2f' % roc_auc)
    plt.plot(fp_rate2, tp_rate2, label='Densenet: AUC = %0.2f' % roc_auc2)

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
