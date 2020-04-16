import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch
import numpy as np
import time
from sklearn.metrics import roc_auc_score

use_gpu = torch.cuda.is_available()

class CheXpertTrainer():

    @classmethod
    def train(cls, model, dataLoaderTrain, dataLoaderVal, nnClassCount, trMaxEpoch, launchTimestamp, checkpoint):

        # SETTINGS: OPTIMIZER & SCHEDULER
        optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)

        # SETTINGS: LOSS
        loss = torch.nn.BCEWithLogitsLoss(size_average=True)
        # LOAD CHECKPOINT
        if checkpoint is not None and use_gpu:
            modelCheckpoint = torch.load(checkpoint)
            model.load_state_dict(modelCheckpoint['state_dict'])
            optimizer.load_state_dict(modelCheckpoint['optimizer'])

        # TRAIN THE NETWORK
        lossMIN = 100000

        for epochID in range(0, trMaxEpoch):

            batchs, losst, losse = CheXpertTrainer.epochTrain(model, dataLoaderTrain, optimizer, trMaxEpoch,
                                                              nnClassCount, loss)
            lossVal = cls.epochVal(model, dataLoaderVal, optimizer, trMaxEpoch, nnClassCount, loss)

            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampEND = timestampDate + '-' + timestampTime

            if lossVal < lossMIN:
                lossMIN = lossVal
                torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'best_loss': lossMIN,
                            'optimizer': optimizer.state_dict()},
                           'm-epoch' + str(epochID) + '-' + launchTimestamp + '.pth.tar')
                print('Epoch [' + str(epochID + 1) + '] [save] [' + timestampEND + '] loss= ' + str(lossVal))
            else:
                print('Epoch [' + str(epochID + 1) + '] [----] [' + timestampEND + '] loss= ' + str(lossVal))

            return batchs, losst, losse
        # --------------------------------------------------------------------------------

    @classmethod
    def epochTrain(cls, model, dataLoader, optimizer, epochMax, classCount, loss):

        batch = []
        losstrain = []
        losseval = []

        model.train()
        len_training_data =  len(dataLoader)
        for batchID, (varInput, target) in enumerate(dataLoader):
            if use_gpu:
                varTarget = target.cuda(non_blocking=True)
            else:
                varTarget = target

            varOutput = model(varInput)
            lossvalue = loss(varOutput, varTarget)

            optimizer.zero_grad()
            lossvalue.backward()
            optimizer.step()

            l = lossvalue.item()
            losstrain.append(l)

            percentage = batchID // (len_training_data // 100)
            if batchID % (len_training_data // 100) == 0:
                print(percentage, "% batches computed")
                # Fill three arrays to see the evolution of the loss

                batch.append(batchID)

                le = cls.epochVal(model, dataLoader, optimizer, epochMax, classCount, loss).item()
                losseval.append(le)

                print("Batch id: %d" % batchID)
                print("Training loss: %.4f" % l)
                print("Evaluation loss: %.4f" % le)
                # for testing
                # if percentage == 3:
                #     break

        return batch, losstrain, losseval

    # --------------------------------------------------------------------------------
    @classmethod
    def epochVal(cls, model, dataLoader, optimizer, epochMax, classCount, loss):

        model.eval()

        lossVal = 0
        lossValNorm = 0

        with torch.no_grad():
            for i, (varInput, target) in enumerate(dataLoader):
                if use_gpu:
                    target = target.cuda(non_blocking=True)
                else:
                    target = target
                varOutput = model(varInput)

                losstensor = loss(varOutput, target)
                lossVal += losstensor
                lossValNorm += 1

        outLoss = lossVal / lossValNorm
        return outLoss

    # --------------------------------------------------------------------------------

    # ---- Computes area under ROC curve
    # ---- dataGT - ground truth data
    # ---- dataPRED - predicted data
    # ---- classCount - number of classes
    @classmethod
    def computeAUROC(cls, data_truth, data_pred, class_count):

        outAUROC = []

        datanpGT = data_truth.cpu().numpy()
        datanpPRED = data_pred.cpu().numpy()

        for i in range(class_count):
            try:
                outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
            except ValueError:
                pass
        return outAUROC

    # --------------------------------------------------------------------------------
    @classmethod
    def test(cls, model, dataLoaderTest, nnClassCount, checkpoint, class_names):

        cudnn.benchmark = True

        if checkpoint is not None:
            modelCheckpoint = torch.load(checkpoint)
            model.load_state_dict(modelCheckpoint['state_dict'])

        if use_gpu:
            outGT = torch.FloatTensor().cuda()
            outPRED = torch.FloatTensor().cuda()
        else:
            outGT = torch.FloatTensor()
            outPRED = torch.FloatTensor()

        model.eval()

        with torch.no_grad():
            for i, (input, target) in enumerate(dataLoaderTest):
                if use_gpu:
                    target = target.cuda(non_blocking=True)
                outGT = torch.cat((outGT, target), 0)

                bs, c, h, w = input.size()
                varInput = input.view(-1, c, h, w)

                out = model(varInput)
                outPRED = torch.cat((outPRED, out), 0).cuda()
        aurocIndividual = CheXpertTrainer.computeAUROC(outGT, outPRED, nnClassCount)
        aurocMean = np.array(aurocIndividual).mean()

        print('AUROC mean ', aurocMean)

        for i in range(0, len(aurocIndividual)):
            print(class_names[i], ' ', aurocIndividual[i])

        return outGT, outPRED