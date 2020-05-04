import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch
import numpy as np
import time
from sklearn.metrics import roc_auc_score

use_gpu = torch.cuda.is_available()


class CheXpertTrainer:
    @classmethod
    def train(cls, model, train_data, validation_data, output_class_count, num_epoch, run_time, checkpoint):

        # optimizer
        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
        # loss
        loss = torch.nn.BCEWithLogitsLoss(size_average=True)

        # we can resume training from an earlier saved checkpoint
        if checkpoint is not None and use_gpu:
            model_checkpoint = torch.load(checkpoint)
            model.load_state_dict(model_checkpoint['state_dict'])
            optimizer.load_state_dict(model_checkpoint['optimizer'])

        loss_upper_bound = 100000

        for epoch in range(0, num_epoch):
            batches, losst, losse = CheXpertTrainer.train_epoch(model, train_data, optimizer, num_epoch,
                                                                output_class_count, loss)
            lossVal = cls.eval_epoch(model, validation_data, optimizer, num_epoch, output_class_count, loss)

            timestamp_end = time.strftime("%d%m%Y") + '-' + time.strftime("%H%M%S")

            if lossVal < loss_upper_bound:
                loss_upper_bound = lossVal
                torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'best_loss': loss_upper_bound,
                            'optimizer': optimizer.state_dict()},
                           'm-epoch' + str(epoch) + '-' + run_time + '.pth.tar')
                print('Epoch [' + str(epoch + 1) + '] [save] [' + timestamp_end + '] loss= ' + str(lossVal))
            else:
                print('Epoch [' + str(epoch + 1) + '] [----] [' + timestamp_end + '] loss= ' + str(lossVal))

            return batches, losst, losse

    @classmethod
    def train_epoch(cls, model, train_data, optimizer, num_epoch, output_class_count, loss):

        batch = []
        losst = []
        lossv = []

        model.train()
        len_training_data = len(train_data)
        for cur_batch_id, (input_data, target) in enumerate(train_data):
            if use_gpu:
                target = target.cuda(non_blocking=True)
            else:
                target = target

            output = model(input_data)
            loss_val = loss(output, target)

            # back propagation
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            loss_item = loss_val.item()
            losst.append(loss_item)

            percentage = cur_batch_id // (len_training_data // 100)
            if cur_batch_id % (len_training_data // 100) == 0:
                print(percentage, "% batches computed")

                batch.append(cur_batch_id)
                le = cls.eval_epoch(model, train_data, optimizer, num_epoch, output_class_count, loss)
                lossv.append(le)

                print("Batch id: %d" % cur_batch_id)
                print("Training loss: %.4f" % loss_item)
                print("Evaluation loss: %.4f" % le)

        return batch, losst, lossv

    @classmethod
    def eval_epoch(cls, model, input_data, optimizer, num_epoch, output_class_count, loss):
        model.eval()

        loss_val = 0
        loss_count = 0

        with torch.no_grad():
            for i, (varInput, target) in enumerate(input_data):
                if use_gpu:
                    target = target.cuda(non_blocking=True)
                else:
                    target = target
                output = model(varInput)

                loss_val += loss(output, target)
                loss_count += 1

        return loss / loss_count

    @classmethod
    def compute_area_under_roc(cls, data_truth, data_pred, class_count):
        result = []
        data_gt = data_truth.cpu().numpy()
        data_pred = data_pred.cpu().numpy()

        for i in range(class_count):
            try:
                result.append(roc_auc_score(data_gt[:, i], data_pred[:, i]))
            except ValueError:
                pass
        return result

    @classmethod
    def test(cls, model, test_data, output_class_count, checkpoint, class_names):
        cudnn.benchmark = True

        if checkpoint is not None:
            model_checkpoint = torch.load(checkpoint)
            model.load_state_dict(model_checkpoint['state_dict'])

        if use_gpu:
            out_gt = torch.FloatTensor().cuda()
            out_pred = torch.FloatTensor().cuda()
        else:
            out_gt = torch.FloatTensor()
            out_pred = torch.FloatTensor()

        model.eval()

        with torch.no_grad():
            for i, (input_data, target) in enumerate(test_data):
                if use_gpu:
                    target = target.cuda(non_blocking=True)
                out_gt = torch.cat((out_gt, target), 0)

                bs, c, h, w = input_data.size()

                out = model(input_data.view(-1, c, h, w))
                out_pred = torch.cat((out_pred, out), 0).cuda()
        area_single = CheXpertTrainer.compute_area_under_roc(out_gt, out_pred, output_class_count)
        area_mean = np.array(area_single).mean()

        print('AUROC mean ', area_mean)

        for i in range(0, len(area_single)):
            print(class_names[i], ' ', area_single[i])

        return out_gt, out_pred
