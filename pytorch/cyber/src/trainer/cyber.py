import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
import src.model.model as M
import src.datasets.cyber as CD
import src.utils.dirs as dirs
from src.base.trainer import BaseTrainer
import src.utils.metrics as mtx
import src.utils.eval_metrics as metr
import src.trainer.net_metrics as nmetr


class ModelTrainer(BaseTrainer):
    """
    # Arguments
        model: Keras model, model for training
        dataproc: BaseDataProcessor, generate datasets (for train/test/validate)
        config: DotMap, settings for training 'model',
                according to 'training_mode' (i.e. pre-training/finetune)
        train_mode: String. 'pretrain_set' or 'finetune_set'.
        verbose: Integer. 0, 1, or 2. Verbosity mode.
                    0 = silent, 1 = progress bar, 2 = one line per epoch.
        model: not trained or initially pre-trained model
    """

    def __init__(self, model, dataproc, config, optimizer, **kwargs):
        super(ModelTrainer, self).__init__(model, dataproc, config)

        self.optimizer = optimizer
        self.train_loader = self.dataproc.train_dataloader()
        self.val_loader = self.dataproc.val_dataloader()

        self.epochs = self.config['epochs']
        self.log_inter = self.config['log_inter']

        self.train_mode = kwargs.get('train_mode', 'pretrain_set')
        self.start_epoch = kwargs.get('start_epoch', 1)
        self.device = kwargs.get('device', 'gpu')
        self.exp_path = kwargs.get('exp_path', './')
        self.score_path = kwargs.get('score_path', './')

        self.verbose = kwargs.get('verbose', 1)
        self.overwrite = kwargs.get('overwrite', False)

        self.callbacks = []
        self.loss = []
        self.val_loss = []

    def train(self):
        # model training
        best_epoch = 0
        best_metric = np.inf
        early_stopping, max_patience = 0, 5  # for early stopping
        for epoch in range(self.start_epoch, self.start_epoch + self.epochs):
            # train one epoch
            self._run_epoch(self.train_loader, self.model, self.optimizer, epoch, self.device, self.log_inter)
            # evaluate on validation set
            metrx_val = self.validate()

            # remember best metric and save checkpoint
            is_best = metrx_val < best_metric
            best_metric = min(metrx_val, best_metric)

            # adjust learning rate + early stopping
            if is_best:
                early_stopping = 0
                best_epoch = epoch + 1
            else:
                early_stopping += 1
                if epoch - best_epoch > 2:
                    self.optimizer.increase_delta()
                    best_epoch = epoch + 1
            if early_stopping == max_patience:
                break

            if is_best:
                M.save_checkpoint({
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'metrics': {'eer': best_metric},
                    'optimizer': self.optimizer.state_dict(),
                }, is_best, self.exp_path, str(epoch) + '.pth.tar')

        # load best model
        best_model_pth = os.path.join(self.exp_path, 'model_best.pth.tar')
        print("===> loading best model for scoring: '{}'".format(best_model_pth))
        checkpoint = torch.load(best_model_pth)
        self.model.load_state_dict(checkpoint['state_dict'])

        # compute metrics
        score_file_pth = self.score_path + '-scores.txt'
        print("===> scoring file saved at: '{}'".format(score_file_pth))
        ModelTrainer.prediction(self.val_loader, self.dataproc.meta_data, self.model, self.device, score_file_pth)

    def _run_epoch(self, train_loader, model, optimizer, epoch, device, log_interval):
        batch_time = nmetr.AverageMeter()
        data_time = nmetr.AverageMeter()
        losses = nmetr.AverageMeter()

        # switch to train mode
        model.train()

        stime = time.time()
        for i, (_, input, target) in enumerate(train_loader):  # TODO training loop
            # measure data loading time
            data_time.update(time.time() - stime)

            # Create vaiables
            # TODO fix sampling accross utterances!!!
            # input = input[0].to(device)
            # target = torch.tensor(target).to(device).view((-1,))
            # TODO note, features comes transposed. Check Kaggle forward pass dfdc
            input = torch.reshape(input, (input.size(0) * input.size(1),) + input.shape[2:])
            input = input.to(device)
            target = torch.stack(target)
            target = target.reshape(-1)
            target = target.to(device).view((-1,))

            # compute output scores
            output = model(input)
            # loss
            loss = F.nll_loss(output, target)
            # TODO add metrics per batch
            losses.update(loss.item(), input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr = optimizer.update_learning_rate()

            # measure elapsed time
            batch_time.update(time.time() - stime)
            stime = time.time()

            if i % log_interval == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'LR {lr:.6f}\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, lr=lr, loss=losses))

    def validate(self):
        batch_time = nmetr.AverageMeter()
        losses = nmetr.AverageMeter()

        # switch to evaluate mode
        self.model.eval()
        utt2scores = defaultdict(list)
        with torch.no_grad():
            stime = time.time()
            for i, (fids, x, y) in enumerate(self.val_loader):
                # TODO fix later the collacation function
                x = x[0].to(self.device)
                p = self.model(x)
                y = torch.tensor(y).to(self.device).view((-1,))
                loss = F.nll_loss(p, y)
                # measure accuracy and record loss
                losses.update(loss.item(), x.size(0))

                # calculate val metrics
                p = p.cpu().numpy()
                score = p[:, 0]  # 1. - 1. / (1. + np.exp(-output[:, 0])) #output[:, 0]
                for j, fid in enumerate(fids):
                    utt2scores[fid[0]].append(score[j].item()) # TODO optimize

                # measure elapsed time
                batch_time.update(time.time() - stime)
                stime = time.time()

                if i % self.log_inter == 0:
                    print('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                        i, len(self.val_loader), batch_time=batch_time, loss=losses))

        file_ids, labels = self.dataproc.meta_data.fold_list(fold=1,
                                                             data_split=CD.ASVSpoof19Meta.DataSplit.validation)
        # predictions averaging
        spoof, bona = [], []
        for i, (fid, dig_lab) in enumerate(zip(file_ids, labels)):
            if fid not in utt2scores:  # condition for leave on out
                print('[INFO] file is missing from val set: %s' % fid)
                continue
            sc_list = utt2scores[fid]
            avg_sc = np.mean(sc_list)
            # spoof_id = id_list[index]
            if dig_lab == 0:  # if bonafide
                bona.append(avg_sc)
            else:
                spoof.append(avg_sc)

        spoof, bona = np.array(spoof), np.array(bona)
        eer_cm = 100. * metr.compute_eer(bona, spoof)[0]
        pred = np.append(bona, spoof)
        y = np.append([0] * len(bona), [1] * len(spoof))
        eer_pool = mtx.pooled_eer(y_true=y, y_pred=pred)

        print('===> EER_CM: {} vs PooledEER: {}\n'.format(eer_cm, eer_pool))
        return eer_cm

    @staticmethod
    def prediction(val_loader, val_meta, model, device, output_file):
        utt2scores = defaultdict(list)
        model.eval()
        with torch.no_grad():
            for i, (utt_batch, x, y) in enumerate(val_loader):
                x = x[0].to(device)
                p = model(x)
                # get score
                p = p.cpu().numpy()
                score = p[:, 0]

                for index, fid in enumerate(utt_batch):
                    utt2scores[fid[0]].append(score[index].item())
            utt_list, lab_ids = val_meta.fold_list(fold=1, data_split=CD.ASVSpoof19Meta.DataSplit.validation)

            with open(output_file, 'w') as f:
                for fid, lid in zip(utt_list, lab_ids):
                    if fid not in utt2scores:
                        print('[WARN] file has no scoring: %s' % fid)
                        continue
                    score_list = utt2scores[fid]
                    avg_score = np.mean(score_list)
                    avg_score = 1. / (1. + np.exp(-avg_score))
                    if lid == 0:
                        f.write('%s %s %s %f\n' % (fid, '-', 'bonafide', avg_score))
                    else:
                        f.write('%s %s %s %f\n' % (fid, lid, 'spoof', avg_score))


# class ModelValidator(Callback):
class ModelValidator(object):
    def __init__(self, batch_gen, metrics, monitor, mode):
        super(ModelValidator, self).__init__()
        self.batch_gen = batch_gen
        self.metrics = metrics
        self.monitor = monitor
        self.best_epoch = 0

        if mode == 'min':
            self.monitor_op = np.less
            self.best_acc = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best_acc = 0.
        else:
            raise AttributeError('[ERROR] ModelValidator mode %s is unknown')

    def on_train_begin(self, logs=None):
        super(ModelValidator, self).on_train_begin(logs)
        vs = ModelValidator.validate_model(self.model, self.batch_gen, self.metrics)
        for k, v in vs.items():
            logs[k] = np.float64(v)

        print(' BEFORE TRAINING: Validation loss: %.4f, Validation %s: %.4f / best %.4f' % (
            vs['val_loss'], self.monitor.upper(), vs[self.monitor], self.best_acc))
        print(logs)

        if self.monitor_op(logs[self.monitor], self.best_acc):
            self.best_acc = logs[self.monitor]
            self.best_epoch = -1

    def on_epoch_end(self, epoch, logs={}):
        super(ModelValidator, self).on_epoch_end(epoch, logs)
        vs = ModelValidator.validate_model(self.model, self.batch_gen, self.metrics)
        for k, v in vs.items():
            logs[k] = np.float64(v)

        print(' EPOCH %d. Validation loss: %.4f, Validation %s: %.4f / best %.4f' % (
            epoch, vs['val_loss'], self.monitor.upper(), vs[self.monitor], self.best_acc))

        if self.monitor_op(logs[self.monitor], self.best_acc):
            self.best_acc = logs[self.monitor]
            self.best_epoch = epoch

    def on_train_end(self, logs=None):
        super(ModelValidator, self).on_train_end(logs)
        print('=' * 20 + ' Training report ' + '=' * 20)
        print('Best validation %s: epoch %s / %.4f\n' % (self.monitor.upper(), self.best_epoch, self.best_acc))

    @staticmethod
    def validate_model(model, batch_gen, metrics):
        """
        # Arguments
            model: Keras model
            data: BaseDataLoader
            metrics: list of metrics
        # Output
            dictionary with values of metrics and loss
        """
        cut_model = model
        if DCASEModelTrainer.is_mfom_objective(model):
            input = model.get_layer(name='input').output
            preact = model.get_layer(name='output').output
            cut_model = Model(input=input, output=preact)

        n_class = cut_model.output_shape[1]
        y_true, y_pred = np.empty((0, n_class)), np.empty((0, n_class))
        loss, cnt = 0, 0

        for X_b, Y_b in batch_gen.batch():
            ps = cut_model.predict_on_batch(X_b)
            y_pred = np.vstack([y_pred, ps])
            y_true = np.vstack([y_true, Y_b])
            # NOTE: it is fake loss, caz Y is fed
            if DCASEModelTrainer.is_mfom_objective(model):
                X_b = [Y_b, X_b]
            l = model.test_on_batch(X_b, Y_b)
            loss += l
            cnt += 1

        vals = {'val_loss': loss / cnt}

        for m in metrics:
            if m == 'micro_f1':
                p = mtx.step(y_pred, threshold=0.5)
                vals[m] = mtx.micro_f1(y_true, p)
            elif m == 'pooled_eer':
                p = y_pred.flatten()
                y = y_true.flatten()
                vals[m] = mtx.eer(y, p)
            elif m == 'class_wise_eer':
                vals[m] = np.mean(mtx.class_wise_eer(y_true, y_pred))
            elif m == 'accuracy':
                p = np.argmax(y_pred, axis=-1)
                y = np.argmax(y_true, axis=-1)
                vals[m] = mtx.pooled_accuracy(y, p)
            else:
                raise KeyError('[ERROR] Such a metric is not implemented: %s...' % m)
        return vals
