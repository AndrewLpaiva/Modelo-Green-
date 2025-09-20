import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    elif args.lradj == "cosine_iter":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_steps * math.pi))}    
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # print('Updating learning rate to {}'.format(lr))
    return lr

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path, optimizer=None, epoch=None, extra_dir=None):
        """
        val_loss: current validation loss
        model: torch nn.Module
        path: legacy path where the original state_dict is saved (./checkpoints/<setting>)
        optimizer: optional optimizer to save state
        epoch: optional current epoch index
        extra_dir: optional extra directory to save a full checkpoint (model+optimizer+epoch)
        """
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, optimizer=optimizer, epoch=epoch, extra_dir=extra_dir)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, optimizer=optimizer, epoch=epoch, extra_dir=extra_dir)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path, optimizer=None, epoch=None, extra_dir=None):
        """
        Legacy behavior: save model.state_dict() at `path/checkpoint.pth` so existing code that
        expects that file keeps working.

        New behavior: also save a full checkpoint dict containing optimizer state and epoch to
        `extra_dir/checkpoint_full.pth` (or to `path/checkpoint_full.pth` if extra_dir is None).
        """
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        # Ensure path exists
        try:
            os.makedirs(path, exist_ok=True)
        except Exception:
            pass

        # Legacy: save only model.state_dict() for compatibility
        torch.save(model.state_dict(), os.path.join(path, 'checkpoint.pth'))

        # Save a full checkpoint (model + optimizer + epoch + val_loss)
        full_ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss,
        }
        if optimizer is not None:
            try:
                full_ckpt['optimizer_state_dict'] = optimizer.state_dict()
            except Exception:
                # optimizer may not be serializable in some edge cases; ignore if so
                pass

        save_dir = extra_dir if extra_dir is not None else path
        try:
            os.makedirs(save_dir, exist_ok=True)
        except Exception:
            pass

        torch.save(full_ckpt, os.path.join(save_dir, 'checkpoint_full.pth'))
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=3.5)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')

def visual_multiple(true, preds=None, name='./pic/test.pdf', vname=None):


    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(15, 3))

    for i in range(5):
        axes[i].plot(true[:, i], label='GroundTruth', linewidth=2)
        axes[i].plot(preds[:, i], label='Prediction', linewidth=2, linestyle='dotted')
        axes[i].set_title(vname[i] if vname is not None else f'Variable {i}')
        axes[i].legend()
    plt.tight_layout()
    plt.savefig(name, bbox_inches='tight')

def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)