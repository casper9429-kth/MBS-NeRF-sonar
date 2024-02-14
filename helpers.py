import torch
import matplotlib
matplotlib.use('Agg')
from MLP import *


torch.autograd.set_detect_anomaly(True) # Turn off for performance


def update_lr(optimizer,lr_decay):
    """Update learning rate of the optimizer"""
    for param_group in optimizer.param_groups:
        if param_group['lr'] > 0.0000001:
            param_group['lr'] = param_group['lr'] * lr_decay
            learning_rate = param_group['lr']
            print('learning rate is updated to ',learning_rate)
    return 0

def save_model(expID, model, i):
    """Save the model to the disk"""
    # save model
    model_name = './experiments/{}/model/epoch.pt'.format(expID)
    torch.save(model, model_name)
    return 0


