from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import shutil
from cyber.src.model import afnet, resnet, senet


def prepare_model(model_select, nclasses, nresnet_block, afn_upsample, afn_activation, focal_loss):
    # TODO fix model_select ~ model_class
    if model_select == 1:
        print('Inflate: ResNet')
        model = resnet.SpoofSmallResNet257_400(nclasses, nresnet_block, focal_loss)
    elif model_select == 5:
        print('Inflate: attentive filtering network')
        model = afnet.SpoofSmallAFNet257_400(nclasses, afn_upsample, afn_activation,
                                             nresnet_block, focal_loss)
    elif model_select == 6:
        print('Inflate: squeeze-and-excitation network')
        model = senet.se_resnet34(num_classes=nclasses, focal_loss=focal_loss)

    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('===> Model total parameter: {}'.format(model_params))
    return model


def save_checkpoint(state, is_best, path, filename='checkpoint.pth.tar'):
    torch.save(state, path + '/' + filename)
    if is_best:
        print("===> save to checkpoint at {}\n".format(path + '/' + 'model_best.pth.tar'))
        shutil.copyfile(path + '/' + filename, path + '/' + 'model_best.pth.tar')


def load_checkpoint(model, optimizer, filename):
    print("===> loading checkpoint '{}'".format(filename))
    checkpoint = torch.load(filename)
    if 'epoch' in checkpoint:
        print('===> Continue from epoch: %d' % checkpoint['epoch'])
    if 'metrics' in checkpoint:
        print('===> Metrics of the model: {}'.format(checkpoint['metrics']))
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("===> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint['epoch']))
    return checkpoint['epoch']


def test_model():
    model_params = {
        'MODEL_SELECT': 6,  # which model
        'NUM_SPOOF_CLASS': 10,  # x-class classification
        'FOCAL_GAMMA': None,  # gamma parameter for focal loss; if obj is not focal loss, set this to None
        'NUM_RESNET_BLOCK': 5,  # number of resnet blocks in ResNet
        'AFN_UPSAMPLE': 'Bilinear',  # upsampling method in AFNet: Conv or Bilinear
        'AFN_ACTIVATION': 'sigmoid',  # activation function in AFNet: sigmoid, softmaxF, softmaxT
        'NUM_HEADS': 3,  # number of heads for multi-head att in SAFNet
        'SAFN_HIDDEN': 10,  # hidden dim for SAFNet
        'SAFN_DIM': 'T',  # SAFNet attention dim: T or F
        'RNN_HIDDEN': 128,  # hidden dim for RNN
        'RNN_LAYERS': 4,  # number of hidden layers for RNN
        'RNN_BI': True,  # bidirection/unidirection for RNN
        'DROPOUT_R': 0.0,  # dropout rate
    }
    model = prepare_model(**model_params)
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('model contains {} parameters'.format(model_params))
    print(model)
    x = torch.randn(2, 1, 257, 400)
    output = model(x)
    print(output)


if __name__ == '__main__':
    test_model()
