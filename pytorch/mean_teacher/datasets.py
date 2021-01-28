# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import torchvision.transforms as transforms

from . import data
from .utils import export


@export
def imagenet():
    channel_stats = dict(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    train_transformation = data.TransformTwice(transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ]))
    eval_transformation = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        'datadir': 'data-local/images/ilsvrc2012/',
        'num_classes': 1000
    }


@export
def cifar10():
    channel_stats = dict(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470, 0.2435, 0.2616])
    train_transformation = data.TransformTwice(transforms.Compose([
        data.RandomTranslateWithReflect(4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ]))
    eval_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        'datadir': 'data-local/images/cifar/cifar10/by-image',
        'num_classes': 10
    }


@export
def asvspoof2019la():
    train_trans = data.TransformTwice(transforms.Compose([
        data.RandomTranslateWithReflect(4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]))
    eval_trans = transforms.Compose([
        transforms.ToTensor(),
    ])

    return {
        'attack_type': 'la',
        'root': '/home/vano/wrkdir/datasets/asvspoof2019',
        'train_meta': 'data-local/cyber/asvspoof19/meta/la/fold1_train.tsv',
        'val_meta': 'data-local/cyber/asvspoof19/meta/la/fold1_validation.tsv',
        'eval_meta': 'data-local/cyber/asvspoof19/meta/la/fold1_evaluation.tsv',
        'feat_storage': '',
        'scoring_path': 'la_scores/',
        'train_trans': train_trans,
        'eval_trans': eval_trans,
        'num_classes': 2
    }


@export
def asvspoof2019pa():
    demo = True

    # TODO transform, mixer

    train_trans = data.TransformTwice(transforms.Compose([
        data.RandomTranslateWithReflect(4),  # TODO add pad, random_crop or consecutive_slides
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]))

    eval_trans = transforms.Compose([
        transforms.ToTensor(),
    ])
    if demo:
        return {
            'attack_type': 'pa',
            'root': '/home/vano/wrkdir/datasets/asvspoof2019',
            'processed_meta': 'data-local/cyber/asvspoof19/meta/',
            # 'val_meta': 'data-local/cyber/asvspoof19/meta/pa/fold1_train.tsv',
            # 'eval_meta': 'data-local/cyber/asvspoof19/meta/pa/fold1_evaluation.tsv',
            'feat_storage': '/home/vano/wrkdir/projects_data/asvspoof2019/ASVspoof2019_PA_train/feats.scp',
            'scoring_path': 'pa_scores/',
            'train_trans': train_trans,
            'eval_trans': eval_trans,
            'num_classes': 2
        }
    else:
        return {
            'attack_type': 'pa',
            'root': '/home/vano/wrkdir/datasets/asvspoof2019',
            # 'train_meta': 'data-local/cyber/asvspoof19/meta/pa/fold1_train.tsv',
            # 'val_meta': 'data-local/cyber/asvspoof19/meta/pa/fold1_validation.tsv',
            # 'eval_meta': 'data-local/cyber/asvspoof19/meta/pa/fold1_evaluation.tsv',
            'feat_storage': '/home/vano/wrkdir/projects_data/antispoofing_speech/logspec/raw_fbank_ASVspoof2019_PA_train_spec.1.scp',
            'scoring_path': 'pa_scores/',
            'num_classes': 2
        }


@export
def aasg_daf():
    pass
