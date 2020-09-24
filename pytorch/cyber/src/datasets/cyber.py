"""
ASVSpoof 2015 / 2017 / 2019 dataset classes

Note: Dataset class is from Pytorch
"""
import os
import pandas as pd
from enum import Enum
import sklearn.preprocessing as sk_proc
import cyber.src.base.datasets as bd
import cyber.src.utils.io as io
import cyber.src.datasets.datasets as ds
import cyber.src.utils.dirs as dirs
import torch.utils.data as thd


class CyberDataProcessor(bd.BaseDataProcessor):
    """
    Dataset handler facade.
    # Arguments
        config: dict. Parameters of data loader.
        pipe_mode: String. 'development' or 'submission'
    """

    class PipeMode(object):
        DEV = 'development'
        SUBMIT = 'submit'

    def __init__(self, config, pipe_mode, **kwargs):
        super(CyberDataProcessor, self).__init__(config)
        self.pipe_mode = pipe_mode
        self.attack_type = kwargs.get('attack_type')
        self.overwrite = kwargs.get('overwrite', False)
        self.data_root = self.config['data_paths']['root']
        self.meta_dir = self.config['data_paths']['processed_meta']
        self._meta = None

    @property
    def meta_data(self):
        if self._meta:
            return self._meta
        else:
            raise Exception('[ERROR] meta is not initialized: run initialize()')

    def initialize(self):
        print('[INFO] Initialize data...')
        if self.pipe_mode == CyberDataProcessor.PipeMode.DEV:
            self._meta = ASVSpoof19Meta(data_dir=self.data_root,
                                        meta_dir=self.meta_dir,
                                        folds_num=1,  # default
                                        attack_type=self.attack_type)
        elif self.pipe_mode == CyberDataProcessor.PipeMode.SUBMIT:
            raise NotImplementedError('[ERROR] Not implemented yet')

    def extract_features(self):
        #  TODO extract all features and store in
        # self.config['data_path']['feat_storage']
        pass

    def train_dataloader(self):
        # TODO by default we have one fold, otherwise we need to initiate the class outside
        # ds.ArkDataGenerator(data_file, fold_list, meta_data, transformer, mixer, **kwargs)
        fold_list = self.meta_data.fold_list(fold=1, data_split=ASVSpoof19Meta.DataSplit.train)
        dgen = ds.ArkDataGenerator(data_file=self.config['data_paths']['feat_storage'],
                                   # utt2index_file=self.config['data_paths']['train_utt2index'],
                                   fold_list=fold_list,
                                   rand_slides=True)
        kwargs = {'num_workers': 2, 'pin_memory': True} if self.config['use_cuda'] else {}
        dl = thd.DataLoader(dgen, batch_size=self.config['batch_size'], shuffle=True, **kwargs)
        return dl

    def val_dataloader(self):
        fold_list = self.meta_data.fold_list(fold=1, data_split=ASVSpoof19Meta.DataSplit.validation)
        dgen = ds.ArkDataGenerator(data_file=self.config['data_paths']['feat_storage'],
                                   fold_list=fold_list,
                                   rand_slides=False)
        kwargs = {'num_workers': 2, 'pin_memory': True} if self.config['use_cuda'] else {}
        dl = thd.DataLoader(dgen, batch_size=self.config['test_batch_size'], shuffle=False, **kwargs)
        return dl

    def eval_dataloader(self):
        pass


class ASVSpoof19Meta(bd.BaseMeta):
    """
    ASVSpoof 2019 dataset meta lists.
    Ref: https://www.asvspoof.org/
    Dataset has the next default splits: [train, dev, eval]

    Meta data format: we use tsv format
        ['file' 'class_label']
        e.g. [path/file.wav  lab1;lab2;lab3], support multi-label format

    # Arguments
        data_dir: root path to the dataset
        meta_dir: path to the processed metadata lists. Format of the folder: fold{1,...}_{train, dev, eval}.tsv
        folds_num: Integer. number of fold splits in the dataset (default: 1)
    TODO MetaDataset architecture :
    https://github.com/LCAV/pyroomacoustics/blob/master/pyroomacoustics/datasets/base.py
    """

    class AttackType(Enum):
        PA = 'pa'
        LA = 'la'

    class MetaCol(object):
        FILE = 'file'
        LAB = 'class_label'

    def __init__(self, data_dir, meta_dir, folds_num=1, **kwargs):
        super(ASVSpoof19Meta, self).__init__()
        self.data_dir = data_dir
        self.meta_dir = meta_dir
        self.folds_num = folds_num

        self.is_multilabel = False
        self.attack_type = kwargs.get('attack_type')
        self._meta_file_template = 'fold{num}_{data_split}.tsv'
        self.lencoder = self._labels_encoder()
        print('[INFO] classes in metadata:', self.lencoder.classes_)

    @property
    def label_names(self):
        """Return list of labels in string format"""
        return list(self.lencoder.classes_)

    @property
    def nfolds(self):
        """Return list of folds indices"""
        return range(1, self.folds_num + 1)

    def fold_list(self, fold, data_split):
        """
        fold: Integer. Number of fold to return
        data_type: String. 'train', 'validation' or 'test' for development lists
        return: list. File names and labels hot vector
        """
        if not (data_split in self.DataSplit):
            raise AttributeError('[ERROR] No dataset type: %s' % data_split)
        flist = self._load_fold_list(fold, data_split)
        return self._format_list(flist)  # file_name, hot_vecs

    def file_list(self):
        """
        Merge file lists of the development dataset (training + validation + test)
        return: list. File names and labels hot vector
        """
        mrg = pd.DataFrame()
        for dt in self.DataSplit:
            fl = self._load_fold_list(fold=1, data_split=dt)
            mrg = mrg.append(fl)
        mrg.reset_index(drop=True, inplace=True)
        return self._format_list(mrg)  # file_name, hot_vecs

    def labels_str_encode(self, lab_dig):
        """
        Transform hot-vector to string 'class_label' format
        lab_dig: list.
        """
        return list(self.lencoder.inverse_transform(lab_dig))

    def labels_dig_encode(self, lab_str):
        """
        Transform 'class_label' to hot-vector format
        lab_str: list.
        """
        return self.lencoder.transform(lab_str)

    def _labels_encoder(self):
        """
        prepare labels encoder from string to digits
        """
        pd_meta = self._load_fold_list(fold=1, data_split=self.DataSplit.train)
        labels_list = pd_meta[self.MetaCol.LAB].astype(str)
        if self.is_multilabel:
            le = sk_proc.MultiLabelBinarizer()
            labels_list = labels_list.str.split(self.LABEL_SEPARATOR)
        else:
            le = sk_proc.LabelEncoder()
        le.fit(labels_list)
        return le

    def _load_fold_list(self, fold, data_split):
        fn = self._meta_file_template.format(num=fold, data_split=data_split.value)
        meta_file = os.path.join(self.meta_dir, self.attack_type, fn)
        col = [self.MetaCol.FILE, self.MetaCol.LAB]
        pd_meta = io.load_csv(meta_file, col_name=col, delim='\t', header=0)
        return pd_meta

    def _format_list(self, df):
        fnames = list(df[self.MetaCol.FILE])
        # labels to hot-vectors
        # labs = df[self.MetaCol.LAB].str.split(self.LABEL_SEPARATOR)
        hot_vecs = list(self.labels_dig_encode(df[self.MetaCol.LAB]))
        return fnames, hot_vecs


class ASVSpoof17Meta(bd.BaseMeta):
    pass


class ASVSpoof15Meta(bd.BaseMeta):
    pass


class DFDCMeta(bd.BaseMeta):
    pass
