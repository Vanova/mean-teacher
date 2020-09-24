"""
Classes for working with datasets:
 load data, meta data, data batches generator
"""
import h5py
from enum import Enum


class BaseDataProcessor(object):
    """
    Abstract class for accessing data: meta info,
    processing and generating data
    """

    def __init__(self, config):
        self.config = config

    @property
    def meta_data(self):
        raise NotImplementedError

    def initialize(self):
        """
        Load data, prepare data meta information
        """
        raise NotImplementedError

    def extract_features(self):
        raise NotImplementedError

    def train_dataloader(self):
        """
        Lazy method, only for small datasets
        # Return
            x, y - whole data
        """
        raise NotImplementedError

    def val_dataloader(self):
        """
        Lazy method, only for small datasets
        # Return
            x, y - whole data
        """
        raise NotImplementedError

    def eval_dataloader(self):
        """
        Lazy method, only for small datasets
        # Return
            x - whole data without labels
        """
        raise NotImplementedError


class BaseMeta(object):
    LABEL_SEPARATOR = ';'

    class DataSplit(Enum):
        train = 'train'
        validation = 'train' #'validation'
        test = 'test'
        evaluation = 'evaluation'

    @property
    def label_names(self):
        raise NotImplementedError

    @property
    def nfolds(self):
        raise NotImplementedError

    def fold_list(self, fold, data_split):
        raise NotImplementedError

    def file_list(self):
        raise NotImplementedError

    def labels_str_encode(self, lab_dig):
        raise NotImplementedError

    def labels_dig_encode(self, lab_str):
        raise NotImplementedError

    def filter(self, **kwargs):
        raise NotImplementedError

    def join(self, meta):
        raise NotImplementedError

    def make_folds(self, n):
        raise NotImplementedError


class BaseGenerator(object):
    def __init__(self, data_file, batch_sz=1, window=1, fold_list=None):
        self.data_file = data_file
        self.batch_sz = batch_sz
        self.window = window
        self.fold_list = fold_list

        self.hdf = h5py.File(self.data_file, 'r')
        if self.fold_list:
            print('Generate from the FOLD list: %d files' % len(set(self.fold_list[0])))
        else:
            self.fold_list = list(self.hdf.keys())
            print('Generate from the WHOLE dataset: %d files' % len(self.fold_list))

    def batch(self):
        raise NotImplementedError

    def batch_shape(self):
        raise NotImplementedError

    def samples_number(self):
        raise NotImplementedError

    def stop(self):
        self.hdf.close()
