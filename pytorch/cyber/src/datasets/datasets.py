"""
    General classes for data sampling from ARK, HDF5 storage.
    Ideally generators should not know anything about dataset it generate from,
    information about dataset comes with MetaData class object.
"""
import numpy as np
import kaldiio as kio
import torch.utils.data as thd
import cyber.src.utils.io as io


class ArkDataGenerator(thd.Dataset):
    """
    Generate sequence of observations.
    # Arguments
        data_file: String. SCP data file in Kaldi format
        fold_list: list.
        meta_data: BaseMeta.
        transformer: BaseTransformer.
        mixer: BaseTransformer.
    # Output
        Slice ARK dataset and return batch: [smp x bands x frames x channel]
    """
    # TODO (data_file, fold_list, meta_data, transformer, mixer, **kwargs)
    #  train_transformation = data.TransformTwice(transforms.Compose([
    #         data.RandomTranslateWithReflect(4),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize(**channel_stats)
    #     ]))
    def __init__(self, data_file, fold_list=None, transform=None, mixer=None, **kwargs):
        self.data_file = data_file
        self.fold_list = fold_list
        self.transform = transform
        self.mixer = mixer

        self.wnd_size = kwargs.get('wnd_size', 400)
        self.rand_slides = kwargs.get('rand_slides', False)
        self.nslides = kwargs.get('nslides', 1)  # 4

        self.scp_feats = io.load_dictionary(data_file)
        feat_keys = list(self.scp_feats.keys())
        fold_keys = self.fold_list[0]
        self.labels = self.fold_list[1]

        # intersect file_ids from fold_list and file_ids from data_file
        dif = set(fold_keys) - set(feat_keys)
        if len(dif):
            print('[WARNING] no features for files: %d / %d' % (len(dif), len(fold_keys)))

        self.file_keys = list(set(fold_keys) & set(feat_keys)) #[:50]  # TODO check
        print('[INFO] sample from %d files' % len(self.file_keys))

    def __len__(self):
        return len(self.file_keys)

    def __getitem__(self, counter):
        fkey = self.file_keys[counter]
        # features in Kaldi have dim [nframes x nbanks] -> [nbanks, nframes]
        feats = kio.load_mat(self.scp_feats[fkey][0]).T

        ##########
        feats = feats.copy()
        y = self.labels[counter].clone()

        if self.transform is not None:
            feats = self.transform(feats)

        if self.mixer is not None:
            feats, y = self.mixer(self, feats, y)

        ##########

        # slide utterance
        pad_feats = self._pad_utterance(feats, self.wnd_size)
        if self.rand_slides:
            # TODO implement as TransformTwice
            slides = self._random_slides(pad_feats, self.wnd_size, self.nslides)
            slides1 = self._random_slides(pad_feats, self.wnd_size, self.nslides)
        else:
            slides = self._consecutive_slides(pad_feats, self.wnd_size)

        utt_ids = [fkey] * slides.shape[0]
        ys = [self.labels[counter]] * slides.shape[0]
        return utt_ids, slides, slides1, ys
        # return slides, ys

    @staticmethod
    def _pad_utterance(feat, wnd_size):
        """
        feat: ndarray, [bands, frames]
        wnd_size: Integer, size of sliding window
        return: fixed utterance features
        """
        init_len = feat.shape[1]
        max_len = int(wnd_size * np.ceil(float(init_len) / wnd_size))
        # in case when utterance is shorter than window
        rep = max_len // init_len
        tensor = np.tile(feat, rep)
        # padding
        rest_n = int(max_len % init_len)
        tensor = np.pad(tensor, ((0, 0), (0, rest_n)), 'wrap')
        return tensor

    @staticmethod
    def _consecutive_slides(feat, wnd_size):
        rep = feat.shape[1] // wnd_size
        rep = 2 * rep - 1  # slides
        hop = wnd_size // 2
        slides = []
        for i in range(rep):
            s = feat[:, hop * i:hop * i + wnd_size]
            s = np.expand_dims(s, axis=0)
            slides.append(s)
        return np.array(slides)

    @staticmethod
    def _random_slides(feat, wnd_size, nslides):
        end = feat.shape[1] - wnd_size + 1
        # start = np.random.randint(0, end)
        starts = np.random.randint(0, end, size=nslides)
        starts.sort()
        chunks = np.zeros((nslides, 1, feat.shape[0], wnd_size), dtype=np.float32)
        for id, s in enumerate(starts):
            buf = feat[:, s:s + wnd_size]
            chunks[id] = np.expand_dims(buf, axis=0) # TODO no need?
        return chunks


class HDFDataGenerator(thd.Dataset):
    pass





#
# class SpoofDatsetFilebase(D.Dataset):
#     ''' multi-class classification for PA: AA, AB, AC, BA, BB, BC, CA, CB, CC --> 10 classes
#         (bonafide: 0), (AA: 1), (AB: 2), (AC: 3), (BA: 4), (BB: 5), (BC: 6),
#         (CA: 7), (CB: 8), (CC: 9)
#
#         multi-class classification for LA: SS_1, SS_2, SS_4, US_1, VC_1, VC_4 --> 7 classes
#         (bonafide: 0), (SS_1: 1), (SS_2: 2), (SS_4: 3), (US_1: 4), (VC_1: 5), (VC_4: 6)
#
#         if leave_one_out:
#             for pa: leave out the class with label == 9
#             for la: leave out the class with label == 6
#     '''
#
#     def __init__(self, file_list, slide_wnd=400):
#         self.wnd_size = slide_wnd
#         self.file_list = file_list
#
#     def __len__(self):
#         return len(self.file_list)
#
#     def __getitem__(self, counter):
#         fname = self.file_list[counter]
#         feats = np.load(fname)
#         # slide utterance
#         pad_feats = self._pad_utterance(feats, self.wnd_size)
#         slides = self._consecutive_slides(pad_feats, self.wnd_size)
#         return fname, slides
#
#     @staticmethod
#     def _pad_utterance(feat, wnd_size):
#         """
#         feat: ndarray, [bands, frames]
#         wnd_size: Integer, size of sliding window
#         return: fixed utterance features
#         """
#         init_len = feat.shape[1]
#         max_len = int(wnd_size * np.ceil(float(init_len) / wnd_size))
#         # in case when utterance is shorter than window
#         rep = max_len // init_len
#         tensor = np.tile(feat, rep)
#         # padding
#         rest_n = int(max_len % init_len)
#         tensor = np.pad(tensor, ((0, 0), (0, rest_n)), 'wrap')
#         return tensor
#
#     @staticmethod
#     def _consecutive_slides(feat, wnd_size):
#         rep = feat.shape[1] // wnd_size
#         rep = 2 * rep - 1  # slides
#         hop = wnd_size // 2
#         slides = []
#         for i in range(rep):
#             s = feat[:, hop * i:hop * i + wnd_size]
#             s = np.expand_dims(s, axis=0)
#             slides.append(s)
#         return np.array(slides)
#
#     @staticmethod
#     def _random_slides(feat, wnd_size, nslides):
#         end = feat.shape[1] - wnd_size + 1
#         # start = np.random.randint(0, end)
#         starts = np.random.randint(0, end, size=nslides)
#         starts.sort()
#         chunks = np.zeros((nslides, 1, feat.shape[0], wnd_size), dtype=np.float32)
#         for id, s in enumerate(starts):
#             buf = feat[:, s:s + wnd_size]
#             chunks[id] = np.expand_dims(buf, axis=0)
#         return chunks
