import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
import src.utils.io as io
import src.utils.dirs as dirs


def process_meta(in_meta, out_meta):
    """ ASVSpoof2019 process meta data
    in_meta:
    out_meta:
    """
    # TODO check if all files are existing
    cols = ['file0', 'file', 'info', 'attack_id', 'class_label']
    pd_meta = io.load_csv(in_meta, col_name=cols, delim=' ')

    file_ids = pd_meta['file']
    attack_ids = pd_meta['attack_id']
    class_labels = pd_meta['class_label']

    cls_le = _labels_encoder(class_labels)
    att_le = _labels_encoder(attack_ids)

    print('Attack ids:', att_le.classes_)
    print('Class labels:', cls_le.classes_)
    out_res = pd_meta[['file', 'class_label']]
    out_res.to_csv(out_file, index=False, sep='\t')
    # io.save_csv(out_meta, list(zip(file_ids, class_labels)), delim='\t')


def _labels_encoder(df_col):
    """
    prepare labels encoder from string to digits
    """
    le = LabelEncoder()
    labels_list = df_col.astype(str)
    le.fit(labels_list)
    return le


if __name__ == '__main__':
    # TODO put this to the pipeline function
    data_root = '/home/vano/wrkdir/datasets/asvspoof2019/'
    feat_dir = '/home/vano/wrkdir/projects_data/antispoofing_speech/logspec/'
    out_dir = 'meta/'  # data/asvspoof19/meta/

    in_meta_files = {'task': ['PA', 'LA'],
                     'PA': {'train': 'PA/ASVspoof2019_PA_cm_protocols/ASVspoof2019.PA.cm.train.trn.txt',
                            'dev': 'PA/ASVspoof2019_PA_cm_protocols/ASVspoof2019.PA.cm.dev.trl.txt',
                            'eval': 'PA/ASVspoof2019_PA_cm_protocols/ASVspoof2019.PA.cm.eval.trl.txt'},
                     'LA': {'train': 'LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt',
                            'dev': 'LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt',
                            'eval': 'LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt'}}

    for t in in_meta_files['task']:
        for dset, fname in in_meta_files[t].items():

            out_meta_dir = os.path.join(out_dir, t.lower())
            dirs.mkdir(out_meta_dir)

            if dset == 'dev':
                out_file = os.path.join(out_meta_dir, 'fold1_validation.tsv')
            elif dset == 'eval':
                out_file = os.path.join(out_meta_dir, 'fold1_evaluation.tsv')
            else:
                out_file = os.path.join(out_meta_dir, 'fold1_' + dset + '.tsv')
            in_meta = os.path.join(data_root, fname)
            process_meta(in_meta, out_file)
