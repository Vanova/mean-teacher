import cyber.src.utils.dirs as dirs

is_demo = True
pipe_mode = 'development'
attack_type = 'pa'

# if is_demo:
#     # NOTE: the philosophy is meta dir contains fold#_{train,test,val}.tsv,
#     #  and feat_storage contains ALL features at once
#     data_paths = {
#         'root': '/home/vano/wrkdir/datasets/asvspoof2019',
#         'processed_meta': '../data/asvspoof19/meta',
#         'feat_storage': '/home/vano/wrkdir/projects_data/antispoofing_speech/logspec/raw_fbank_ASVspoof2019_PA_train_spec.1.scp',
#         'scoring_path': 'system/%s_scores/' % attack_type,
#     }
# else:
#     data_paths = {
#         'root': '/home/vano/wrkdir/datasets/asvspoof2019',
#         'processed_meta': '../data/asvspoof19/meta',
#         'feat_storage': '../data/asvspoof19/meta/%s/feats.scp' % attack_type,
#         'scoring_path': 'system/%s_scores/' % attack_type,
#     }
# dirs.mkdir(data_paths['scoring_path'])

model_params = {
    'model_select': 6,  # which model
    'nclasses': 2,  # 2,  # LA: 7 or PA: 10 x-class classification
    'focal_loss': None,  # gamma parameter for focal loss; if obj is not focal loss, set this to None
    'nresnet_block': 5,  # number of resnet blocks in ResNet
    'afn_upsample': 'Bilinear',  # upsampling method in AFNet: Conv or Bilinear
    'afn_activation': 'sigmoid',  # activation function in AFNet: sigmoid, softmaxF, softmaxT
}

# metrics = ['eer_avg', 'eer_pool']
# monitor = 'eer_avg'
# max_patience = 5
batch_size = 32  # 64
test_batch_size = 1  # per utterance, TODO check how to collate
epochs = 3  # 20 for PA, 10 for LA
start_epoch = 1
n_warmup_steps = 1000
log_interval = 10
pretrain_file = './'  # '../pretrained/pa/senet34_py3'  # None