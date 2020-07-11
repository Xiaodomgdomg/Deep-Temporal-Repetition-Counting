import argparse
import time
import os

def parse_opts():

    learning_policy = '2stream'
    validate_policy = '2stream'
    parser = argparse.ArgumentParser()

    # paths
    parser.add_argument(
        '--root_path',
        default= './data/',
        type=str,
        help='Root directory path of data')
    parser.add_argument(
        '--dataset_path',
        default='ori_data/', 
        type=str,
        help='Directory path of Videos')
    parser.add_argument(
        '--result_path',
        default='results/' +time.strftime('%m%d-%H:%M_',time.localtime(time.time()))+learning_policy,
        type=str,
        help='Result directory path')
    parser.add_argument(
        '--train_dataset',
        default='ucf_aug',
        type=str,
        help='')
    parser.add_argument(
        '--val_dataset',
        default= ['ucf_aug','quva',  'yt_seg' ],
        type=str,
        help='Used dataset (yt_seg | quva | ucf_aug)')

    # button
    parser.add_argument(
        '--no_train',
        action='store_true',
        help='If true, training is not performed.')
    parser.set_defaults(no_train=False)
    parser.add_argument(
        '--no_val',
        action='store_true',
        help='If true, validation is not performed.')
    parser.set_defaults(no_val=False)

    # training argument
    parser.add_argument(
        '--sample_duration',
        default=300,
        type=int,
        help='Temporal duration of training sample')
    parser.add_argument(
        '--mean_std_dataset',
        default='quva',
        type=str,
        help=
        '')
    parser.add_argument(
        '--sample_size',
        default=112,
        type=int,
        help='Height and width of inputs')
    parser.add_argument(
        '--batch_size', default=24, type=int, help='Batch Size') #32
    parser.add_argument(
        '--val_batch_size', default=5, type=int, help='Batch Size') #32
    parser.add_argument(
        '--n_epochs',
        default=100,
        type=int,
        help='Number of total epochs to run')
    parser.add_argument(
        '--lr_patience',
        default=1,
        type=int,
        help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.'
    )
    parser.add_argument(
        '--begin_epoch',
        default=1,
        type=int,
        help=
        'Training begins at this epoch. Previous trained model indicated by resume_path is loaded.'
    )
    parser.add_argument(
        '--pretrain_path',
        default= '', # 'weights/resnext-101-kinetics.pth', 
        type=str)
    parser.add_argument(
        '--train_from_scratch',
        action='store_true')
    parser.set_defaults(train_from_scratch=True)
    parser.add_argument(
        '--resume_path',
        default= '', # 'weights/resnext101_ucf526.pth', 
        type=str,
        help='Save data (.pth) of previous training')
    parser.add_argument(
        '--checkpoint',
        default=1,
        type=int,
        help='Trained model is saved at every this epochs.')

    # learning policy
    parser.add_argument(
        '--learning_policy',
        default=learning_policy,
        type=str,
        help='')
    parser.add_argument(
        '--validate_policy',
        default=validate_policy,
        type=str)
    parser.add_argument(
        '--optimizer',
        default='adam',
        type=str,
        help='Currently only support [adam, sgd]')
    parser.add_argument(
        '--learning_rate',
        default=0.00005,
        type=float,
        help=
        'Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument(
        '--weight_decay', default=1e-3, type=float, help='Weight Decay')
    parser.add_argument('--nesterov', action='store_true', help='Nesterov momentum')
    parser.set_defaults(nesterov=False)
    parser.add_argument('--dampening', default=0.9, type=float, help='dampening of SGD')


    # network argument
    parser.add_argument(
        '--basic_duration',
        default=32, # 48
        type=float,
        help='Temporal duration of network input')
    parser.add_argument(
        '--l_context_ratio',
        default=1.0,
        type=float,
        help='')
    parser.add_argument(
        '--r_context_ratio',
        default=2.0,
        type=float,
        help='')
    parser.add_argument(
        '--norm_value',
        default=255,
        type=int,
        help=
        'If 1, range of inputs is [0-255]. If 255, range of inputs is [0-1].')
    parser.add_argument(
        '--model',
        default='resnext',
        type=str,
        help='(resnet | resnext')
    parser.add_argument(
        '--model_depth',
        default=101,
        type=int,
        help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument(
        '--resnet_shortcut',
        default='B',
        type=str,
        help='Shortcut type of resnet (A | B)')
    parser.add_argument(
        '--resnext_cardinality',
        default=32,
        type=int,
        help='ResNeXt cardinality')
    parser.add_argument(
        '--n_classes',
        default=7, 
        type=int,
        help=
        '[count, enlarge, narrow, miss]'
    )
    parser.add_argument(
        '--anchors',
        default=[0.5, 0.67, 0.8, 1.0, 1.25, 1.5, 2.0], 
        type=float
    )
    parser.add_argument(
        '--iou_ubound',
        default=0.5,
        type=float
    )
    parser.add_argument(
        '--iou_lbound',
        default=0.5,
        type=float
    )

    # hardware argument
    parser.add_argument(
        '--no_cuda', action='store_true', help='If true, cuda is not used.')
    parser.set_defaults(no_cuda=False)
    parser.add_argument(
        '--n_threads',
        default=1,
        type=int,
        help='Number of threads for multi-thread loading')
    parser.add_argument(
        '--manual_seed', default=1, type=int, help='Manually set random seed')

    # reserved argument
    parser.add_argument(                                                                                                                                                
        '--initial_scale',
        default=1.0,
        type=float,
        help='Initial scale for multiscale cropping')
    parser.add_argument(
        '--n_scales',
        default=1,
        type=int,
        help='Number of scales for multiscale cropping')
    parser.add_argument(
        '--scale_step',
        default=0.9457416090031758, 
        type=float,
        help='Scale step for multiscale cropping')
    parser.add_argument(
        '--train_crop',
        default='center',
        type=str,
        help=
        'Spatial cropping method in training. random is uniform. corner is selection from 4 corners and 1 center.  (random | corner | center)'
    )
    
    args = parser.parse_args()

    return args
