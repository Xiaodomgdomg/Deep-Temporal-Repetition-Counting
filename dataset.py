# from datasets.kinetics import Kinetics
from datasets.quva import QUVA
from datasets.ucf_aug import UCF_AUG
from datasets.yt_seg import YT_SEG

def get_training_set(opt, spatial_transform, target_transform):
    assert opt.train_dataset in ['ucf_aug']

    if opt.train_dataset == 'ucf_aug':
        training_data = UCF_AUG(
            opt.dataset_path,
            'train',
            sample_duration=opt.sample_duration,
            opt=opt,
            n_samples_for_each_video=10,
            spatial_transform=spatial_transform)

    return training_data


def get_validation_set(dataset, spatial_transform, target_transform, opt):
    assert dataset in ['quva', 'ucf_aug', 'yt_seg']

    if dataset == 'quva':
        validation_data = QUVA(
            opt.dataset_path,
            'val',
            sample_duration=opt.sample_duration,
            n_samples_for_each_video=1,
            spatial_transform=spatial_transform)
    elif dataset == 'ucf_aug':
        validation_data = UCF_AUG(
            opt.dataset_path,
            'val',
            sample_duration=opt.sample_duration,
            opt=opt,
            n_samples_for_each_video=1,
            spatial_transform=spatial_transform)
    elif dataset == 'yt_seg':
        validation_data = YT_SEG(
            opt.dataset_path,
            'val',
            sample_duration=opt.sample_duration,
            n_samples_for_each_video=1,
            spatial_transform=spatial_transform)
    
    return validation_data

