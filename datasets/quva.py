import torch
import torch.utils.data as data
from PIL import Image
import os
import math
import functools
import json
import copy
import numpy as np

from utils.utils import load_value_file


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def video_loader(video_dir_path, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, 'image_{:05d}.jpg'.format(i))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video

    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


def get_video_names_and_annotations(dataset_path, subset):
    annotation_path = os.path.join(dataset_path, 'annotations')

    video_names = []
    annotations = []
    
    lists = os.listdir(os.path.join(annotation_path,subset))
    lists.sort()

    for i in range(len(lists)):
        anno = np.load(os.path.join(annotation_path, subset, lists[i]))
        video_names.append(lists[i][0:-4])
        annotations.append(anno)

    return video_names, annotations


def make_dataset(dataset_path, subset, sample_duration, n_samples_for_each_video):
    dataset_path = os.path.join(dataset_path, 'QUVA')
    video_path = os.path.join(dataset_path, 'imgs')

    video_names, annotations = get_video_names_and_annotations(dataset_path, subset)

    dataset = []
    max_n_frames = 0


    for i in range(len(video_names)):
        if (i+1) % 50 == 0 or i+1 == len(video_names):
            print('{} dataset loading [{}/{}]'.format(subset, i+1, len(video_names)))

        video_path_i = os.path.join(video_path, subset, video_names[i])
 
        if not os.path.exists(video_path_i):
            continue

        n_frames_file_path = os.path.join(video_path_i, 'n_frames')
        n_frames = int(load_value_file(n_frames_file_path))
        max_n_frames = max(max_n_frames, n_frames)
        if n_frames <= 0:
            continue

        begin_t = 1
        end_t = n_frames
        sample = {
            'video': video_path_i,
            'segment': [begin_t, end_t],
            'n_frames': n_frames,
            'video_id': video_names[i][0:3],
            'label':  annotations[i]
        }

        if n_samples_for_each_video == 1:
            sample['frame_indices'] = list(range(1, n_frames + 1))
            dataset.append(sample)
        else:
            if n_samples_for_each_video > 1:
                step = max(1,
                           math.ceil((n_frames - 1 - sample_duration) /
                                     (n_samples_for_each_video - 1)))
                step = int(step)
            else:
                raise('error, n_samples_for_each_video should >=1\n')
                # step = sample_duration
            for j in range(1, n_frames-sample_duration, step):
                sample_j = copy.deepcopy(sample)
                sample_j['frame_indices'] = list(
                    range(j, min(n_frames + 1, j + sample_duration)))
                dataset.append(sample_j)

    return dataset, max_n_frames


class QUVA(data.Dataset):

    def __init__(self,
                 dataset_path,
                 subset,
                 sample_duration,
                 n_samples_for_each_video=10,
                 spatial_transform=None,
                 target_transform=None,
                 get_loader=get_default_video_loader):
        self.data, self.max_n_frames = make_dataset(dataset_path, subset, sample_duration, n_samples_for_each_video)

        self.spatial_transform = spatial_transform
        self.target_transform = target_transform
        self.loader = get_loader()

        self.mean = [0.0,0.0,0.0]
        self.var = [0.0,0.0,0.0]
        self.readed_num = 0.0

    def __getitem__(self, index):

        path = self.data[index]['video']

        frame_indices = self.data[index]['frame_indices']

        clip = self.loader(path, frame_indices)
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
        
        target = self.data[index]['label']

        sample_len = clip.size(1)
        if clip.size(1) != self.max_n_frames:
            clip_zeros = torch.zeros([clip.size(0), self.max_n_frames - clip.size(1), clip.size(2), clip.size(3)], dtype=torch.float)
            clip = torch.cat([clip, clip_zeros], dim=1)

        return clip, -1, -1, len(target), sample_len

    def __len__(self):
        return len(self.data)
