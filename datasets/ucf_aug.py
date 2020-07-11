import torch
import torch.utils.data as data
from PIL import Image
import os
import math
import functools
import json
import copy
import numpy as np
import random
from scipy.io import loadmat

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
        image_path = os.path.join(video_dir_path, '{:06d}.png'.format(i))
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
        anno = loadmat(os.path.join(annotation_path, subset, lists[i]))
        anno = anno['label'][0,0]

        
        video_names.append(lists[i][0:-4])
        annotations.append(anno)

    return video_names, annotations


def make_dataset(dataset_path, subset, sample_duration, n_samples_for_each_video, opt):
    dataset_path = os.path.join(dataset_path, 'ucf526')
    video_path = os.path.join(dataset_path, 'imgs')
          
    video_names, annotations = get_video_names_and_annotations(dataset_path, subset)

    dataset = []
    max_n_frames = 0


    for i in range(len(video_names)):
        if (i+1) % 50 == 0 or i+1 == len(video_names):
            print('{} dataset loading [{}/{}]'.format(subset, i+1, len(video_names)))

        video_path_i = os.path.join(video_path, subset, video_names[i][2:-8], video_names[i])
        if not os.path.exists(video_path_i):
            print('error', video_path_i)
            continue

        n_frames = int(annotations[i]['duration'][0,0])
        begin_t = int(annotations[i]['start_frame'][0,0])
        end_t = int(annotations[i]['end_frame'][0,0])
        next_t = annotations[i]['offset_next_estimate'][0,:]
        pre_t =  annotations[i]['offset_pre_estimate'][0,:]
        bound_t = annotations[i]['temporal_bound'][:,0]

        if n_frames <= 0 or len(bound_t) < 3:
            continue

        if n_samples_for_each_video == 1:
            sample = {
                'video': video_path_i,
                'segment': [begin_t, end_t],
                'n_frames': n_frames,
                'video_id': video_names[i],
                'label_next':  next_t,
                'label_pre':  pre_t,
                'frame_indices': list(range(begin_t, end_t + 1)),
                'counts': len(bound_t) - 1
            }

            max_n_frames = max(max_n_frames, sample['n_frames'])
            dataset.append(sample)
        else:
            if n_samples_for_each_video < 1:
                raise('error, n_samples_for_each_video should >=1\n')
            
            sample = {
                'video': video_path_i,
                'video_id': video_names[i],
            }


            for j in range(0, n_samples_for_each_video):
                sample_j = copy.deepcopy(sample)

                begin_j_p = random.randint(0, len(bound_t)-3)
                end_j_p = random.randint(begin_j_p + 2, len(bound_t))

                counts = end_j_p - begin_j_p
                if begin_j_p == 0:
                    begin_j = begin_t
                else:
                    begin_j = random.randint(bound_t[begin_j_p-1], bound_t[begin_j_p]) 

                if end_j_p == len(bound_t):
                    end_j = end_t
                else:
                    end_j = random.randint(bound_t[end_j_p-1], bound_t[end_j_p]) 
                end_j = min(end_j, begin_j + sample_duration - 1)



                sample_j['segment'] = [begin_j, end_j]
                sample_j['n_frames'] = end_j - begin_j + 1
                sample_j['label_next'] = next_t[begin_j - begin_t: end_j - begin_t + 1]
                sample_j['label_pre'] = pre_t[begin_j - begin_t: end_j - begin_t + 1]
                sample_j['frame_indices'] = list(range(begin_j, end_j + 1))
                sample_j['counts'] = counts

                max_n_frames = max(max_n_frames, sample_j['n_frames'])
            
                dataset.append(sample_j)

    max_n_frames = max(max_n_frames, sample_duration)
    print('[size of dataset, max_n_frames] =  ', len(dataset), max_n_frames)

    return dataset, max_n_frames


class UCF_AUG(data.Dataset):
 

    def __init__(self,
                 dataset_path,
                 subset,
                 sample_duration,
                 opt,
                 n_samples_for_each_video=10,
                 spatial_transform=None,
                 get_loader=get_default_video_loader):
        print('dataset_path, subset, sample_duration, n_samples_for_each_video: ', dataset_path, subset, sample_duration, n_samples_for_each_video)
        self.data, self.max_n_frames = make_dataset(dataset_path, subset, sample_duration, n_samples_for_each_video, opt)

        self.spatial_transform = spatial_transform
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

        if clip.size(1) != self.max_n_frames:
            clip_zeros = torch.zeros([clip.size(0), self.max_n_frames - clip.size(1), clip.size(2), clip.size(3)], dtype=torch.float)
            clip = torch.cat([clip, clip_zeros], dim=1)
        
        label_next = np.zeros(self.max_n_frames, dtype=np.int32)
        label_pre = np.zeros(self.max_n_frames, dtype=np.int32)

        for i in range(0, self.data[index]['n_frames']):
            label_next[i] = self.data[index]['label_next'][i]
            label_pre[i] = self.data[index]['label_pre'][i]

        return clip, label_next, label_pre, self.data[index]['counts'], self.data[index]['n_frames']

    def __len__(self):
        return len(self.data)
