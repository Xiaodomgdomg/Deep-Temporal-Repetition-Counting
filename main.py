import os
import sys
import json
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from collections import OrderedDict

from opts import parse_opts
from models.model import generate_model
from utils.mean import get_mean, get_std
from utils.spatial_transforms import (
    Compose, Normalize, Scale_shorterside, Scale_longerside, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)
from utils.temporal_transforms import LoopPadding, TemporalRandomCrop
from utils.target_transforms import ClassLabel, VideoID
from utils.target_transforms import Compose as TargetCompose

from dataset import get_training_set, get_validation_set
from utils.utils import Logger


if __name__ == '__main__':
    import sys 
    print(sys.version)
    print(torch.__version__)

    opt = parse_opts()
    if opt.root_path != '':
        opt.dataset_path = os.path.join(opt.root_path, opt.dataset_path)
        opt.result_path = os.path.join(opt.root_path, opt.result_path)
        if not os.path.exists(opt.result_path):
            os.makedirs(opt.result_path) 
        if opt.resume_path:
            opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
        if opt.pretrain_path:
            opt.pretrain_path = os.path.join(opt.root_path, opt.pretrain_path)

    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    opt.mean = get_mean(opt.norm_value, dataset=opt.mean_std_dataset)
    opt.std = get_std(opt.norm_value, dataset=opt.mean_std_dataset)
    print(opt)
    with open(os.path.join(opt.result_path, 'opts.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)

    torch.manual_seed(opt.manual_seed)

    
    criterion = nn.CrossEntropyLoss()
    if not opt.no_cuda:
        criterion = criterion.cuda()

    norm_method = Normalize(opt.mean, opt.std)

    model, parameters = generate_model(opt)
    print(model)

    if not opt.no_train:
        assert opt.train_crop in ['random', 'corner', 'center']
        if opt.train_crop == 'random':
            crop_method = MultiScaleRandomCrop(opt.scales, opt.sample_size)
        elif opt.train_crop == 'corner':
            crop_method = MultiScaleCornerCrop(opt.scales, opt.sample_size)
        elif opt.train_crop == 'center':
            crop_method = MultiScaleCornerCrop(
                opt.scales, opt.sample_size, crop_positions=['c'])

        spatial_transform = Compose([
            Scale_longerside(opt.sample_size),
            CenterCrop(opt.sample_size),
            RandomHorizontalFlip(),
            ToTensor(opt.norm_value), norm_method
        ])
        

        target_transform = ClassLabel()
        training_data = get_training_set(opt, spatial_transform, target_transform)

        train_loader = torch.utils.data.DataLoader(
            training_data,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_threads,
            pin_memory=True)

        

        if opt.learning_policy == '2stream':
            train_logger = Logger(
                os.path.join(opt.result_path, 'train.log'),
                ['epoch', 'loss', 'loss_cls', 'loss_box', 'OBOA', 'MAE', 'MAEP', 'MAEN', 'lr'])
            train_batch_logger = Logger(
                os.path.join(opt.result_path, 'train_batch.log'),
                ['epoch', 'batch', 'iter', 'loss', 'loss_cls', 'loss_box', 'OBOA', 'MAE', 'MAEP', 'MAEN', 'lr'])
            from train_2stream import train_epoch
        

        if opt.nesterov:
            dampening = 0
        else:
            dampening = opt.dampening


        finetune_parameters = []

        
        
        ignored_params = list(map(id, finetune_parameters)) 
        base_parameters = filter(lambda p: id(p) not in ignored_params,model.parameters())

        if opt.optimizer == 'sgd':
            optimizer = optim.SGD(
                parameters,
                lr=opt.learning_rate,
                momentum=opt.momentum,
                dampening=dampening,
                weight_decay=opt.weight_decay,
                nesterov=opt.nesterov)

        elif opt.optimizer == 'adam':
            if opt.train_from_scratch == True:
                optimizer = optim.Adam([
                    {'params': base_parameters},
                    {'params': finetune_parameters, 'lr': opt.learning_rate*2}], 
                    lr=opt.learning_rate,
                    weight_decay=opt.weight_decay)
            else:
                optimizer = optim.Adam(
                    finetune_parameters,
                    lr=opt.learning_rate*2,
                    weight_decay=opt.weight_decay)
            
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[5,15], gamma=0.1)
    

    if not opt.no_val:
        spatial_transform = Compose([
            Scale_longerside(opt.sample_size), 
            CenterCrop(opt.sample_size),
            ToTensor(opt.norm_value), norm_method
        ])
        target_transform = ClassLabel()

        val_loader = {}
        for j in range(0, len(opt.val_dataset)):
            validation_data = get_validation_set(opt.val_dataset[j], spatial_transform, target_transform, opt)

            val_loader[j] = torch.utils.data.DataLoader(
                validation_data,
                batch_size=opt.val_batch_size,
                shuffle=False,
                num_workers=opt.n_threads,
                pin_memory=True)
        

        if opt.validate_policy == '2stream':
            val_logger = {}
            for j in range(0, len(val_loader)):
                val_logger[j] = Logger(
                    os.path.join(opt.result_path, 'val_'+opt.val_dataset[j]+'.log'), 
                    ['epoch', 'OBOA', 'MAE', 'MAE_std', 'MAEP', 'MAEN'])
            from val_2stream import val_epoch
        

    if opt.pretrain_path:
        print('loading pretrained checkpoint {}'.format(opt.pretrain_path))
        pretrain = torch.load(opt.pretrain_path)
        pretrain = pretrain['state_dict']
        new_state_dict = OrderedDict()

        for k, v in pretrain.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict, strict=False)

    
    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)
        # assert opt.arch == checkpoint['arch']

        opt.begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        if not opt.no_train:
            optimizer.load_state_dict(checkpoint['optimizer'])

        del checkpoint
        torch.cuda.empty_cache()
    

    print('run')
    for i in range(opt.begin_epoch, opt.n_epochs + 1):

        if not opt.no_train:
            if opt.learning_policy == '2stream':
                train_epoch(i, train_loader, model, optimizer, opt, train_logger, train_batch_logger)

        if not opt.no_val:
            for j in range(0, len(val_loader)):
                validation_loss = val_epoch(i, val_loader[j], model, opt, val_logger[j], opt.val_dataset[j])
            if opt.no_train:
                break
    
    