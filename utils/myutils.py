import csv
import torch
import math
import numpy as np
import random


def update_inputs_2stream(sample_inputs, state, sample_len, opt):
    inputs = torch.zeros([3, opt.basic_duration, opt.sample_size, opt.sample_size], dtype=torch.float).cuda()
    
    padding = torch.zeros([3, opt.sample_size, opt.sample_size], dtype=torch.float).cuda()
    padding.index_fill_(0, torch.tensor([0]).cuda(), opt.mean[0])
    padding.index_fill_(0, torch.tensor([1]).cuda(), opt.mean[1])
    padding.index_fill_(0, torch.tensor([2]).cuda(), opt.mean[2])

    frame_indices = torch.zeros([opt.basic_duration], dtype=torch.float).cuda()

    sduration = int(opt.basic_duration / 2)

    pos = int(state[0] - (state[1] - state[0] + 1) * opt.l_context_ratio)
    pos2 = int(state[1])
    steps = (pos2-pos+1)*1.0/(sduration-1)
    for j in range(0, sduration):
        p = round(pos-0.5+j*steps)
        p = int(max(pos, min(pos2, p)))

        frame_indices[j] = p
        if p < 0 or p >= sample_len:
            inputs[:,j,:,:] = padding
        else:
            inputs[:,j,:,:] = sample_inputs[:,p,:,:]

    
    pos = int(state[1] + 1)
    pos2 = int(state[2] + (state[2] - pos + 1) * (opt.r_context_ratio - 1))
    steps = (pos2-pos+1)*1.0/(sduration-1)
    for j in range(0, sduration):
        p = round(pos-0.5+j*steps)
        p = int(max(pos, min(pos2, p)))

        frame_indices[sduration + j] = p
        if p < 0 or p >= sample_len:
            inputs[:,sduration + j,:,:] = padding
        else:
            inputs[:,sduration + j,:,:] = sample_inputs[:,p,:,:]

        
    return inputs, frame_indices



