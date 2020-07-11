import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time
import os
import sys
import numpy as np
import random
import math

from utils.utils import AverageMeter
from utils.myutils import update_inputs_2stream

val_opt = {}
val_opt['iter_terminal_num'] = 1e7

val_opt['merge_level'] = 5  
val_opt['merge_w'] = 0.5 

val_opt['min_scale'] = 0.03 
val_opt['max_scale'] = 0.35 
val_opt['init_scale_num'] = 30 
val_opt['abandon_second_box'] = False


def action_step(state, action_1, action_2, step, sample_len, opt, dataset):
    lp, mp, rp = state

    seg_len_1 = (mp - lp + 1) * action_1
    seg_len_2 = (rp - mp) * action_2

    seg_len_1 = min(max(4, seg_len_1), sample_len/val_opt['min_cycles'])
    seg_len_2 = min(max(4, seg_len_2), sample_len/val_opt['min_cycles'])

    mp = int(mp + step)
    lp = int(mp - seg_len_1 + 1) 
    rp = int(mp + seg_len_2)

    state = (lp, mp, rp)

    done_flag =  mp >= sample_len
    fail_flag =  (mp - lp + 1) < 4 or (rp - mp) < 4

    return state, done_flag, fail_flag


def val_epoch(epoch, data_loader, model, opt, epoch_logger, val_dataset):    
        
    print('eval at epoch {}'.format(epoch))

    if val_dataset=='ucf_aug':
        val_opt['min_cycles']=2
    else:
        val_opt['min_cycles']=4

    if val_dataset=='yt_seg':
        val_opt['merge_w']=0.1

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    maes = AverageMeter()
    maeps = AverageMeter()
    maens = AverageMeter()
    oboas = AverageMeter() 

    end_time = time.time()
    counts_oboa = []
    counts_all = []
    maes_all = []
    oboas_all = []

    cycle_length_dataset = np.zeros([150, pow(2, val_opt['merge_level'])], dtype=np.float)
    cycle_length_dataset_ptr = 0


    for i, (sample_inputs, _, _, label_counts, sample_len) in enumerate(data_loader):

        if val_opt['iter_terminal_num'] != -1 and i > val_opt['iter_terminal_num']:
            break
      


        data_time.update(time.time() - end_time)
        end_time = time.time()

        batch_size = sample_inputs.size(0)

        # targets init
        label_counts = label_counts.numpy()
        sample_len = sample_len.numpy()
        level_pow = pow(2, val_opt['merge_level'])
        
        # track state init
        mp = np.zeros([batch_size, val_opt['merge_level'], level_pow], dtype=np.int)
        lp_l = np.zeros([batch_size, val_opt['merge_level'], level_pow], dtype=np.int)
        lp_r = np.zeros([batch_size, val_opt['merge_level'], level_pow], dtype=np.int)
        rp_l = np.zeros([batch_size, val_opt['merge_level'], level_pow], dtype=np.int)
        rp_r = np.zeros([batch_size, val_opt['merge_level'], level_pow], dtype=np.int)

        load_lp = np.zeros(batch_size, dtype=np.int)
        load_mp = np.zeros(batch_size, dtype=np.int)
        load_rp = np.zeros(batch_size, dtype=np.int)
        save_lp = np.zeros(batch_size, dtype=np.int)
        save_mp = np.zeros(batch_size, dtype=np.int)
        save_rp = np.zeros(batch_size, dtype=np.int)

        load_ls = np.zeros(batch_size, dtype=np.float)
        load_rs = np.zeros(batch_size, dtype=np.float)
        save_ls = np.zeros(batch_size, dtype=np.float)
        save_rs = np.zeros(batch_size, dtype=np.float)
        
        counts = np.zeros(batch_size, dtype=np.float)
        

        # get the first estimation
        max_mp = np.zeros(batch_size, dtype=np.int)
        max_score = np.zeros(batch_size, dtype=np.float)
        for j in range(0, batch_size):
            max_score[j] = -1e6

        for k in range(0, val_opt['init_scale_num']):
            powers_level = (val_opt['max_scale'] / val_opt['min_scale']) ** (float(k)/(val_opt['init_scale_num']-1))
            inputs = torch.zeros([batch_size, 3, opt.basic_duration, opt.sample_size, opt.sample_size], dtype=torch.float).cuda()

            for j in range(0, batch_size):
                mp_k = sample_len[j] * val_opt['min_scale'] * powers_level
                mid_pt = sample_len[j]/2
                inputs[j], _ = update_inputs_2stream(sample_inputs[j], [mid_pt-mp_k, mid_pt, mid_pt+mp_k+1], sample_len[j], opt)
            
            pred_cls, pred_box, _, _ = model(inputs)
            pred_box = torch.clamp(pred_box, min=-0.5, max=0.5)
            
            for j in range(0, batch_size):

                for p in range(3, 4):
                    box_exp = math.exp(pred_box[j][p])
                    pred_seg = box_exp * opt.anchors[p]
                    penalty = 1 
                    score = F.softmax(pred_cls, dim=1)[j][1][p] * penalty
                    mp_k = sample_len[j] * val_opt['min_scale'] * powers_level * pred_seg
                    if score > max_score[j] and mp_k >= 4 and mp_k < sample_len[j]/val_opt['min_cycles']:
                        max_score[j], max_mp[j] = score, mp_k

        for k in range(0, 4):
            inputs = torch.zeros([batch_size, 3, opt.basic_duration, opt.sample_size, opt.sample_size], dtype=torch.float).cuda()
            for j in range(0, batch_size):
                mp_k = max_mp[j]
                mid_pt = sample_len[j]/2
                inputs[j], _ = update_inputs_2stream(sample_inputs[j], [mid_pt-mp_k, mid_pt, mid_pt+mp_k+1], sample_len[j], opt)

            pred_cls, pred_box, _, _ = model(inputs)
            pred_box = torch.clamp(pred_box, min=-0.5, max=0.5)
                
            for j in range(0, batch_size):
                max_score[j] = -1e6
                tmp = max_mp[j]
                for p in range(3, 4):
                    box_exp = math.exp(pred_box[j][p])
                    pred_seg = box_exp * opt.anchors[p]
                    penalty = 1
                    score = F.softmax(pred_cls, dim=1)[j][1][p] * penalty
                    mp_k = tmp * pred_seg
                    if score > max_score[j] and mp_k  >= 4 and mp_k < sample_len[j]/val_opt['min_cycles']:
                        max_score[j], max_mp[j] = score, round(float(max_mp[j]*(1-val_opt['merge_w']))+float(mp_k*val_opt['merge_w']))


        for j in range(0, batch_size):
            for l2 in range(0, level_pow):
                mp[j,0,l2] = int(float(sample_len[j]) / float(level_pow+1) * (l2+0.5))  
                lp_l[j,0,l2] = mp[j,0,l2] - max_mp[j]
                rp_l[j,0,l2] = mp[j,0,l2] + max_mp[j] + 1
                lp_r[j,0,l2] = lp_l[j,0,l2]
                rp_r[j,0,l2] = rp_l[j,0,l2]

        

        total_steps = 0
        for l1 in range(1, val_opt['merge_level']):

            steps = pow(2, val_opt['merge_level']-l1-1)
            pos = -steps
            for l2 in range(0, pow(2,l1)):
                pos = pos + 2*steps

                if l1==1:
                    iters = 4
                elif l1==2:
                    iters = 2
                else:
                    iters = 1

                for l3 in range(0, iters):
                    
                    total_steps = total_steps + 1

                    inputs = torch.zeros([batch_size, 3, opt.basic_duration, opt.sample_size, opt.sample_size], dtype=torch.float).cuda()
                    # network input initilization
                    for j in range(0, batch_size):

                        if l3 == 0:
                            load_mp[j] = mp[j,l1-1,pos]
                            load_lp[j] = round(float(lp_l[j,l1-1,pos]+lp_r[j,l1-1,pos])/2)
                            load_rp[j] = round(float(rp_l[j,l1-1,pos]+rp_r[j,l1-1,pos])/2)
                        else:
                            load_mp[j] = save_mp[j]
                            load_lp[j] = round(float(save_lp[j]) * val_opt['merge_w'] + float(load_lp[j]) * (1.0-val_opt['merge_w']))
                            load_rp[j] = round(float(save_rp[j]) * val_opt['merge_w'] + float(load_rp[j]) * (1.0-val_opt['merge_w']))

                        inputs[j], _ = update_inputs_2stream(sample_inputs[j], [load_lp[j], load_mp[j], load_rp[j]], sample_len[j], opt)

                    # do the forward
                    inputs = Variable(inputs)
                    pred_cls_1, pred_box_1, pred_cls_2, pred_box_2 = model(inputs)

                    
                    pred_box_1 = torch.clamp(pred_box_1, min=-0.5, max=0.5)
                    pred_box_2 = torch.clamp(pred_box_2, min=-0.5, max=0.5)

                    # track state update
                    for j in range(0, batch_size):
                        
                        max_score, action_1 = -1e6, -1
                        for k in range(0, opt.n_classes):
                            box_exp = math.exp(pred_box_1[j][k])
                            pred_seg = box_exp * opt.anchors[k]
                            penalty = 1 
                            score = F.softmax(pred_cls_1, dim=1)[j][1][k] * penalty
                            if score > max_score:
                                max_score, action_1 = score, pred_seg
                                save_ls[j] = score

                        max_score, action_2 = -1e6, -1
                        for k in range(0, opt.n_classes):
                            box_exp = math.exp(pred_box_2[j][k])
                            pred_seg = box_exp * opt.anchors[k]
                            penalty = 1 
                            score = F.softmax(pred_cls_2, dim=1)[j][1][k] * penalty
                            if score > max_score:
                                max_score, action_2 = score, pred_seg
                                save_rs[j] = score

                        if val_opt['abandon_second_box'] == True:
                            action_2 = action_1
                            save_rs[j] = save_ls[j]


                        new_state, done_flag, fail_flag = action_step([load_lp[j], load_mp[j], load_rp[j]], action_1, action_2, 0, sample_len[j], opt, val_dataset)
                        save_lp[j], save_mp[j], save_rp[j] = new_state

                        
                        if fail_flag:
                            save_lp[j] = load_lp[j]
                            save_rp[j] = load_rp[j]

                for j in range(0, batch_size):
  

                    l_segments = float(save_lp[j]) * val_opt['merge_w'] + float(load_lp[j]) * (1.0-val_opt['merge_w'])
                    r_segments = float(save_rp[j]) * val_opt['merge_w'] + float(load_rp[j]) * (1.0-val_opt['merge_w'])

                    for s in range(-steps, 0):
                        mp[j,l1,pos+s] = mp[j,l1-1,pos+s]               
                        lp_r[j,l1,pos+s] = mp[j,l1-1,pos+s] + (l_segments-mp[j,l1-1,pos])
                        rp_r[j,l1,pos+s] = mp[j,l1-1,pos+s] + (r_segments-mp[j,l1-1,pos])


                        if l1 <= 2 or l1 == val_opt['merge_level']-1 or l2 == 0:
                            lp_l[j,l1,pos+s] = lp_r[j,l1,pos+s]
                            rp_l[j,l1,pos+s] = rp_r[j,l1,pos+s]
                        else: 
                            lp_l[j,l1,pos+s] = lp_l[j,l1-1,pos+s]
                            rp_l[j,l1,pos+s] = rp_l[j,l1-1,pos+s]


                    for s in range(0, steps):
                        mp[j,l1,pos+s] = mp[j,l1-1,pos+s]  
                        lp_l[j,l1,pos+s] = mp[j,l1-1,pos+s] + (l_segments-mp[j,l1-1,pos])
                        rp_l[j,l1,pos+s] = mp[j,l1-1,pos+s] + (r_segments-mp[j,l1-1,pos])
                        if l1 <= 2 or l1 == val_opt['merge_level']-1 or l2 == pow(2,l1)-1:
                            lp_r[j,l1,pos+s] = lp_l[j,l1,pos+s]
                            rp_r[j,l1,pos+s] = rp_l[j,l1,pos+s]
                        else: 
                            lp_r[j,l1,pos+s] = lp_r[j,l1-1,pos+s]
                            rp_r[j,l1,pos+s] = rp_r[j,l1-1,pos+s]
                

        for j in range(0, batch_size):
            left_avg = AverageMeter()
            right_avg = AverageMeter()
            for k in range(0, level_pow):
                last = val_opt['merge_level'] -1

                lp_avg = round(float(lp_l[j,last,k]+lp_r[j,last,k])/2)
                rp_avg = round(float(rp_l[j,last,k]+rp_r[j,last,k])/2)

                pos1 = int(lp_avg - (mp[j,last,k] - lp_avg + 1) * opt.l_context_ratio)
                pos2 = int(rp_avg + (rp_avg - mp[j,last,k] + 0) * (opt.r_context_ratio - 1))
                if pos1 >= 0 and pos2 < sample_len[j]:

                    if val_dataset == 'quva' or val_dataset == 'yt_seg' or val_dataset == 'ucf_aug':
                        left_avg.update(1.0/float(mp[j,last,k]-lp_avg+1))
                        right_avg.update(1.0/float(rp_avg - mp[j,last,k]))
                        
                    else:
                        left_avg.update(float(mp[j,last,k] - lp_avg+1))
                        right_avg.update(float(rp_avg - mp[j,last,k]))

                cycle_length_dataset[cycle_length_dataset_ptr+j, k] = 1.0/float(mp[j,last,k]-lp_avg+1)+1.0/float(rp_avg - mp[j,last,k])
                        

            if left_avg.avg == 0 or right_avg.avg == 0:
                counts[j] = float(sample_len[j]) / float(max_mp[j]+1)
            else:
                if val_dataset == 'quva' or val_dataset == 'yt_seg' or val_dataset == 'ucf_aug':
                    counts[j] = float(sample_len[j]) * float(left_avg.sum*0.5+right_avg.sum*0.5) /float(left_avg.count)
                    
                else:
                    counts[j] = float(sample_len[j]+1e-6) / float(left_avg.avg*0.5+right_avg.avg*0.5)
                
            counts[j] = float(round(counts[j]))
            # print(sample_inputs.size(), sample_len[j], label_counts[j], counts[j], float(sample_len[j]) / float(max_mp[j]+1))
            
            counts_all.append(counts[j])

            mae = float(abs(counts[j] - label_counts[j]))/ float(label_counts[j])
            if mae > 0.33:
                counts_oboa.append(i)

            if abs(counts[j] - label_counts[j]) > 1:
                oboa = 0.0
            else:
                oboa = 1.0

            maes_all.append(mae)
            oboas_all.append(oboa)
            
            maes.update(mae)
            if counts[j] > label_counts[j]:
                maeps.update(mae)
            elif counts[j] < label_counts[j]:
                maens.update(mae)
            oboas.update(oboa)



        batch_time.update(time.time() - end_time)
        cycle_length_dataset_ptr = cycle_length_dataset_ptr + batch_size
        

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'OBOA {oboa.val:.4f} ({oboa.avg:.4f})\t'
              'MAE {maes.val:.4f} ({maes.avg:.4f})\t'
              'MAEstd {maestd:.4f}\t'
              'MAEP {maeps.val:.4f} ({maeps.avg:.4f})\t'
              'MAEN {maens.val:.4f} ({maens.avg:.4f})\t'
              'total_steps {total_steps: d}\n'.format(
                  epoch,
                  i + 1,
                  len(data_loader),
                  batch_time=batch_time,
                  oboa=oboas,
                  maes=maes,
                  maestd=maes.std(),
                  maeps=maeps,
                  maens=maens,
                  total_steps=total_steps))


    # np.save(val_dataset, cycle_length_dataset)


    epoch_logger.log({
        'epoch': epoch,
        'OBOA': oboas.avg,
        'MAE': maes.avg,
        'MAE_std': maes.std(),
        'MAEP': maeps.avg,
        'MAEN': maens.avg,
    })
    return maes.avg

            
