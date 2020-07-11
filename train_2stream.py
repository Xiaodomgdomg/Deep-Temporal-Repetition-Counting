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

train_opt = {}
train_opt['early_stop'] = 10
train_opt['iter_terminal_num'] = 1000

val_opt = {}
val_opt['min_scale'] = 0.03 
val_opt['max_scale'] = 0.35
val_opt['init_scale_num'] = 30
val_opt['abandon_second_box'] = False

def update_labels(label, state, sample_len, opt):
    if (state[0] >= sample_len):
        tmp = -1
    else:
        tmp = label[state[0]]
    gt_cls = torch.zeros([opt.n_classes], dtype=torch.float).cuda()
    gt_box = torch.zeros([opt.n_classes], dtype=torch.float).cuda()
    if tmp == -1:
        for i in range(0, opt.n_classes):
            gt_cls[i] = -1
            gt_box[i] = -1
    else:
        tmp = tmp + 1
        for i in range(0, opt.n_classes):
            anchor = opt.anchors[i] * state[1]

            I = min(tmp, anchor)
            U = max(tmp, anchor)
            IOU = float(I) / U

            if IOU >= opt.iou_ubound: 
                gt_cls[i] = 1
                gt_box[i] = math.log(tmp / anchor)
            elif IOU <= opt.iou_lbound:
                gt_cls[i] = 0
                gt_box[i] = -1  
            else:
                gt_cls[i] = -1
                gt_box[i] = -1

    return gt_cls, gt_box

def action_step(state, action_1, action_2, step, sample_len, opt):
    lp, mp, rp = state

    seg_len_1 = (mp - lp + 1) * action_1
    seg_len_2 = (rp - mp) * action_2

    mp = int(mp + step)
    lp = int(mp - seg_len_1 + 1) 
    rp = int(mp + seg_len_2)

    state = (lp, mp, rp)

    done_flag =  mp >= sample_len 
    fail_flag =  (mp - lp + 1) < 4 or (rp - mp) < 4

    return state, done_flag, fail_flag

def state_init(epoch, label_next, label_pre, label_counts, sample_len, opt):
    if label_next[0] == -1:
        lp2, rp2 = 0, sample_len / label_counts - 1 
    else:
        lp2, rp2 = 0, label_next[0] - 1

    lp = lp2 + int(random.random() * 1.0 * (rp2 - lp2 + 1))
    
    magic = random.random()
    if magic < 0.25: 
        seg_ratio = math.pow(2, (random.random()-0.5)*2)
        seg_len = (rp2 - lp2 + 1) * seg_ratio
    elif magic > 0.75:
        seg_ratio = random.randint(-1, 1)
        if seg_ratio == -2:
            seg_len = (rp2 - lp2 + 1) * (0.33+(random.random()-0.5)*0.1)
        else:
            seg_len = (rp2 - lp2 + 1) * (math.pow(2, seg_ratio)+(random.random()-0.5)*0.1)
    else:
        k = random.randint(0, val_opt['init_scale_num'])
        powers_level = (val_opt['max_scale'] / val_opt['min_scale']) ** (float(k)/(val_opt['init_scale_num']-1))
        seg_len = sample_len * val_opt['min_scale'] * powers_level
        
    rp = lp + int(seg_len-1)
    rp = max(rp, lp)
    
    if rp >= sample_len:
        lp, rp = 0, sample_len / label_counts - 1
    
    if rp * 2 + 1 >= sample_len:
        lp, rp = 0, sample_len / 2 - 1

    return lp, rp

def train_epoch(epoch, data_loader, model, optimizer, opt,
                epoch_logger, batch_logger):
        
    print('train at epoch {}'.format(epoch))

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_cls = AverageMeter()
    losses_box = AverageMeter()
    maes = AverageMeter()
    maeps = AverageMeter()
    maens = AverageMeter()
    oboas = AverageMeter() 

    end_time = time.time()

    CrossEntropyLoss = nn.CrossEntropyLoss(ignore_index = -1).cuda()
    SmoothL1Loss = nn.SmoothL1Loss().cuda() 



    for i, (sample_inputs, label_next, label_pre, label_counts, sample_len) in enumerate(data_loader):

        if train_opt['iter_terminal_num'] != -1 and i > train_opt['iter_terminal_num']:
            break

        data_time.update(time.time() - end_time)

        batch_size = sample_inputs.size(0)

        # targets init
        label_next = label_next.numpy()
        label_pre = label_pre.numpy()
        label_counts = label_counts.numpy()
        sample_len = sample_len.numpy()
        total_steps = 0
        
        # track state init
        lp = np.zeros(batch_size, dtype=np.int)
        mp = np.zeros(batch_size, dtype=np.int)
        rp = np.zeros(batch_size, dtype=np.int)
        counts = np.zeros(batch_size, dtype=np.float)
        pre_counts = np.zeros(batch_size, dtype=np.float)
        end_flag = np.zeros(batch_size, dtype=np.int)
        for j in range(0, batch_size):
            while rp[j] == 0 or rp[j] >= sample_len[j]:
                lp[j], mp[j] = state_init(epoch, label_next[j], label_pre[j], label_counts[j], sample_len[j], opt)
                rp[j] = mp[j] + (mp[j] - lp[j] + 1)
            


        while 1:
            inputs = torch.zeros([batch_size, 3, opt.basic_duration, opt.sample_size, opt.sample_size], dtype=torch.float).cuda()
            # network input initilization
            for j in range(0, batch_size):
                inputs[j], _ = update_inputs_2stream(sample_inputs[j], [lp[j], mp[j], rp[j]], sample_len[j], opt)

            # prepare label
            gt_cls_1 = torch.zeros([batch_size, opt.n_classes], dtype=torch.long).cuda()
            gt_box_1 = torch.zeros([batch_size, opt.n_classes], dtype=torch.float).cuda() 
            gt_cls_2 = torch.zeros([batch_size, opt.n_classes], dtype=torch.long).cuda()
            gt_box_2 = torch.zeros([batch_size, opt.n_classes], dtype=torch.float).cuda() 

            for j in range(0, batch_size):
                gt_cls_1[j], gt_box_1[j] = update_labels(label_pre[j], [mp[j], mp[j]-lp[j]+1], sample_len[j], opt) 
                gt_cls_2[j], gt_box_2[j] = update_labels(label_next[j], [mp[j]+1, rp[j]-mp[j]], sample_len[j], opt) 

            # do the forward
            inputs = Variable(inputs)
            pred_cls_1, pred_box_1, pred_cls_2, pred_box_2 = model(inputs)

            for j in range(0, batch_size):
                for k in range(0, opt.n_classes):
                    if gt_box_1[j][k] == -1:
                        gt_box_1[j][k] = pred_box_1[j][k].detach()

                    if gt_box_2[j][k] == -1:
                        gt_box_2[j][k] = pred_box_2[j][k].detach()
                
            # loss calculate
            if val_opt['abandon_second_box'] == True:
                loss_cls = CrossEntropyLoss(pred_cls_1, gt_cls_1) * 1.0
                loss_box = SmoothL1Loss(pred_box_1, gt_box_1) * 50.0
            else:
                loss_cls = CrossEntropyLoss(pred_cls_1, gt_cls_1) * 1.0 + CrossEntropyLoss(pred_cls_2, gt_cls_2) * 1.0
                loss_box = SmoothL1Loss(pred_box_1, gt_box_1) * 50.0 + SmoothL1Loss(pred_box_2, gt_box_2) * 50.0 # 10 is from the faster-rcnn imple
            loss = loss_cls + loss_box

            losses_cls.update(loss_cls.item(), inputs.size(0))
            losses_box.update(loss_box.item(), inputs.size(0))
            losses.update(loss.item(), inputs.size(0))

            # optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred_box_1 = torch.clamp(pred_box_1, min=-2.0, max=2.0)
            pred_box_2 = torch.clamp(pred_box_2, min=-2.0, max=2.0)

            # track state update
            for j in range(0, batch_size):
                magic_step = 5 + random.random() * 15 
                step = int(max(sample_len[j]/magic_step, 1))

                max_score, action_1 = -1e6, -1
                for k in range(0, opt.n_classes):
                    box_exp = math.exp(pred_box_1[j][k])
                    pred_seg = box_exp * opt.anchors[k]
                    penalty = 1  
                    score = F.softmax(pred_cls_1, dim=1)[j][1][k] * penalty
                    if score > max_score:
                        max_score, action_1 = score, pred_seg

                max_score, action_2 = -1e6, -1
                for k in range(0, opt.n_classes):
                    box_exp = math.exp(pred_box_2[j][k])
                    pred_seg = box_exp * opt.anchors[k]
                    penalty = 1 
                    score = F.softmax(pred_cls_2, dim=1)[j][1][k] * penalty
                    if score > max_score:
                        max_score, action_2 = score, pred_seg
                if val_opt['abandon_second_box'] == True:
                    action_2 = action_1

                new_state, done_flag, fail_flag = action_step([lp[j], mp[j], rp[j]], action_1, action_2, step, sample_len[j], opt)
                lp[j], mp[j], rp[j] = new_state
                
                if fail_flag or done_flag:
                    rp[j] = 0
                    while rp[j] == 0 or rp[j] >= sample_len[j]:
                        lp[j], mp[j] = state_init(epoch, label_next[j], label_pre[j], label_counts[j], sample_len[j], opt)
                        rp[j] = mp[j] + (mp[j] - lp[j] + 1)
                    pre_counts[j] = 0
                    counts[j] = pre_counts[j] + float(sample_len[j]-lp[j]+1e-6) / float(mp[j]-lp[j]+1)
                else:
                    pre_counts[j] = pre_counts[j] + step / float(mp[j]-lp[j]+1)
                    counts[j] = pre_counts[j] + float(sample_len[j]-lp[j]+1e-6) / float(mp[j]-lp[j]+1)
                
                

                if done_flag:            
                    end_flag[j] = 1

            # terminal condition
            total_steps += 1
            if sum(end_flag) == batch_size or total_steps > train_opt['early_stop']:
                for j in range(0, batch_size):
                    if counts[j] == 0:
                        counts[j] = float(sample_len[j]) / float(mp[j]-lp[j]+1)

                    mae = float(abs(counts[j] - label_counts[j]))/ float(label_counts[j])
                    if abs(counts[j] - label_counts[j]) > 1:
                        oboa = 0.0
                    else:
                        oboa = 1.0
                    
                    maes.update(mae)
                    if counts[j] > label_counts[j]:
                        maeps.update(mae)
                    elif counts[j] < label_counts[j]:
                        maens.update(mae)
                    oboas.update(oboa)
                break


        batch_time.update(time.time() - end_time)
        end_time = time.time()

        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(data_loader) + (i + 1),
            'loss': losses.val,
            'loss_cls': losses_cls.val,
            'loss_box': losses_box.val,
            'OBOA': oboas.val,
            'MAE': maes.val,
            'MAEP': maeps.val,
            'MAEN': maens.val,
            'lr': optimizer.param_groups[0]['lr']
        })

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Loss_cls {loss_cls.val:.4f} ({loss_cls.avg:.4f})\t'
              'Loss_box {loss_box.val:.4f} ({loss_box.avg:.4f})\t'
              'OBOA {oboa.val:.4f} ({oboa.avg:.4f})\t'
              'MAE {maes.val:.4f} ({maes.avg:.4f})\t'
              'MAEP {maeps.val:.4f} ({maeps.avg:.4f})\t'
              'MAEN {maens.val:.4f} ({maens.avg:.4f})\t'
              'total_steps {total_steps: d}'.format(
                  epoch,
                  i + 1,
                  len(data_loader),
                  batch_time=batch_time,
                  loss=losses,
                  loss_cls=losses_cls,
                  loss_box=losses_box,
                  oboa=oboas,
                  maes=maes,
                  maeps=maeps,
                  maens=maens,
                  total_steps=total_steps))

    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg,
        'loss_cls': losses_cls.avg,
        'loss_box': losses_box.avg,
        'OBOA': oboas.avg,
        'MAE': maes.avg,
        'MAEP': maeps.avg,
        'MAEN': maens.avg,
        'lr': optimizer.param_groups[0]['lr']
    })


    if epoch % opt.checkpoint == 0:
        save_file_path = os.path.join(opt.result_path,
                                      'save_{}.pth'.format(epoch))
        states = {
            'epoch': epoch,
            'opt': opt,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(states, save_file_path)


            
