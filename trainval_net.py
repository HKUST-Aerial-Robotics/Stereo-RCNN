# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick

# Modified by Peiliang Li for Stereo RCNN train
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import time

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
      adjust_learning_rate, save_checkpoint, clip_gradient

from model.stereo_rcnn.resnet import resnet

def parse_args():
  '''
  Parse input arguments
  '''
  parser = argparse.ArgumentParser(description='Train the Stereo R-CNN network')

  parser.add_argument('--start_epoch', dest='start_epoch',
                      help='starting epoch',
                      default=1, type=int)
  parser.add_argument('--epochs', dest='max_epochs',
                      help='number of epochs to train',
                      default=12, type=int)

  parser.add_argument('--save_dir', dest='save_dir',
                      help='directory to save models', default="models_stereo",
                      type=str)
  parser.add_argument('--nw', dest='num_workers',
                      help='number of worker to load data',
                      default=8, type=int)
  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=1, type=int)

  # config optimization
  parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                      help='step to do learning rate decay, unit is epoch',
                      default=10, type=int)
  parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                      help='learning rate decay ratio',
                      default=0.1, type=float)

  # resume trained model
  parser.add_argument('--r', dest='resume',
                      help='resume checkpoint or not',
                      default=False, type=bool)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load model',
                      default=1, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load model',
                      default=6477, type=int)

  args = parser.parse_args()
  return args


class sampler(Sampler):
  def __init__(self, train_size, batch_size):
    self.num_data = train_size
    self.num_per_batch = int(train_size / batch_size)
    self.batch_size = batch_size
    self.range = torch.arange(0,batch_size).view(1, batch_size).long()
    self.leftover_flag = False
    if train_size % batch_size:
      self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
      self.leftover_flag = True

  def __iter__(self):
    rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
    self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

    self.rand_num_view = self.rand_num.view(-1)

    if self.leftover_flag:
      self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

    return iter(self.rand_num_view)

  def __len__(self):
    return self.num_data

if __name__ == '__main__':

  args = parse_args()

  print('Using config:')
  np.random.seed(cfg.RNG_SEED)

  imdb, roidb, ratio_list, ratio_index = combined_roidb('kitti_train')
  train_size = len(roidb)

  print('{:d} roidb entries'.format(len(roidb)))

  output_dir = args.save_dir + '/'
  if not os.path.exists(output_dir):
    print('save dir', output_dir)
    os.makedirs(output_dir)
  log_info = open((output_dir + 'trainlog.txt'), 'w')

  def log_string(out_str):  
    log_info.write(out_str+'\n')
    log_info.flush()
    print(out_str)

  sampler_batch = sampler(train_size, args.batch_size)

  dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                           imdb.num_classes, training=True)

  dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                            sampler=sampler_batch, num_workers=args.num_workers)

  # initilize the tensor holder here.
  im_left_data = Variable(torch.FloatTensor(1).cuda())
  im_right_data = Variable(torch.FloatTensor(1).cuda())
  im_info = Variable(torch.FloatTensor(1).cuda())
  num_boxes = Variable(torch.LongTensor(1).cuda())
  gt_boxes_left = Variable(torch.FloatTensor(1).cuda())
  gt_boxes_right = Variable(torch.FloatTensor(1).cuda())
  gt_boxes_merge = Variable(torch.FloatTensor(1).cuda())
  gt_dim_orien = Variable(torch.FloatTensor(1).cuda())
  gt_kpts = Variable(torch.FloatTensor(1).cuda())

  # initilize the network here.
  stereoRCNN = resnet(imdb.classes, 101, pretrained=True)

  stereoRCNN.create_architecture()

  lr = cfg.TRAIN.LEARNING_RATE

  uncert = Variable(torch.rand(6).cuda(), requires_grad=True)
  torch.nn.init.constant(uncert, -1.0)

  params = []
  for key, value in dict(stereoRCNN.named_parameters()).items():
    if value.requires_grad:
      if 'bias' in key:
        params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
      else:
        params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
  params += [{'params':[uncert], 'lr':lr}]

  optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

  if args.resume:
    load_name = os.path.join(output_dir,
      'stereo_rcnn_{}_{}.pth'.format(args.checkepoch, args.checkpoint))
    log_string('loading checkpoint %s' % (load_name))
    checkpoint = torch.load(load_name)
    args.start_epoch = checkpoint['epoch']
    stereoRCNN.load_state_dict(checkpoint['model'])
    lr = optimizer.param_groups[0]['lr']
    uncert.data = checkpoint['uncert']
    log_string('loaded checkpoint %s' % (load_name))

  stereoRCNN.cuda()

  iters_per_epoch = int(train_size / args.batch_size)
  for epoch in range(args.start_epoch, args.max_epochs + 1):
    
    stereoRCNN.train()
    start = time.time()

    if epoch % (args.lr_decay_step + 1) == 0:
        adjust_learning_rate(optimizer, args.lr_decay_gamma)
        lr *= args.lr_decay_gamma

    data_iter = iter(dataloader)
    for step in range(iters_per_epoch):
      data = next(data_iter)
      im_left_data.data.resize_(data[0].size()).copy_(data[0])
      im_right_data.data.resize_(data[1].size()).copy_(data[1])
      im_info.data.resize_(data[2].size()).copy_(data[2])
      gt_boxes_left.data.resize_(data[3].size()).copy_(data[3])
      gt_boxes_right.data.resize_(data[4].size()).copy_(data[4])
      gt_boxes_merge.data.resize_(data[5].size()).copy_(data[5])
      gt_dim_orien.data.resize_(data[6].size()).copy_(data[6])
      gt_kpts.data.resize_(data[7].size()).copy_(data[7])
      num_boxes.data.resize_(data[8].size()).copy_(data[8]) 

      start = time.time() 
      stereoRCNN.zero_grad()
      rois_left, rois_right, cls_prob, bbox_pred, dim_orien_pred, kpts_prob, \
      left_border_prob, right_border_prob, rpn_loss_cls, rpn_loss_box_left_right,\
      RCNN_loss_cls, RCNN_loss_bbox, RCNN_loss_dim_orien, RCNN_loss_kpts, rois_label =\
      stereoRCNN(im_left_data, im_right_data, im_info, gt_boxes_left, gt_boxes_right, \
                 gt_boxes_merge, gt_dim_orien, gt_kpts, num_boxes)

      loss = rpn_loss_cls.mean() * torch.exp(-uncert[0]) + uncert[0] +\
              rpn_loss_box_left_right.mean() * torch.exp(-uncert[1]) + uncert[1] +\
              RCNN_loss_cls.mean() * torch.exp(-uncert[2]) + uncert[2]+\
              RCNN_loss_bbox.mean() * torch.exp(-uncert[3]) + uncert[3] +\
              RCNN_loss_dim_orien.mean() * torch.exp(-uncert[4]) + uncert[4] +\
              RCNN_loss_kpts.mean() * torch.exp(-uncert[5]) + uncert[5]
      uncert_data = uncert.data
      log_string('uncert: %.4f, %.4f, %.4f, %.4f, %.4f, %.4f' \
                %(uncert_data[0], uncert_data[1], uncert_data[2], uncert_data[3], uncert_data[4], uncert_data[5])) 

      optimizer.zero_grad()
      loss.backward()
      clip_gradient(stereoRCNN, 10.)
      optimizer.step()

      end = time.time()

      loss_rpn_cls = rpn_loss_cls.item()
      loss_rpn_box_left_right = rpn_loss_box_left_right.item()
      loss_rcnn_cls = RCNN_loss_cls.item()
      loss_rcnn_box = RCNN_loss_bbox.item()
      loss_rcnn_dim_orien = RCNN_loss_dim_orien.item()
      loss_rcnn_kpts = RCNN_loss_kpts
      fg_cnt = torch.sum(rois_label.data.ne(0))
      bg_cnt = rois_label.data.numel() - fg_cnt

      log_string('[epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e'\
            %(epoch, step, iters_per_epoch, loss.item(), lr))
      log_string('\t\t\tfg/bg=(%d/%d), time cost: %f' %(fg_cnt, bg_cnt, end-start))
      log_string('\t\t\trpn_cls: %.4f, rpn_box_left_right: %.4f, rcnn_cls: %.4f, rcnn_box_left_right %.4f,dim_orien %.4f, kpts %.4f' \
            %(loss_rpn_cls, loss_rpn_box_left_right, loss_rcnn_cls, loss_rcnn_box, loss_rcnn_dim_orien, loss_rcnn_kpts))

      del loss, rpn_loss_cls, rpn_loss_box_left_right, RCNN_loss_cls, RCNN_loss_bbox, RCNN_loss_dim_orien, RCNN_loss_kpts

    save_name = os.path.join(output_dir, 'stereo_rcnn_{}_{}.pth'.format(epoch, step))
    save_checkpoint({
      'epoch': epoch + 1,
      'model': stereoRCNN.state_dict(),
      'optimizer': optimizer.state_dict(),
      'uncert':uncert.data,
    }, save_name)

    log_string('save model: {}'.format(save_name)) 
    end = time.time()
    log_string('time %.4f' %(end - start))





    
