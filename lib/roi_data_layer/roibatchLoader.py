
"""The data layer used during training to train a Fast R-CNN network.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
from PIL import Image
import torch
import cv2
import math
from model.utils.config import cfg
from roi_data_layer.minibatch import get_minibatch, get_minibatch
from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes

import numpy as np
import random
import time
import pdb

class roibatchLoader(data.Dataset):
  def __init__(self, roidb, ratio_list, ratio_index, batch_size, num_classes, training=True, normalize=None):
    self._roidb = roidb
    self._num_classes = num_classes
    # we make the height of image consistent to trim_height, trim_width
    self.trim_height = cfg.TRAIN.TRIM_HEIGHT
    self.trim_width = cfg.TRAIN.TRIM_WIDTH
    self.max_num_box = cfg.MAX_NUM_GT_BOXES
    self.training = training
    self.normalize = normalize
    self.ratio_list = ratio_list
    self.ratio_index = ratio_index
    self.batch_size = batch_size
    self.data_size = len(self.ratio_list)

    # given the ratio_list, we want to make the ratio same for each batch.
    self.ratio_list_batch = torch.Tensor(self.data_size).zero_()
    num_batch = int(np.ceil(len(ratio_index) / batch_size))
    for i in range(num_batch):
        left_idx = i*batch_size
        right_idx = min((i+1)*batch_size-1, self.data_size-1)

        if ratio_list[right_idx] < 1:
            # for ratio < 1, we preserve the leftmost in each batch.
            target_ratio = ratio_list[left_idx]
        elif ratio_list[left_idx] > 1:
            # for ratio > 1, we preserve the rightmost in each batch.
            target_ratio = ratio_list[right_idx]
        else:
            # for ratio cross 1, we make it to be 1.
            target_ratio = 1

        self.ratio_list_batch[left_idx:(right_idx+1)] = target_ratio


  def __getitem__(self, index):
    if self.training:
        index_ratio = int(self.ratio_index[index])
    else:
        index_ratio = index

    # get the anchor index for current sample index
    # here we set the anchor index to the last one
    # sample in this group
    minibatch_db = [self._roidb[index_ratio]]
    blobs = get_minibatch(minibatch_db, self._num_classes, self.training)
    '''    
    img_left = blobs['data_left'][0]
    img_right = blobs['data_right'][0]
    for i in range(blobs['gt_boxes_left'].shape[0]):
        #print('debug',blobs['gt_kpts'][i])
        alpha = math.atan2(blobs['gt_dim_orien'][i,3],blobs['gt_dim_orien'][i,4])*180.0/3.14159
        cv2.putText(img_left, '%.1f' % (alpha), (int(blobs['gt_boxes_left'][i,0]), int(blobs['gt_boxes_left'][i,1]-15)), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), thickness=1)
        cv2.rectangle(img_left, (blobs['gt_boxes_left'][i,0], blobs['gt_boxes_left'][i,1]),\
                                (blobs['gt_boxes_left'][i,2], blobs['gt_boxes_left'][i,3]), (0,255,0),1)
        cv2.rectangle(img_right, (blobs['gt_boxes_right'][i,0], blobs['gt_boxes_right'][i,1]),\
                                (blobs['gt_boxes_right'][i,2], blobs['gt_boxes_right'][i,3]), (0,255,0),1)
        cv2.rectangle(img_left, (blobs['gt_boxes_merge'][i,0], blobs['gt_boxes_merge'][i,1]),\
                                (blobs['gt_boxes_merge'][i,2], blobs['gt_boxes_merge'][i,3]), (0,0,255),1)
        if blobs['gt_kpts'][i,0] >=0:
            cv2.circle(img_left, (blobs['gt_kpts'][i,0], blobs['gt_boxes_left'][i,3]), 3, (0,0,255), -1)
        if blobs['gt_kpts'][i,1] >=0:
            cv2.circle(img_left, (blobs['gt_kpts'][i,1], blobs['gt_boxes_left'][i,3]), 3, (0,255,0), -1)
        if blobs['gt_kpts'][i,2] >=0:
            cv2.circle(img_left, (blobs['gt_kpts'][i,2], blobs['gt_boxes_left'][i,3]), 3, (255,0,0), -1)
        if blobs['gt_kpts'][i,3] >=0:
            cv2.circle(img_left, (blobs['gt_kpts'][i,3], blobs['gt_boxes_left'][i,3]), 3, (0,255,255), -1)
        if blobs['gt_kpts'][i,4] >=0:
            cv2.circle(img_left, (blobs['gt_kpts'][i,4], blobs['gt_boxes_left'][i,3]), 3, (255,255,0), -1)
        if blobs['gt_kpts'][i,5] >=0:
            cv2.circle(img_left, (blobs['gt_kpts'][i,5], blobs['gt_boxes_left'][i,3]), 3, (255,0,255), -1)
    img_left = np.concatenate((img_left, img_right), axis=0)
    cv2.imwrite('debug_'+str(index_ratio) + '.png', img_left)
    '''
    data_left = torch.from_numpy(blobs['data_left'])
    data_right = torch.from_numpy(blobs['data_right'])
    im_info = torch.from_numpy(blobs['im_info'])

    data_height, data_width = data_left.size(1), data_left.size(2) 
    if self.training:
        boxes_all = np.concatenate((blobs['gt_boxes_left'], blobs['gt_boxes_right'],blobs['gt_boxes_merge'],\
                                    blobs['gt_dis'], blobs['gt_dim_orien'], blobs['gt_kpts']), axis=1)
        np.random.shuffle(boxes_all) 
        
        gt_boxes_left = torch.from_numpy(boxes_all[:,0:5])
        gt_boxes_right = torch.from_numpy(boxes_all[:,5:10])
        gt_boxes_merge = torch.from_numpy(boxes_all[:,10:15])
        gt_dis = torch.from_numpy(boxes_all[:,15:17])
        gt_dim_orien = torch.from_numpy(boxes_all[:,17:22])
        gt_kpts = torch.from_numpy(boxes_all[:,22:28])

        num_boxes = min(gt_boxes_left.size(0), self.max_num_box) 

        data_left = data_left[0].permute(2, 0, 1).contiguous()
        im_info = im_info.view(3) 
        gt_boxes_left_padding = torch.FloatTensor(self.max_num_box, gt_boxes_left.size(1)).zero_()
        gt_boxes_left_padding[:num_boxes,:] = gt_boxes_left[:num_boxes]
        gt_boxes_left_padding = gt_boxes_left_padding.contiguous()

        gt_boxes_right_padding = torch.FloatTensor(self.max_num_box, gt_boxes_right.size(1)).zero_()
        gt_boxes_right_padding[:num_boxes,:] = gt_boxes_right[:num_boxes]
        gt_boxes_right_padding = gt_boxes_right_padding.contiguous() 

        gt_boxes_merge_padding = torch.FloatTensor(self.max_num_box, gt_boxes_merge.size(1)).zero_()
        gt_boxes_merge_padding[:num_boxes,:] = gt_boxes_merge[:num_boxes]
        gt_boxes_merge_padding = gt_boxes_merge_padding.contiguous()

        gt_dis_padding = torch.FloatTensor(self.max_num_box, gt_dis.size(1)).zero_()
        gt_dis_padding[:num_boxes,:] = gt_dis[:num_boxes]
        gt_dis_padding = gt_dis_padding.contiguous()

        gt_dim_orien_padding = torch.FloatTensor(self.max_num_box, gt_dim_orien.size(1)).zero_()
        gt_dim_orien_padding[:num_boxes,:] = gt_dim_orien[:num_boxes]
        gt_dim_orien_padding = gt_dim_orien_padding.contiguous()

        gt_kpts_padding = torch.FloatTensor(self.max_num_box, gt_kpts.size(1)).zero_()
        gt_kpts_padding[:num_boxes,:] = gt_kpts[:num_boxes]
        gt_kpts_padding = gt_kpts_padding.contiguous()
        #print(data_left.size(), data_right.size(), im_info.size(), gt_boxes_left_padding. size(), gt_boxes_right_padding.
        return data_left, im_info, gt_boxes_left_padding, gt_boxes_right_padding,\
               gt_boxes_merge_padding, gt_dis_padding, gt_dim_orien_padding, gt_kpts_padding, num_boxes 
    else: 
        data_left = data_left[0].permute(2, 0, 1).contiguous()
        data_right = data_right[0].permute(2, 0, 1).contiguous()
        im_info = im_info.view(3) 

        gt_boxes = torch.FloatTensor([1,1,1,1,1])
        num_boxes = 0

        return data_left,data_right,  im_info, gt_boxes, gt_boxes, gt_boxes, gt_boxes, gt_boxes, gt_boxes, num_boxes

  def __len__(self):
    return len(self._roidb)
