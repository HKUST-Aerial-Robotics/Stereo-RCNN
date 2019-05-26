# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen

# Modified by Peiliang Li for Stereo RCNN
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
import cv2
from model.utils.config import cfg
from model.utils.blob import prep_im_for_blob, im_list_to_blob
import pdb
import math

def get_minibatch(roidb, num_classes):
  """Given a roidb, construct a minibatch sampled from it."""
  num_images = len(roidb)
  # Sample random scales to use for each image in this batch
  random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                  size=num_images)
  assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
    'num_images ({}) must divide BATCH_SIZE ({})'. \
    format(num_images, cfg.TRAIN.BATCH_SIZE)

  # Get the input image blob, formatted for caffe
  im_blob_left, im_blob_right, im_scales = _get_image_blob(roidb, random_scale_inds)

  blobs = {'data_left': im_blob_left}
  blobs['data_right'] = im_blob_right

  assert len(im_scales) == 1, "Single batch only"
  assert len(roidb) == 1, "Single batch only"
  
  # gt boxes: (x1, y1, x2, y2, cls)
  gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]

  im_width = im_blob_left.shape[2]

  gt_boxes_left = np.empty((len(gt_inds), 5), dtype=np.float32)
  gt_boxes_left[:, 0:4] = roidb[0]['boxes_left'][gt_inds, :] * im_scales[0]
  for i in range(gt_boxes_left.shape[0]):
    gt_boxes_left[i, 0] = min(gt_boxes_left[i, 0], im_width)  
  gt_boxes_left[:, 4] = roidb[0]['gt_classes'][gt_inds]

  gt_boxes_right = np.empty((len(gt_inds), 5), dtype=np.float32)
  gt_boxes_right[:, 0:4] = roidb[0]['boxes_right'][gt_inds, :] * im_scales[0]
  for i in range(gt_boxes_right.shape[0]):
    gt_boxes_right[i, 0] = min(gt_boxes_right[i, 0], im_width)  
  gt_boxes_right[:, 4] = roidb[0]['gt_classes'][gt_inds]

  gt_boxes_merge = np.empty((len(gt_inds), 5), dtype=np.float32)
  gt_boxes_merge[:, 0:4] = roidb[0]['boxes_merge'][gt_inds, :] * im_scales[0]
  for i in range(gt_boxes_merge.shape[0]):
    gt_boxes_merge[i, 0] = min(gt_boxes_merge[i, 0], im_width)  
  gt_boxes_merge[:, 4] = roidb[0]['gt_classes'][gt_inds]
  
  gt_dim_orien = np.empty((len(gt_inds), 5), dtype=np.float32)
  gt_dim_orien[:, 0:3] = roidb[0]['dim_orien'][gt_inds][:,0:3]
  for i in range(gt_dim_orien.shape[0]):
    gt_dim_orien[i, 3] = math.sin(roidb[0]['dim_orien'][gt_inds][i,3])
    gt_dim_orien[i, 4] = math.cos(roidb[0]['dim_orien'][gt_inds][i,3])

  gt_kpts = np.empty((len(gt_inds), 6), dtype=np.float32)
  gt_kpts = roidb[0]['kpts'][gt_inds]*im_scales[0]
  for i in range(gt_kpts.shape[0]):
    for j in range(6):
      if gt_kpts[i,j] < 0 or gt_kpts[i,j] > im_width-1:
        gt_kpts[i,j] = -1

  blobs['gt_boxes_left'] = gt_boxes_left
  blobs['gt_boxes_right'] = gt_boxes_right
  blobs['gt_boxes_merge'] = gt_boxes_merge
  blobs['gt_dim_orien'] = gt_dim_orien
  blobs['gt_kpts'] = gt_kpts
  blobs['im_info'] = np.array(
    [[im_blob_left.shape[1], im_blob_left.shape[2], im_scales[0]]],
    dtype=np.float32)

  return blobs

def _get_image_blob(roidb, scale_inds):
  """Builds an input blob from the images in the roidb at the specified
  scales.
  """
  num_images = len(roidb)

  processed_ims_left = []
  processed_ims_right = []
  im_scales = []
  for i in range(num_images):
    img_left = cv2.imread(roidb[i]['img_left'])
    img_right = cv2.imread(roidb[i]['img_right'])

    if roidb[i]['flipped']:
      img_left_flip = img_right[:, ::-1, :].copy()
      img_right = img_left[:, ::-1, :].copy()
      img_left = img_left_flip

    target_size = cfg.TRAIN.SCALES[scale_inds[i]]
    img_left, img_right, im_scale = prep_im_for_blob(img_left, img_right, cfg.PIXEL_MEANS, target_size,
                    cfg.TRAIN.MAX_SIZE)
    im_scales.append(im_scale)
    processed_ims_left.append(img_left)
    processed_ims_right.append(img_right)

  # Create a blob to hold the input images
  blob_left, blob_right = im_list_to_blob(processed_ims_left, processed_ims_right)

  return blob_left, blob_right, im_scales
