# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import math as m
import PIL
import numpy as np
import scipy.sparse
from model.utils.config import cfg
import pdb

ROOT_DIR = osp.join(osp.dirname(__file__), '..', '..')

class imdb(object):
  """Image database."""

  def __init__(self, name, classes=None):
    self._name = name
    self._num_classes = 0
    if not classes:
      self._classes = []
    else:
      self._classes = classes
    self._image_index = []
    self._obj_proposer = 'gt'
    self._roidb = None
    self._roidb_handler = self.default_roidb
    # Use this dict for storing dataset specific config options
    self.config = {}

  @property
  def name(self):
    return self._name

  @property
  def num_classes(self):
    return len(self._classes)

  @property
  def classes(self):
    return self._classes

  @property
  def image_index(self):
    return self._image_index

  @property
  def roidb_handler(self):
    return self._roidb_handler

  @roidb_handler.setter
  def roidb_handler(self, val):
    self._roidb_handler = val

  def set_proposal_method(self, method):
    method = eval('self.' + method + '_roidb')
    self.roidb_handler = method

  @property
  def roidb(self):
    # A roidb is a list of dictionaries, each with the following keys:
    #   boxes
    #   gt_overlaps
    #   gt_classes
    #   flipped
    if self._roidb is not None:
      return self._roidb
    self._roidb = self.roidb_handler()
    return self._roidb

  @property
  def cache_path(self):
    cache_path = osp.abspath(osp.join(cfg.DATA_DIR, 'cache'))
    if not os.path.exists(cache_path):
      os.makedirs(cache_path)
    return cache_path

  @property
  def num_images(self):
    return len(self.image_index)

  def img_left_path_at(self, i):
    raise NotImplementedError
 
  def img_right_path_at(self, i):
    raise NotImplementedError

  def image_id_at(self, i):
    raise NotImplementedError

  def default_roidb(self):
    raise NotImplementedError

  def evaluate_detections(self, all_boxes, output_dir=None):
    """
    all_boxes is a list of length number-of-classes.
    Each list element is a list of length number-of-images.
    Each of those list elements is either an empty list []
    or a numpy array of detection.

    all_boxes[class][image] = [] or np.array of shape #dets x 5
    """
    raise NotImplementedError

  def _get_widths(self):
    return [PIL.Image.open(self.img_left_path_at(i)).size[0]
            for i in range(self.num_images)]

  def append_flipped_images(self):
    num_images = self.num_images
    widths = self._get_widths()
    for i in range(num_images):
      boxes_left = self.roidb[i]['boxes_right'].copy()
      oldx1_left = boxes_left[:, 0].copy()
      oldx2_left = boxes_left[:, 2].copy()
      boxes_left[:, 0] = widths[i] - oldx2_left - 1
      boxes_left[:, 2] = widths[i] - oldx1_left - 1
      assert (boxes_left[:, 2] >= boxes_left[:, 0]).all()

      boxes_right = self.roidb[i]['boxes_left'].copy()
      oldx1_right = boxes_right[:, 0].copy()
      oldx2_right = boxes_right[:, 2].copy()
      boxes_right[:, 0] = widths[i] - oldx2_right - 1
      boxes_right[:, 2] = widths[i] - oldx1_right - 1
      assert (boxes_right[:, 2] >= boxes_right[:, 0]).all()

      boxes_merge = self.roidb[i]['boxes_merge'].copy()
      oldx1_merge = boxes_merge[:, 0].copy()
      oldx2_merge = boxes_merge[:, 2].copy()
      boxes_merge[:, 0] = widths[i] - oldx2_merge - 1
      boxes_merge[:, 2] = widths[i] - oldx1_merge - 1
      assert (boxes_merge[:, 2] >= boxes_merge[:, 0]).all()
      
      dim_orien = self.roidb[i]['dim_orien'].copy()
      out_range1 = dim_orien[:,3] > m.pi
      out_range2 = dim_orien[:,3] < -m.pi
      dim_orien[:,3][out_range1] = dim_orien[:,3][out_range1]-2.0*m.pi
      dim_orien[:,3][out_range2] = dim_orien[:,3][out_range2]+2.0*m.pi
      postive = dim_orien[:,3]>=0
      negtive = dim_orien[:,3]<0
      dim_orien[:,3][postive] = m.pi - dim_orien[:,3][postive]
      dim_orien[:,3][negtive] = -m.pi - dim_orien[:,3][negtive]

      kpts = self.roidb[i]['kpts_right'].copy()
      kpts0_old = kpts[:, 0].copy()
      kpts1_old = kpts[:, 1].copy()
      kpts2_old = kpts[:, 2].copy()
      kpts3_old = kpts[:, 3].copy()
      kpts4_old = kpts[:, 4].copy()
      kpts5_old = kpts[:, 5].copy()

      kpts[:, 0] = widths[i] - kpts3_old - 1
      kpts[:, 1] = widths[i] - kpts2_old - 1
      kpts[:, 2] = widths[i] - kpts1_old - 1
      kpts[:, 3] = widths[i] - kpts0_old - 1
      kpts[:, 4] = widths[i] - kpts5_old - 1
      kpts[:, 5] = widths[i] - kpts4_old - 1

      entry = {'boxes_left': boxes_left,
               'boxes_right': boxes_right,
               'boxes_merge': boxes_merge,
               'dim_orien' : dim_orien,
               'kpts' : kpts,
               'gt_overlaps': self.roidb[i]['gt_overlaps'],
               'gt_classes': self.roidb[i]['gt_classes'],
               'flipped': True}
      self.roidb.append(entry)
    self._image_index = self._image_index * 2

  def create_roidb_from_box_list(self, box_list, gt_roidb):
    assert len(box_list) == self.num_images, \
      'Number of boxes must match number of ground-truth images'
    roidb = []
    for i in range(self.num_images):
      boxes = box_list[i]
      num_boxes = boxes.shape[0]
      overlaps = np.zeros((num_boxes, self.num_classes), dtype=np.float32)

      if gt_roidb is not None and gt_roidb[i]['boxes'].size > 0:
        gt_boxes = gt_roidb[i]['boxes']
        gt_classes = gt_roidb[i]['gt_classes']
        gt_overlaps = bbox_overlaps(boxes.astype(np.float),
                                    gt_boxes.astype(np.float))
        argmaxes = gt_overlaps.argmax(axis=1)
        maxes = gt_overlaps.max(axis=1)
        I = np.where(maxes > 0)[0]
        overlaps[I, gt_classes[argmaxes[I]]] = maxes[I]

      overlaps = scipy.sparse.csr_matrix(overlaps)
      roidb.append({
        'boxes': boxes,
        'gt_classes': np.zeros((num_boxes,), dtype=np.int32),
        'gt_overlaps': overlaps,
        'flipped': False,
        'seg_areas': np.zeros((num_boxes,), dtype=np.float32),
      })
    return roidb

  @staticmethod
  def merge_roidbs(a, b):
    assert len(a) == len(b)
    for i in range(len(a)):
      a[i]['boxes'] = np.vstack((a[i]['boxes'], b[i]['boxes']))
      a[i]['gt_classes'] = np.hstack((a[i]['gt_classes'],
                                      b[i]['gt_classes']))
      a[i]['gt_overlaps'] = scipy.sparse.vstack([a[i]['gt_overlaps'],
                                                 b[i]['gt_overlaps']])
      a[i]['seg_areas'] = np.hstack((a[i]['seg_areas'],
                                     b[i]['seg_areas']))
    return a

