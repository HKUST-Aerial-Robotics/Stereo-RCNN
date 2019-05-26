from __future__ import print_function
from __future__ import absolute_import
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick

# Modified by Peiliang Li for Stereo RCNN
# --------------------------------------------------------

import xml.dom.minidom as minidom

import os
# import PIL
import numpy as np
import math as m
import scipy.sparse
import subprocess
import math
import cv2
import glob
import uuid
import scipy.io as sio
import xml.etree.ElementTree as ET
import pickle
from .imdb import imdb
from .imdb import ROOT_DIR
from model.utils import kitti_utils
from model.utils.config import cfg

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

class kitti(imdb):
    def __init__(self, image_set, kitti_path=None):
        imdb.__init__(self, 'kitti_' + image_set)

        self._image_set = image_set
        assert kitti_path is not None
        self._kitti_path = kitti_path

        self._data_path = os.path.join(self._kitti_path)
        self._classes = ('__background__', 'Car')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.png'
        self._image_index = self._load_image_set_index_new()
        # Default to roidb handler
        self._roidb_handler = self.gt_roidb
        if image_set == 'train' or image_set == 'val':
            prefix = 'validation'
        else:
            prefix = 'test'

        assert os.path.exists(self._kitti_path), \
            'kitti path does not exist: {}'.format(self._kitti_path)
        assert os.path.exists(self._data_path), \
            'Path does not exist: {}'.format(self._data_path)

    def img_left_path_at(self, i):
        '''
        Return the absolute path to image i in the image sequence.
        '''
        return self.img_left_path_from_index(self._image_index[i])

    def img_right_path_at(self, i):
        '''
        Return the absolute path to image i in the image sequence.
        '''
        return self.img_right_path_from_index(self._image_index[i])

    def img_left_path_from_index(self, index):
        '''
        Construct an image path from the image's "index" identifier.
        '''
        if self._image_set == 'test':
            prefix = 'testing/image_2'
        else:
            prefix = 'training/image_2'

        image_path = os.path.join(self._data_path, prefix,\
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    def img_right_path_from_index(self, index):
        '''
        Construct an image path from the image's "index" identifier.
        '''
        if self._image_set == 'test':
            prefix = 'testing/image_3'
        else:
            prefix = 'training/image_3'

        image_path = os.path.join(self._data_path, prefix,\
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index_new(self):
        '''
        Load the indexes listed in this dataset's image set file.
        '''
        if self.name == 'kitti_train':
            train_set_file = open('data/kitti/splits/train.txt', 'r')
            image_index = train_set_file.read().split('\n')
        elif self.name == 'kitti_val':
            val_set_file = open('data/kitti/splits/val.txt', 'r')
            image_index = val_set_file.read().split('\n')

        return image_index

    def gt_roidb(self):
        '''
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        '''
        cache_file = os.path.join(self._kitti_path, self.name + '_gt_roidb.pkl')
        print('cache file', cache_file)
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = [self._load_kitti_annotation(index)
                    for index in self.image_index]

        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    def remove_occluded_keypoints(self, objects, left=True):
        '''
            Generate the visible range of the bounding box according to
            the occlusion relations between all objects

            Remove almost totally occluded ones
        '''
        ix = 0 if left else 1

        depth_line = np.zeros(1260, dtype=float)
        for i in range(len(objects)):
            for col in range(int(objects[i].boxes[ix].box[0]), int(objects[i].boxes[ix].box[2])+1):
                pixel = depth_line[col]
                if pixel == 0.0:
                    depth_line[col] = objects[i].pos[2]
                elif objects[i].pos[2] < depth_line[col]:
                    depth_line[col] = (objects[i].pos[2]+pixel)/2.0

        for i in range(len(objects)):
            objects[i].boxes[ix].visible_left = objects[i].boxes[ix].box[0]
            objects[i].boxes[ix].visible_right = objects[i].boxes[ix].box[2]
            left_visible = True
            right_visible = True
            if depth_line[int(objects[i].boxes[ix].box[0])] < objects[i].pos[2]:
                left_visible = False
            if depth_line[int(objects[i].boxes[ix].box[2])] < objects[i].pos[2]:
                right_visible = False

            if right_visible == False and left_visible == False:
                objects[i].boxes[ix].visible_right = objects[i].boxes[ix].box[0]
                objects[i].boxes[ix].keypoints[:] = -1

            for col in range(int(objects[i].boxes[ix].box[0]), int(objects[i].boxes[ix].box[2])+1):
                if left_visible and depth_line[col] >= objects[i].pos[2]:
                    objects[i].boxes[ix].visible_right = col
                elif right_visible and depth_line[col] < objects[i].pos[2]:
                    objects[i].boxes[ix].visible_left = col

        objects = [x for x in objects if np.sum(x.boxes[ix].keypoints)>-4]

        for i in range(len(objects)):
            left_kpt = 5000
            right_kpt = 0
            for j in range(4):
                if objects[i].boxes[ix].keypoints[j] != -1:
                    if objects[i].boxes[ix].keypoints[j] < left_kpt:
                        left_kpt = objects[i].boxes[ix].keypoints[j]
                    if objects[i].boxes[ix].keypoints[j] > right_kpt:
                        right_kpt = objects[i].boxes[ix].keypoints[j]
            for j in range(4):
                if objects[i].boxes[ix].keypoints[j] != -1:
                    if objects[i].boxes[ix].keypoints[j] < objects[i].boxes[ix].visible_left-5 or \
                       objects[i].boxes[ix].keypoints[j] > objects[i].boxes[ix].visible_right+5 or \
                       objects[i].boxes[ix].keypoints[j] < left_kpt+3 or \
                       objects[i].boxes[ix].keypoints[j] > right_kpt-3:
                         objects[i].boxes[ix].keypoints[j] = -1
        return objects

    def _load_kitti_annotation(self,index):
        if self._image_set == 'test':
            objects = [] 
        else:
            filename = os.path.join(self._data_path, 'training', 'label_2', index + '.txt')

            calib_file = os.path.join(self._data_path, 'training', 'calib', index + '.txt')
            calib_it = kitti_utils.read_obj_calibration(calib_file)
            im_left = cv2.imread(self.img_left_path_from_index(index))
            objects_origin = kitti_utils.read_obj_data(filename, calib_it, im_left.shape)
            objects = [] 

            objects_origin = self.remove_occluded_keypoints(objects_origin)
            objects_origin = self.remove_occluded_keypoints(objects_origin, left=False)

            for i in range(len(objects_origin)):
                if objects_origin[i].truncate < 0.98 and objects_origin[i].occlusion < 3 and \
                   (objects_origin[i].boxes[0].box[3] - objects_origin[i].boxes[0].box[1])>10 and \
                   objects_origin[i].cls in self._classes and \
                   objects_origin[i].boxes[0].visible_right - objects_origin[i].boxes[0].visible_left > 3 and\
                   objects_origin[i].boxes[1].visible_right - objects_origin[i].boxes[1].visible_left > 3:
                    objects.append(objects_origin[i])

            f = calib_it.p2[0,0]
            cx = calib_it.p2[0,2]
            base_line = (calib_it.p2[0,3] - calib_it.p3[0,3])/f

            num_objs = len(objects)

            boxes_left = np.zeros((num_objs, 4), dtype=np.float32)
            boxes_right = np.zeros((num_objs, 4), dtype=np.float32)
            boxes_merge = np.zeros((num_objs, 4), dtype=np.float32)
            dim_orien = np.zeros((num_objs, 4), dtype=np.float32)
            kpts = np.zeros((num_objs, 6), dtype=np.float32)
            kpts_right = np.zeros((num_objs, 6), dtype=np.float32)
            truncation = np.zeros((num_objs), dtype=np.float32)
            occlusion = np.zeros((num_objs), dtype=np.float32)
            gt_classes = np.zeros((num_objs), dtype=np.int32)
            overlaps = np.zeros((num_objs, self.num_classes),dtype=np.float32)
            
            for i in range(len(objects)):
                cls = self._class_to_ind[objects[i].cls]
                boxes_left[i,:] = objects[i].boxes[0].box
                boxes_right[i,:] = objects[i].boxes[1].box
                boxes_merge[i,:] = objects[i].boxes[2].box
                dim_orien[i,0:3] = objects[i].dim
                dim_orien[i,3] = objects[i].alpha
                kpts[i,:4] = objects[i].boxes[0].keypoints
                kpts[i,4] = objects[i].boxes[0].visible_left
                kpts[i,5] = objects[i].boxes[0].visible_right
                kpts_right[i,:4] = objects[i].boxes[1].keypoints
                kpts_right[i,4] = objects[i].boxes[1].visible_left
                kpts_right[i,5] = objects[i].boxes[1].visible_right
                occlusion[i] = objects[i].occlusion
                truncation[i] = objects[i].truncate
                gt_classes[i] = cls
                overlaps[i, cls] = 1.0 

            overlaps = scipy.sparse.csr_matrix(overlaps)
            gt_subclasses = np.zeros((num_objs), dtype=np.int32)
            gt_subclasses_flipped = np.zeros((num_objs), dtype=np.int32)
            subindexes = np.zeros((num_objs, self.num_classes), dtype=np.int32)
            subindexes_flipped = np.zeros((num_objs, self.num_classes), dtype=np.int32)
            subindexes = scipy.sparse.csr_matrix(subindexes)
            subindexes_flipped = scipy.sparse.csr_matrix(subindexes_flipped)

            return {'boxes_left' : boxes_left, 
                    'boxes_right': boxes_right,
                    'boxes_merge': boxes_merge,
                    'dim_orien' : dim_orien,
                    'kpts' : kpts,
                    'kpts_right' : kpts_right,
                    'truncation' : truncation,
                    'occlusion' : occlusion,
                    'gt_classes': gt_classes,
                    'igt_subclasses': gt_subclasses,
                    'gt_subclasses_flipped': gt_subclasses_flipped,
                    'gt_overlaps' : overlaps,
                    'gt_subindexes': subindexes,
                    'gt_subindexes_flipped': subindexes_flipped,
                    'flipped' : False}
        
