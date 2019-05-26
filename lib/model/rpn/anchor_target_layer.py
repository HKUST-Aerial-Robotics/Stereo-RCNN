# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
# --------------------------------------------------------
# Reorganized and modified by Jianwei Yang and Jiasen Lu
# --------------------------------------------------------
# Modified by Peiliang Li for Stereo RCNN
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import numpy.random as npr

from model.utils.config import cfg
from model.rpn.generate_anchors import generate_anchors, generate_anchors_all_pyramids
from model.rpn.bbox_transform import clip_boxes, bbox_overlaps_batch, bbox_transform_batch

import pdb

DEBUG = False

class _AnchorTargetLayer(nn.Module):
    """
        Assign anchors to ground-truth targets. Produces anchor classification
        labels and bounding-box regression targets.
    """
    def __init__(self, feat_stride, ratios):
        super(_AnchorTargetLayer, self).__init__()
        self._anchor_ratios = ratios
        self._feat_stride = feat_stride
        self._fpn_scales = np.array(cfg.FPN_ANCHOR_SCALES)
        self._fpn_feature_strides = np.array(cfg.FPN_FEAT_STRIDES)
        self._fpn_anchor_stride = cfg.FPN_ANCHOR_STRIDE

        # allow boxes to sit over the edge by a small amount
        self._allowed_border = 0  # default is 0

    def forward(self, input):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate 9 anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the 9 anchors
        # filter out-of-image anchors

        scores = input[0]
        gt_boxes_left = input[1]
        gt_boxes_right = input[2]
        gt_boxes_merge = input[3]
        im_info = input[4]
        num_boxes = input[5]
        feat_shapes = input[6]

        # NOTE: need to change
        # height, width = scores.size(2), scores.size(3)
        height, width = 0, 0

        batch_size = gt_boxes_left.size(0)

        anchors = torch.from_numpy(generate_anchors_all_pyramids(self._fpn_scales, self._anchor_ratios, 
                feat_shapes, self._fpn_feature_strides, self._fpn_anchor_stride)).type_as(scores)    
        total_anchors = anchors.size(0)
        
        keep = ((anchors[:, 0] >= -self._allowed_border) &
                (anchors[:, 1] >= -self._allowed_border) &
                (anchors[:, 2] < int(im_info[0][1]) + self._allowed_border) &
                (anchors[:, 3] < int(im_info[0][0]) + self._allowed_border))

        inds_inside = torch.nonzero(keep).view(-1)

        # keep only inside anchors
        anchors = anchors[inds_inside, :]

        # label: 1 is positive, 0 is negative, -1 is dont care
        merged_labels = gt_boxes_left.new(batch_size, inds_inside.size(0)).fill_(-1)
        bbox_inside_weights = gt_boxes_left.new(batch_size, inds_inside.size(0)).zero_()
        bbox_outside_weights = gt_boxes_left.new(batch_size, inds_inside.size(0)).zero_()

        overlaps = bbox_overlaps_batch(anchors, gt_boxes_merge)

        max_overlaps, argmax_overlaps = torch.max(overlaps, 2)
        gt_max_overlaps, _ = torch.max(overlaps, 1)
        if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            merged_labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        gt_max_overlaps[gt_max_overlaps==0] = 1e-5
        keep = torch.sum(overlaps.eq(gt_max_overlaps.view(batch_size,1,-1).expand_as(overlaps)), 2)

        if torch.sum(keep) > 0:
            merged_labels[keep>0] = 1

        # fg label: above threshold IOU
        merged_labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

        if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            merged_labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)

        sum_fg = torch.sum((merged_labels == 1).int(), 1)
        sum_bg = torch.sum((merged_labels == 0).int(), 1)

        for i in range(batch_size):
            # subsample positive merged_labels if we have too many
            if sum_fg[i] > num_fg:
                fg_inds = torch.nonzero(merged_labels[i] == 1).view(-1)
                # torch.randperm seems has a bug on multi-gpu setting that cause the segfault. 
                # See https://github.com/pytorch/pytorch/issues/1868 for more details.
                # use numpy instead.                
                #rand_num = torch.randperm(fg_inds.size(0)).type_as(gt_boxes_left).long()
                rand_num = torch.from_numpy(np.random.permutation(fg_inds.size(0))).type_as(gt_boxes_left).long()
                disable_inds = fg_inds[rand_num[:fg_inds.size(0)-num_fg]]
                merged_labels[i][disable_inds] = -1

            num_bg = cfg.TRAIN.RPN_BATCHSIZE - sum_fg[i]

            # subsample negative merged_labels if we have too many
            if sum_bg[i] > num_bg:
                bg_inds = torch.nonzero(merged_labels[i] == 0).view(-1)
                #rand_num = torch.randperm(bg_inds.size(0)).type_as(gt_boxes_left).long()

                rand_num = torch.from_numpy(np.random.permutation(bg_inds.size(0))).type_as(gt_boxes_left).long()
                disable_inds = bg_inds[rand_num[:bg_inds.size(0)-num_bg]]
                merged_labels[i][disable_inds] = -1

        offset = torch.arange(0, batch_size)*gt_boxes_left.size(1)

        argmax_overlaps = argmax_overlaps + offset.view(batch_size, 1).type_as(argmax_overlaps)
        bbox_targets_left = _compute_targets_batch(anchors, gt_boxes_left.view(-1,5)[argmax_overlaps.view(-1), :].view(batch_size, -1, 5))
        bbox_targets_right = _compute_targets_batch(anchors, gt_boxes_right.view(-1,5)[argmax_overlaps.view(-1), :].view(batch_size, -1, 5))

        # use a single value instead of 4 values for easy index.
        bbox_inside_weights[merged_labels==1] = cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS[0]

        if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
            num_examples = torch.sum(merged_labels[i] >= 0)
            positive_weights = 1.0 / num_examples.item()
            negative_weights = 1.0 / num_examples.item()
        else:
            assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                    (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))

        bbox_outside_weights[merged_labels == 1] = positive_weights
        bbox_outside_weights[merged_labels == 0] = negative_weights

        merged_labels = _unmap(merged_labels, total_anchors, inds_inside, batch_size, fill=-1)
        bbox_targets_left = _unmap(bbox_targets_left, total_anchors, inds_inside, batch_size, fill=0)
        bbox_targets_right = _unmap(bbox_targets_right, total_anchors, inds_inside, batch_size, fill=0)
        bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, batch_size, fill=0)
        bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, batch_size, fill=0)

        outputs = []

        outputs.append(merged_labels)
        outputs.append(bbox_targets_left)
        outputs.append(bbox_targets_right)
        outputs.append(bbox_inside_weights)
        outputs.append(bbox_outside_weights)

        return outputs

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

def _unmap(data, count, inds, batch_size, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """

    if data.dim() == 2:
        ret = torch.Tensor(batch_size, count).fill_(fill).type_as(data)
        ret[:, inds] = data
    else:
        ret = torch.Tensor(batch_size, count, data.size(2)).fill_(fill).type_as(data)
        ret[:, inds,:] = data
    return ret


def _compute_targets_batch(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    return bbox_transform_batch(ex_rois, gt_rois[:, :, :4])