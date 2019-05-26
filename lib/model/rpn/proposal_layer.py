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
import math
from model.utils.config import cfg
from model.rpn.generate_anchors import generate_anchors, generate_anchors_all_pyramids
from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes, clip_boxes_batch
from model.roi_layers import nms

import pdb

DEBUG = False

class _ProposalLayer(nn.Module):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def __init__(self, feat_stride, ratios):
        super(_ProposalLayer, self).__init__()
        self._anchor_ratios = ratios
        self._feat_stride = feat_stride
        self._fpn_scales = np.array(cfg.FPN_ANCHOR_SCALES)
        self._fpn_feature_strides = np.array(cfg.FPN_FEAT_STRIDES)
        self._fpn_anchor_stride = cfg.FPN_ANCHOR_STRIDE
        # self._anchors = torch.from_numpy(generate_anchors_all_pyramids(self._fpn_scales, ratios, self._fpn_feature_strides, fpn_anchor_stride))
        # self._num_anchors = self._anchors.size(0)

    def forward(self, input):

        # Algorithm:
        #
        # for each (H, W) location i
        #   generate A anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the A anchors
        # clip predicted boxes to image
        # remove predicted boxes with either height or width < threshold
        # sort all (proposal, score) pairs by score from highest to lowest
        # take top pre_nms_topN proposals before NMS
        # apply NMS with threshold 0.7 to remaining proposals
        # take after_nms_topN proposals after NMS
        # return the top proposals (-> RoIs top, scores top)


        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs
        scores = input[0][:, :, 1]  # batch_size x num_rois x 1
        bbox_deltas_left_right = input[1]      # batch_size x num_rois x 4
        bbox_deltas_left = bbox_deltas_left_right[:,:,:4].clone()
        bbox_deltas_right = bbox_deltas_left_right[:,:,:4].clone()
        bbox_deltas_right[:,:,0] = bbox_deltas_left_right[:,:,4]
        bbox_deltas_right[:,:,2] = bbox_deltas_left_right[:,:,5]
        im_info = input[2]
        cfg_key = input[3]
        feat_shapes = input[4]        

        pre_nms_topN  = cfg[cfg_key].RPN_PRE_NMS_TOP_N
        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
        nms_thresh    = cfg[cfg_key].RPN_NMS_THRESH
        min_size      = cfg[cfg_key].RPN_MIN_SIZE

        batch_size = bbox_deltas_left.size(0)

        anchors = torch.from_numpy(generate_anchors_all_pyramids(self._fpn_scales, self._anchor_ratios, 
                feat_shapes, self._fpn_feature_strides, self._fpn_anchor_stride)).type_as(scores)
        num_anchors = anchors.size(0)

        anchors = anchors.view(1, num_anchors, 4).expand(batch_size, num_anchors, 4)

        # Convert anchors into proposals via bbox transformations
        proposals_left = bbox_transform_inv(anchors, bbox_deltas_left, batch_size)
        proposals_right = bbox_transform_inv(anchors, bbox_deltas_right, batch_size)

        # 2. clip predicted boxes to image
        proposals_left = clip_boxes(proposals_left, im_info, batch_size)
        proposals_right = clip_boxes(proposals_right, im_info, batch_size)
        # keep_idx = self._filter_boxes(proposals, min_size).squeeze().long().nonzero().squeeze()
                
        scores_keep = scores
        proposals_keep_left = proposals_left
        proposals_keep_right = proposals_right

        _, order = torch.sort(scores_keep, 1, True)

        output_left = scores.new(batch_size, post_nms_topN, 5).zero_()
        output_right = scores.new(batch_size, post_nms_topN, 5).zero_()
        for i in range(batch_size):
            # # 3. remove predicted boxes with either height or width < threshold
            # # (NOTE: convert min_size to input image scale stored in im_info[2])
            proposals_single_left = proposals_keep_left[i]
            proposals_single_right = proposals_keep_right[i]
            scores_single = scores_keep[i]

            # # 4. sort all (proposal, score) pairs by score from highest to lowest
            # # 5. take top pre_nms_topN (e.g. 6000)
            order_single = order[i]

            if pre_nms_topN > 0 and pre_nms_topN < scores_keep.numel():
                order_single = order_single[:pre_nms_topN]

            proposals_single_left = proposals_single_left[order_single, :]
            proposals_single_right = proposals_single_right[order_single, :]
            scores_single = scores_single[order_single].view(-1,1)

            # 6. apply nms (e.g. threshold = 0.7)
            # 7. take after_nms_topN (e.g. 300)
            # 8. return the top proposals (-> RoIs top)
            keep_idx_i_left = nms(proposals_single_left, scores_single.squeeze(1), nms_thresh)
            keep_idx_i_left = keep_idx_i_left.long().view(-1)

            keep_idx_i_right = nms(proposals_single_right, scores_single.squeeze(1), nms_thresh)
            keep_idx_i_right = keep_idx_i_right.long().view(-1)

            keep_idx_i = torch.from_numpy(np.intersect1d(keep_idx_i_left.cpu().numpy(), \
                                                         keep_idx_i_right.cpu().numpy())).cuda()
            if post_nms_topN > 0:
                keep_idx_i = keep_idx_i[:post_nms_topN]

            proposals_single_left = proposals_single_left[keep_idx_i, :]
            proposals_single_right = proposals_single_right[keep_idx_i, :]
            scores_single = scores_single[keep_idx_i, :]

            # padding 0 at the end.
            num_proposal = proposals_single_left.size(0)
            output_left[i,:,0] = i
            output_left[i,:num_proposal,1:] = proposals_single_left

            output_right[i,:,0] = i
            output_right[i,:num_proposal,1:] = proposals_single_right

        return output_left, output_right

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def _filter_boxes(self, boxes, min_size):
        """Remove all boxes with any side smaller than min_size."""
        ws = boxes[:, :, 2] - boxes[:, :, 0] + 1
        hs = boxes[:, :, 3] - boxes[:, :, 1] + 1
        keep = ((ws >= min_size) & (hs >= min_size))
        return keep
