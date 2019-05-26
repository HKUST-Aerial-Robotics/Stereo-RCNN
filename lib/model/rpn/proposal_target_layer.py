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
from ..utils.config import cfg
from model.rpn.bbox_transform import bbox_overlaps_batch, bbox_transform_batch
import pdb

class _ProposalTargetLayer(nn.Module):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def __init__(self, nclasses):
        super(_ProposalTargetLayer, self).__init__()
        self._num_classes = nclasses
        self.BBOX_NORMALIZE_MEANS = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
        self.BBOX_NORMALIZE_STDS = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS)
        self.DIM_NORMALIZE_MEANS = torch.FloatTensor(cfg.TRAIN.DIM_NORMALIZE_MEANS)
        self.DIM_NORMALIZE_STDS = torch.FloatTensor(cfg.TRAIN.DIM_NORMALIZE_STDS)
        self.BBOX_INSIDE_WEIGHTS = torch.FloatTensor(cfg.TRAIN.BBOX_INSIDE_WEIGHTS)

    def forward(self, all_rois_left, all_rois_right, gt_boxes_left, gt_boxes_right, \
                gt_dim_orien, gt_kpts, num_boxes):

        self.BBOX_NORMALIZE_MEANS = self.BBOX_NORMALIZE_MEANS.type_as(gt_boxes_left)
        self.BBOX_NORMALIZE_STDS = self.BBOX_NORMALIZE_STDS.type_as(gt_boxes_left)
        self.DIM_NORMALIZE_MEANS = self.DIM_NORMALIZE_MEANS.type_as(gt_boxes_left)
        self.DIM_NORMALIZE_STDS = self.DIM_NORMALIZE_STDS.type_as(gt_boxes_left)
        self.BBOX_INSIDE_WEIGHTS = self.BBOX_INSIDE_WEIGHTS.type_as(gt_boxes_left)

        gt_boxes_append_left = gt_boxes_left.new(gt_boxes_left.size()).zero_()
        gt_boxes_append_left[:,:,1:5] = gt_boxes_left[:,:,:4]

        gt_boxes_append_right = gt_boxes_right.new(gt_boxes_right.size()).zero_()
        gt_boxes_append_right[:,:,1:5] = gt_boxes_right[:,:,:4]

        # Include ground-truth boxes in the set of candidate rois
        all_rois_left = torch.cat([all_rois_left, gt_boxes_append_left], 1)
        all_rois_right = torch.cat([all_rois_right, gt_boxes_append_right], 1)

        num_images = 1
        rois_per_image = int(cfg.TRAIN.BATCH_SIZE / num_images)
        fg_rois_per_image = int(np.round(cfg.TRAIN.FG_FRACTION * rois_per_image))

        labels, rois_left, rois_right, gt_assign_left, bbox_targets_left, bbox_targets_right, \
            dim_orien_targets, kpts_targets, kpts_weight, bbox_inside_weights = self._sample_rois_pytorch(\
            all_rois_left, all_rois_right, gt_boxes_left, gt_boxes_right, gt_dim_orien, \
            gt_kpts, fg_rois_per_image, rois_per_image, self._num_classes)

        bbox_outside_weights = (bbox_inside_weights > 0).float()

        return rois_left, rois_right, labels, bbox_targets_left, bbox_targets_right,\
               dim_orien_targets, kpts_targets, kpts_weight, bbox_inside_weights, bbox_outside_weights

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def _get_bbox_regression_labels_pytorch(self, bbox_target_data, labels_batch, num_classes):
        """Bounding-box regression targets (bbox_target_data) are stored in a
        compact form b x N x (class, tx, ty, tw, th)
        This function expands those targets into the 4-of-4*K representation used
        by the network (i.e. only one class has non-zero targets).
        Returns:
            bbox_target (ndarray): b x N x 4K blob of regression targets
            bbox_inside_weights (ndarray): b x N x 4K blob of loss weights
        """
        batch_size = labels_batch.size(0)
        rois_per_image = labels_batch.size(1)
        clss = labels_batch
        bbox_targets = bbox_target_data.new(batch_size, rois_per_image, 4).zero_()
        bbox_inside_weights = bbox_target_data.new(bbox_targets.size()).zero_()

        for b in range(batch_size):
            # assert clss[b].sum() > 0
            if clss[b].sum() == 0:
                continue
            inds = torch.nonzero(clss[b] > 0).view(-1)
            for i in range(inds.numel()):
                ind = inds[i]
                bbox_targets[b, ind, :] = bbox_target_data[b, ind, :]
                bbox_inside_weights[b, ind, :] = self.BBOX_INSIDE_WEIGHTS

        return bbox_targets, bbox_inside_weights

    def _get_dim_orien_regression_labels_pytorch(self, dim_orien_target_data, labels_batch, num_classes): 
        
        batch_size = labels_batch.size(0) 
        rois_per_image = labels_batch.size(1)
        clss = labels_batch
        dim_orien_targets = dim_orien_target_data.new(batch_size, rois_per_image, 5).zero_() 

        for b in range(batch_size): 
            # assert clss[b].sum() > 0 
            if clss[b].sum() == 0:
                continue
            inds = torch.nonzero(clss[b] > 0).view(-1)
            for i in range(inds.numel()):
                ind = inds[i]
                dim_orien_targets[b, ind, :] = dim_orien_target_data[b, ind, :] 

        return dim_orien_targets

    def _get_kpts_regression_labels_pytorch(self, kpts_target_data, labels_batch, num_classes, kpts_weight):

        batch_size = labels_batch.size(0)
        rois_per_image = labels_batch.size(-1)
        clss = labels_batch
        kpts_targets = kpts_target_data.new(batch_size, rois_per_image, 3).zero_()
        weight = kpts_weight.new(batch_size, rois_per_image, 3).zero_()
        for b in range(batch_size):
            # assert clss[b].sum() > 0
            if clss[b].sum() == 0:
                continue
            inds = torch.nonzero(clss[b] == 1).view(-1)
            for i in range(inds.numel()):
                ind = inds[i]
                kpts_targets[b, ind] = kpts_target_data[b, ind]
                weight[b, ind] = kpts_weight[b,ind]
        return kpts_targets, weight

    def _compute_targets_pytorch(self, ex_rois, gt_rois):

        assert ex_rois.size(1) == gt_rois.size(1)
        assert ex_rois.size(2) == 4
        assert gt_rois.size(2) == 4 

        batch_size = ex_rois.size(0)
        rois_per_image = ex_rois.size(1)

        targets = bbox_transform_batch(ex_rois, gt_rois)
        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            # Optionally normalize targets by a precomputed mean and stdev
            targets = ((targets - self.BBOX_NORMALIZE_MEANS.expand_as(targets))
                        / self.BBOX_NORMALIZE_STDS.expand_as(targets))

        return targets

    def _compute_dim_orien_targets_pytorch(self, gt_dim_orien):

        assert gt_dim_orien.size(2) == 5

        if cfg.TRAIN.DIM_NORMALIZE_TARGETS_PRECOMPUTED:
            # Optionally normalize targets by a precomputed mean and stdev
            target_dim_orien = ((gt_dim_orien - self.DIM_NORMALIZE_MEANS.expand_as(gt_dim_orien))
                                / self.DIM_NORMALIZE_STDS.expand_as(gt_dim_orien))

        return target_dim_orien

    def _compute_kpts_targets_pytorch(self, ex_rois, gt_kpts):

        assert ex_rois.size(1) == gt_kpts.size(1)
        assert ex_rois.size(2) == 4
        assert gt_kpts.size(2) == 6

        grid_size = cfg.KPTS_GRID
        start = ex_rois[:,:,0]  # bs x 128
        start = start.unsqueeze(2).expand(-1,-1,6) # bs x 128 x 6, k0, k1, k2, k3, left_b, right_b
        width = ex_rois[:,:,2] - ex_rois[:,:,0] + 1 
        width = width.unsqueeze(2).expand(-1,-1,6) 
        target = torch.round((gt_kpts - start)*grid_size/width) 
        target[target < 0] = -225 
        target[target > grid_size-1] = -225 
        kpts_pos, kpts_type = torch.max(target[:,:,:4], 2) # B x num_rois 
        kpts_pos = kpts_pos.unsqueeze(2) 
        kpts_type = kpts_type.unsqueeze(2) 
        target = torch.cat((kpts_type.type(torch.cuda.FloatTensor)*grid_size+kpts_pos,\
                            target[:,:,4:].type(torch.cuda.FloatTensor)),2) 
        weight = target.new(target.size()).zero_() 
        weight[:] = 1 
        weight[target < 0] = 0 
        target[target < 0]= 0 
        
        return target.type(torch.cuda.LongTensor), weight # bs x 128 x 3


    def _sample_rois_pytorch(self, all_rois_left, all_rois_right, gt_boxes_left, gt_boxes_right, \
                            gt_dim_orien, gt_kpts, fg_rois_per_image, rois_per_image, num_classes):
        """Generate a random sample of RoIs comprising foreground and background
        examples.
        """
        # overlaps: (rois x gt_boxes)
        overlaps_left = bbox_overlaps_batch(all_rois_left, gt_boxes_left)  # B x num_rois x num_gt
        overlaps_right = bbox_overlaps_batch(all_rois_right, gt_boxes_right) # B x num_rois(2030) x num_gt(30)

        max_overlaps_left, gt_assignment_left = torch.max(overlaps_left, 2)  # B x num_rois(2030)
        max_overlaps_right, gt_assignment_right = torch.max(overlaps_right, 2)

        batch_size = overlaps_left.size(0)
        num_proposal = overlaps_left.size(1)
        num_boxes_per_img = overlaps_left.size(2)

        offset = torch.arange(0, batch_size)*gt_boxes_left.size(1)
        offset = offset.view(-1, 1).type_as(gt_assignment_left) + gt_assignment_left

        labels = gt_boxes_left[:,:,4].contiguous().view(-1)[(offset.view(-1),)].view(batch_size, -1)
        labels_batch = labels.new(batch_size, rois_per_image).zero_()

        rois_batch_left  = all_rois_left.new(batch_size, rois_per_image, 5).zero_()
        gt_assign_batch_left = all_rois_left.new(batch_size, rois_per_image).zero_()
        gt_rois_batch_left = all_rois_left.new(batch_size, rois_per_image, 5).zero_()

        rois_batch_right  = all_rois_right.new(batch_size, rois_per_image, 5).zero_()
        gt_assign_batch_right = all_rois_right.new(batch_size, rois_per_image).zero_()
        gt_rois_batch_right = all_rois_right.new(batch_size, rois_per_image, 5).zero_()

        gt_dim_orien_batch = all_rois_right.new(batch_size, rois_per_image, 5).zero_()
        gt_kpts_batch = all_rois_right.new(batch_size, rois_per_image, 6).zero_()
        # Guard against the case when an image has fewer than max_fg_rois_per_image
        # foreground RoIs
        for i in range(batch_size):

            fg_inds = torch.nonzero((max_overlaps_left[i] >= cfg.TRAIN.FG_THRESH) &\
                                    (max_overlaps_right[i] >= cfg.TRAIN.FG_THRESH) & \
                                    (gt_assignment_left[i] == gt_assignment_right[i])).view(-1)
            fg_num_rois = fg_inds.numel()

            # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
            bg_inds_left = torch.nonzero((max_overlaps_left[i] < cfg.TRAIN.BG_THRESH_HI) &
                                    (max_overlaps_left[i] >= cfg.TRAIN.BG_THRESH_LO)).view(-1)
            bg_inds_right = torch.nonzero((max_overlaps_right[i] < cfg.TRAIN.BG_THRESH_HI) &
                                    (max_overlaps_right[i] >= cfg.TRAIN.BG_THRESH_LO)).view(-1)
            bg_inds = torch.from_numpy(np.union1d(bg_inds_left.cpu().numpy(), bg_inds_right.cpu().numpy())).cuda()
            bg_num_rois = bg_inds.numel()

            if fg_num_rois > 0 and bg_num_rois > 0:
                # sampling fg
                fg_rois_per_this_image = min(fg_rois_per_image, fg_num_rois)

                # torch.randperm seems has a bug on multi-gpu setting that cause the segfault.
                # See https://github.com/pytorch/pytorch/issues/1868 for more details.
                # use numpy instead.
                #rand_num = torch.randperm(fg_num_rois).long().cuda()
                rand_num = torch.from_numpy(np.random.permutation(fg_num_rois)).type_as(gt_boxes_left).long()
                fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]]

                # sampling bg
                bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image

                # Seems torch.rand has a bug, it will generate very large number and make an error.
                # We use numpy rand instead.
                #rand_num = (torch.rand(bg_rois_per_this_image) * bg_num_rois).long().cuda()
                rand_num = np.floor(np.random.rand(bg_rois_per_this_image) * bg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes_left).long()
                bg_inds = bg_inds[rand_num]

            elif fg_num_rois > 0 and bg_num_rois == 0:
                # sampling fg
                #rand_num = torch.floor(torch.rand(rois_per_image) * fg_num_rois).long().cuda()
                rand_num = np.floor(np.random.rand(rois_per_image) * fg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes_left).long()
                fg_inds = fg_inds[rand_num]
                fg_rois_per_this_image = rois_per_image
                bg_rois_per_this_image = 0
            elif bg_num_rois > 0 and fg_num_rois == 0:
                # sampling bg
                #rand_num = torch.floor(torch.rand(rois_per_image) * bg_num_rois).long().cuda()
                rand_num = np.floor(np.random.rand(rois_per_image) * bg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes_left).long()

                bg_inds = bg_inds[rand_num]
                bg_rois_per_this_image = rois_per_image
                fg_rois_per_this_image = 0
            else:
                raise ValueError("bg_num_rois = 0 and fg_num_rois = 0, this should not happen!")

            # The indices that we're selecting (both fg and bg)
            keep_inds = torch.cat([fg_inds, bg_inds], 0)

            # Select sampled values from various arrays:
            labels_batch[i].copy_(labels[i][keep_inds])

            # Clamp labels for the background RoIs to 0
            labels_batch[i][fg_rois_per_this_image:] = 0

            rois_batch_left[i] = all_rois_left[i][keep_inds]
            rois_batch_left[i,:,0] = i

            rois_batch_right[i] = all_rois_right[i][keep_inds]
            rois_batch_right[i,:,0] = i

            # TODO: check the below line when batch_size > 1, no need to add offset here
            gt_assign_batch_left[i] = gt_assignment_left[i][keep_inds]
            gt_assign_batch_right[i] = gt_assignment_right[i][keep_inds]

            gt_rois_batch_left[i] = gt_boxes_left[i][gt_assignment_left[i][keep_inds]]
            gt_rois_batch_right[i] = gt_boxes_right[i][gt_assignment_right[i][keep_inds]]

            gt_dim_orien_batch[i] = gt_dim_orien[i][gt_assignment_left[i][keep_inds]]
            gt_kpts_batch[i] = gt_kpts[i][gt_assignment_left[i][keep_inds]]

        bbox_target_data_left = self._compute_targets_pytorch(
                rois_batch_left[:,:,1:5], gt_rois_batch_left[:,:,:4])
        
        bbox_target_data_right = self._compute_targets_pytorch(
                rois_batch_right[:,:,1:5], gt_rois_batch_right[:,:,:4])

        dim_orien_target_data = self._compute_dim_orien_targets_pytorch(gt_dim_orien_batch)

        kpts_target_data, kpts_weight = self._compute_kpts_targets_pytorch(rois_batch_left[:,:,1:5], gt_kpts_batch)
        
        bbox_targets_left, bbox_inside_weights_left = \
                self._get_bbox_regression_labels_pytorch(bbox_target_data_left, labels_batch, num_classes)

        bbox_targets_right, bbox_inside_weights_right = \
                self._get_bbox_regression_labels_pytorch(bbox_target_data_right, labels_batch, num_classes)

        dim_orien_target = self._get_dim_orien_regression_labels_pytorch(dim_orien_target_data, labels_batch, num_classes)

        kpts_targets, kpts_weight = self._get_kpts_regression_labels_pytorch(kpts_target_data, labels_batch, num_classes, kpts_weight)
        
        return labels_batch, rois_batch_left, rois_batch_right, gt_assign_batch_left, \
               bbox_targets_left, bbox_targets_right, dim_orien_target, kpts_targets, kpts_weight, bbox_inside_weights_left





