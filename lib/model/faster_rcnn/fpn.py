import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, gradcheck
from torch.autograd.gradcheck import gradgradcheck
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
import torchvision.utils as vutils
from model.utils.config import cfg
from model.rpn.rpn_fpn import _RPN_FPN
from model.roi_pooling.modules.roi_pool import _RoIPooling
from model.roi_crop.modules.roi_crop import _RoICrop
from model.roi_align.modules.roi_align import RoIAlignAvg
from model.rpn.proposal_target_layer_fpn import _ProposalTargetLayer
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta
import time
import pdb

class _FPN(nn.Module):
    """ FPN """
    def __init__(self, classes, class_agnostic):
        super(_FPN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.maxdisp = 200
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox_left_right = 0
        self.RCNN_loss_dis = 0
        self.RCNN_loss_dim = 0
        self.RCNN_loss_dim_orien = 0
        self.RCNN_loss_kpts = 0

        self.maxpool2d = nn.MaxPool2d(1, stride=2)
        # define rpn
        self.RCNN_rpn = _RPN_FPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)

        # NOTE: the original paper used pool_size = 7 for cls branch, and 14 for mask branch, to save the
        # computation time, we first use 14 as the pool_size, and then do stride=2 pooling for cls branch.
        self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        self.RCNN_roi_kpts_align = RoIAlignAvg(cfg.POOLING_SIZE*2, cfg.POOLING_SIZE*2, 1.0/16.0)
        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        self.RCNN_roi_crop = _RoICrop()

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        # custom weights initialization called on netG and netD
        def weights_init(m, mean, stddev, truncated=False):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)
            elif classname.find('BatchNorm') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

        normal_init(self.RCNN_toplayer, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_smooth1, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_smooth2, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_smooth3, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_latlayer1, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_latlayer2, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_latlayer3, 0, 0.01, cfg.TRAIN.TRUNCATED)

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred_left_right, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_dim_orien_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)
        normal_init(self.kpts_class, 0, 0.1, cfg.TRAIN.TRUNCATED)
        weights_init(self.RCNN_top, 0, 0.01, cfg.TRAIN.TRUNCATED)
        weights_init(self.RCNN_kpts, 0, 0.1, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y

    def PyramidRoI_Feat(self, feat_maps, rois, im_info, kpts=False, single_level=None):
        ''' roi pool on pyramid feature maps'''
        # do roi pooling based on predicted rois
        img_area = im_info[0][0] * im_info[0][1]
        h = rois.data[:, 4] - rois.data[:, 2] + 1
        w = rois.data[:, 3] - rois.data[:, 1] + 1
        roi_level = torch.log(torch.sqrt(h * w) / 224.0)
        roi_level = torch.round(roi_level + 4)
        roi_level[roi_level < 2] = 2
        roi_level[roi_level > 5] = 5
        # roi_level.fill_(5)
        if cfg.POOLING_MODE == 'crop':
            # pdb.set_trace()
            # pooled_feat_anchor = _crop_pool_layer(base_feat, rois.view(-1, 5))
            # NOTE: need to add pyrmaid
            grid_xy = _affine_grid_gen(rois, base_feat.size()[2:], self.grid_size)
            grid_yx = torch.stack([grid_xy.data[:,:,:,1], grid_xy.data[:,:,:,0]], 3).contiguous()
            roi_pool_feat = self.RCNN_roi_crop(base_feat, Variable(grid_yx).detach())
            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                roi_pool_feat = F.max_pool2d(roi_pool_feat, 2, 2)

        elif cfg.POOLING_MODE == 'align':
            roi_pool_feats = []
            box_to_levels = []
            for i, l in enumerate(range(2, 6)):
                if (roi_level == l).sum() == 0:
                    continue
                idx_l = (roi_level == l).nonzero().squeeze()
                box_to_levels.append(idx_l)
                scale = feat_maps[i].size(2) / im_info[0][0]
                if kpts is True:
                    feat = self.RCNN_roi_kpts_align(feat_maps[i], rois[idx_l], scale)
                else:
                    feat = self.RCNN_roi_align(feat_maps[i], rois[idx_l], scale)
                roi_pool_feats.append(feat)
            roi_pool_feat = torch.cat(roi_pool_feats, 0)
            box_to_level = torch.cat(box_to_levels, 0)
            idx_sorted, order = torch.sort(box_to_level)
            roi_pool_feat = roi_pool_feat[order]

        elif cfg.POOLING_MODE == 'pool':
            roi_pool_feats = []
            box_to_levels = []
            for i, l in enumerate(range(2, 6)):
                if (roi_level == l).sum() == 0:
                    continue
                idx_l = (roi_level == l).nonzero().squeeze()
                box_to_levels.append(idx_l)
                scale = feat_maps[i].size(2) / im_info[0][0]
                feat = self.RCNN_roi_pool(feat_maps[i], rois[idx_l], scale)
                roi_pool_feats.append(feat)
            roi_pool_feat = torch.cat(roi_pool_feats, 0)
            box_to_level = torch.cat(box_to_levels, 0)
            idx_sorted, order = torch.sort(box_to_level)
            roi_pool_feat = roi_pool_feat[order]
            
        return roi_pool_feat

    def forward(self, im_left_data,  im_info, gt_boxes_left, gt_dis, gt_dim_orien, gt_kpts, num_boxes):
        batch_size = im_left_data.size(0)

        im_info = im_info.data
        gt_boxes_left = gt_boxes_left.data
        gt_dis = gt_dis.data
        gt_dim_orien = gt_dim_orien.data
        gt_kpts = gt_kpts.data
        num_boxes = num_boxes.data

        # feed left image data to base model to obtain base feature map
        # Bottom-up
        c1_left = self.RCNN_layer0(im_left_data) # 64 x 1/4
        c2_left = self.RCNN_layer1(c1_left)      # 256 x 1/4
        c3_left = self.RCNN_layer2(c2_left)      # 512 x 1/8
        c4_left = self.RCNN_layer3(c3_left)      # 1024 x 1/16
        c5_left = self.RCNN_layer4(c4_left)      # 2048 x 1/32
        # Top-down
        p5_left = self.RCNN_toplayer(c5_left)    # 256 x 1/32
        p4_left = self._upsample_add(p5_left, self.RCNN_latlayer1(c4_left))
        p4_left = self.RCNN_smooth1(p4_left)     # 256 x 1/16
        p3_left = self._upsample_add(p4_left, self.RCNN_latlayer2(c3_left))
        p3_left = self.RCNN_smooth2(p3_left)     # 256 x 1/8
        p2_left = self._upsample_add(p3_left, self.RCNN_latlayer3(c2_left))
        p2_left = self.RCNN_smooth3(p2_left)     # 256 x 1/4
        p6_left = self.maxpool2d(p5_left)        # 256 x 1/64

        rpn_feature_maps_left = [p2_left, p3_left, p4_left, p5_left, p6_left]
        mrcnn_feature_maps_left = [p2_left, p3_left, p4_left, p5_left]

        rois_left, rpn_loss_cls, rpn_loss_bbox_left_right = \
            self.RCNN_rpn(rpn_feature_maps_left, im_info, gt_boxes_left, num_boxes)

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois_left, gt_boxes_left, gt_dim_orien, gt_kpts, num_boxes)
            rois_left, rois_label, rois_target_left, \
            rois_target_dim_orien, kpts_label_all, kpts_weight_all, rois_inside_ws4, rois_outside_ws4 = roi_data

            rois_target_left_right = rois_target_left.new(rois_target_left.size()[0],rois_target_left.size()[1],4)
            rois_target_left_right[:,:,:4] = rois_target_left

            rois_inside_ws = rois_inside_ws4.new(rois_inside_ws4.size()[0],rois_inside_ws4.size()[1],4)
            rois_inside_ws[:,:,:4] = rois_inside_ws4

            rois_outside_ws = rois_outside_ws4.new(rois_outside_ws4.size()[0],rois_outside_ws4.size()[1],4)
            rois_outside_ws[:,:,:4] = rois_outside_ws4

            rois_label = rois_label.view(-1).long()
            rois_label = Variable(rois_label)
            kpts_label = Variable(kpts_label_all[:,:,0].contiguous().view(-1))
            left_border_label = Variable(kpts_label_all[:,:,1].contiguous().view(-1))
            right_border_label = Variable(kpts_label_all[:,:,2].contiguous().view(-1))

            kpts_weight = Variable(kpts_weight_all[:,:,0].contiguous().view(-1))
            left_border_weight = Variable(kpts_weight_all[:,:,1].contiguous().view(-1))
            right_border_weight = Variable(kpts_weight_all[:,:,2].contiguous().view(-1))

            rois_target_left_right = Variable(rois_target_left_right.view(-1, rois_target_left_right.size(2)))
            rois_target_dim_orien = Variable(rois_target_dim_orien.view(-1, rois_target_dim_orien.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target_left_right = None
            rois_target_dim_orien = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois_left = rois_left.view(-1,5)
        rois_left = Variable(rois_left)

        # pooling features based on rois, output 14x14 map
        #roi_feat_left = self.PyramidRoI_Feat(mrcnn_feature_maps_left, rois_left, im_info)
        #roi_feat_right = self.PyramidRoI_Feat(mrcnn_feature_maps_right, rois_right, im_info)
        roi_feat_semantic = self.PyramidRoI_Feat(mrcnn_feature_maps_left, rois_left, im_info)

        # feed pooled features to top model
        roi_feat_semantic = self._head_to_tail(roi_feat_semantic)
        bbox_pred = self.RCNN_bbox_pred(roi_feat_semantic)            # num x 6
        dim_orien_pred = self.RCNN_dim_orien_pred(roi_feat_semantic)  # num x 5

        cls_score = self.RCNN_cls_score(roi_feat_semantic)
        cls_prob = F.softmax(cls_score, 1) 

        # for keypoint
        roi_feat_dense = self.PyramidRoI_Feat(mrcnn_feature_maps_left, rois_left, im_info, kpts=True)
        roi_feat_dense = self.RCNN_kpts(roi_feat_dense) # num x 256 x 28 x 28
        kpts_pred_all = self.kpts_class(roi_feat_dense) # num x 6 x cfg.KPTS_GRID x cfg.KPTS_GRID
        kpts_pred_all = kpts_pred_all.sum(2)            # num x 6 x cfg.KPTS_GRID
        kpts_pred = kpts_pred_all[:,:4,:].contiguous().view(-1, 4*cfg.KPTS_GRID)
        kpts_prob = F.softmax(kpts_pred,1) # num x (4xcfg.KPTS_GRID) 

        left_border_pred = kpts_pred_all[:,4,:].contiguous().view(-1, cfg.KPTS_GRID)
        left_border_prob = F.softmax(left_border_pred,1) # num x cfg.KPTS_GRID

        right_border_pred = kpts_pred_all[:,5,:].contiguous().view(-1, cfg.KPTS_GRID)
        right_border_prob = F.softmax(right_border_pred,1) # num x cfg.KPTS_GRID

        if self.training and not self.class_agnostic:
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1)/4), 4) # (128L, 2L, 6L)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.long().view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

            dim_orien_pred_view = dim_orien_pred.view(dim_orien_pred.size(0), int(dim_orien_pred.size(1)/5), 5) # (128L, 4L, 5L)
            dim_orien_pred_select = torch.gather(dim_orien_pred_view, 1, rois_label.long().view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 5))
            dim_orien_pred = dim_orien_pred_select.squeeze(1) 

        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0 

        if self.training:
            # classification loss
            self.RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)
            # bounding box regression L1 loss
            self.RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target_left_right, rois_inside_ws, rois_outside_ws)
            self.RCNN_loss_dim_orien = _smooth_l1_loss(dim_orien_pred, rois_target_dim_orien)
            
            kpts_pred = kpts_pred.view(-1, 4*cfg.KPTS_GRID)
            kpts_label = kpts_label.view(-1)
            self.RCNN_loss_kpts = F.cross_entropy(kpts_pred, kpts_label, reduce=False)
            if torch.sum(kpts_weight).data[0] < 1:
                self.RCNN_loss_kpts = torch.sum(self.RCNN_loss_kpts*kpts_weight)
            else:
                self.RCNN_loss_kpts = torch.sum(self.RCNN_loss_kpts*kpts_weight)/torch.sum(kpts_weight)

            self.RCNN_loss_left_border = F.cross_entropy(left_border_pred, left_border_label, reduce=False)
            if torch.sum(left_border_weight).data[0] < 1:
                self.RCNN_loss_left_border = torch.sum(self.RCNN_loss_left_border*left_border_weight)
            else:
                self.RCNN_loss_left_border = torch.sum(self.RCNN_loss_left_border*left_border_weight)/torch.sum(left_border_weight)

            self.RCNN_loss_right_border = F.cross_entropy(right_border_pred, right_border_label, reduce=False)
            if torch.sum(right_border_weight).data[0]<1:
                self.RCNN_loss_right_border = torch.sum(self.RCNN_loss_right_border*right_border_weight)
            else:
                self.RCNN_loss_right_border = torch.sum(self.RCNN_loss_right_border*right_border_weight)/torch.sum(right_border_weight)
            self.RCNN_loss_kpts = (self.RCNN_loss_kpts+self.RCNN_loss_left_border+self.RCNN_loss_right_border)/3.0

        rois_left = rois_left.view(batch_size,-1, rois_left.size(1))
        cls_prob = cls_prob.view(batch_size, -1, cls_prob.size(1))
        bbox_pred = bbox_pred.view(batch_size, -1, bbox_pred.size(1))
        dim_orien_pred = dim_orien_pred.view(batch_size, -1, dim_orien_pred.size(1)) 

        if self.training:
            rois_label = rois_label.view(batch_size, -1)

        return rois_left, cls_prob, bbox_pred, dim_orien_pred, \
               kpts_prob, left_border_prob, right_border_prob, rpn_loss_cls, rpn_loss_bbox_left_right, \
               self.RCNN_loss_cls, self.RCNN_loss_bbox, self.RCNN_loss_dim_orien, self.RCNN_loss_kpts, rois_label  









