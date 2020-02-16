# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import pickle
import math as m
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_transform_inv, kpts_transform_inv, border_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.faster_rcnn.resnet import resnet
from model.align_layer.align_layer import AlignLayer
from model.utils import kitti_utils
from model.utils import vis_3d_utils as vis_utils
from model.utils import optimizer as optimizer

import pdb

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

def infer_boundary(cls_dets_left):
    left_right = cls_dets_left.new(cls_dets_left.size()[0],2)
    depth_line = np.zeros(1260, dtype=float)
    for i in range(cls_dets_left.size()[0]):
        for col in range(int(cls_dets_left[i,0]), int(cls_dets_left[i,2])+1):
            pixel = depth_line[col]
            depth = 1050.0/cls_dets_left[i,3]
            if pixel == 0.0:
                depth_line[col] = depth
            elif depth < depth_line[col]:
                depth_line[col] = (depth+pixel)/2.0

    for i in range(cls_dets_left.size()[0]):
        left_right[i,0] = cls_dets_left[i,0]
        left_right[i,1] = cls_dets_left[i,2]
        left_visible = True
        right_visible = True
        if depth_line[int(cls_dets_left[i,0])] < 1050.0/cls_dets_left[i,3]:
            left_visible = False
        if depth_line[int(cls_dets_left[i,2])] < 1050.0/cls_dets_left[i,3]:
            right_visible = False

        if right_visible == False and left_visible == False:
            left_right[i,1] = cls_dets_left[i,0]

        for col in range(int(cls_dets_left[i,0]), int(cls_dets_left[i,2])+1):
            if left_visible and depth_line[col] >= 1050.0/cls_dets_left[i,3]:
                left_right[i,1] = col
            elif right_visible and depth_line[col] < 1050.0/cls_dets_left[i,3]:
                left_right[i,0] = col
    return left_right

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='kitti', type=str)
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default='cfgs/vgg16.yml', type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152',
                      default='res101', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)
  parser.add_argument('--load_dir', dest='load_dir',
                      help='directory to load models', default="models",
                      type=str)
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--ls', dest='large_scale',
                      help='whether use large imag scale',
                      action='store_true')
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')
  parser.add_argument('--parallel_type', dest='parallel_type',
                      help='which part of model to parallel, 0: all, 1: model before roi pooling',
                      default=0, type=int)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load network',
                      default=1, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load network',
                      default=10021, type=int)
  parser.add_argument('--vis', dest='vis',
                      help='visualization mode',
                      action='store_true')
  args = parser.parse_args()
  return args

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

if __name__ == '__main__':

  args = parse_args()

  print('Called with args:')
  print(args)

  align_layer = AlignLayer(40)
  
  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  np.random.seed(cfg.RNG_SEED)
  if args.dataset == "pascal_voc":
      args.imdb_name = "voc_2007_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "pascal_voc_0712":
      args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "coco":
      args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
      args.imdbval_name = "coco_2014_minival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "imagenet":
      args.imdb_name = "imagenet_train"
      args.imdbval_name = "imagenet_val"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "vg":
      args.imdb_name = "vg_150-50-50_minitrain"
      args.imdbval_name = "vg_150-50-50_minival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "kitti":
      args.imdb_name = "kitti_train"
      args.imdbval_name = "kitti_val"
      args.imdbtest_name = "kitti_test"
      args.set_cfgs = ['FPN_ANCHOR_SCALES', '[32, 64, 128, 256, 512]', 'FPN_FEAT_STRIDES', '[4, 8, 16, 32, 64]', 'MAX_NUM_GT_BOXES', '30']
  args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)

  cfg.TRAIN.USE_FLIPPED = False
  imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdbtest_name, False)
  imdb.competition_mode(on=True)

  print('{:d} roidb entries'.format(len(roidb)))

  input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
  if not os.path.exists(input_dir):
    raise Exception('There is no input directory for loading network from ' + input_dir)
  load_name = os.path.join(input_dir,
    'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
  kitti_classes = np.asarray(['__background__', 'Car'])
  # initilize the network here.
  if args.net == 'vgg16':
    fasterRCNN = vgg16(imdb.classes, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res101':
    fasterRCNN = resnet(imdb.classes, 101, pretrained=False, class_agnostic=False)
  elif args.net == 'res50':
    fasterRCNN = resnet(imdb.classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res152':
    fasterRCNN = resnet(imdb.classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
  else:
    print("network is not defined")
    pdb.set_trace()

  fasterRCNN.create_architecture()

  print("load checkpoint %s" % (load_name))
  checkpoint = torch.load(load_name)
  fasterRCNN.load_state_dict(checkpoint['model'])
  if 'pooling_mode' in checkpoint.keys():
    cfg.POOLING_MODE = checkpoint['pooling_mode']


  print('load model successfully!')
    # initilize the tensor holder here.
  im_left_data = torch.FloatTensor(1)
  im_right_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes_left = torch.FloatTensor(1)
  gt_boxes_right = torch.FloatTensor(1)
  gt_boxes_merge = torch.FloatTensor(1)
  gt_dis = torch.FloatTensor(1)
  gt_dim_orien = torch.FloatTensor(1)
  gt_kpts = torch.FloatTensor(1)
  # ship to cuda
  if args.cuda:
    im_left_data = im_left_data.cuda()
    im_right_data = im_right_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes_left = gt_boxes_left.cuda()
    gt_boxes_right = gt_boxes_right.cuda()
    gt_boxes_merge = gt_boxes_merge.cuda()
    gt_dis = gt_dis.cuda()
    gt_dim_orien = gt_dim_orien.cuda()
    gt_kpts = gt_kpts.cuda() 

  # make variable 
  im_left_data = Variable(im_left_data, volatile=True)
  im_right_data = Variable(im_right_data, volatile=True)
  im_info = Variable(im_info, volatile=True)
  num_boxes = Variable(num_boxes, volatile=True)
  gt_boxes_left = Variable(gt_boxes_left, volatile=True)
  gt_boxes_right = Variable(gt_boxes_right, volatile=True)
  gt_boxes_merge = Variable(gt_boxes_merge, volatile=True)
  gt_dis = Variable(gt_dis, volatile=True)
  gt_dim_orien = Variable(gt_dim_orien, volatile=True)
  gt_kpts = Variable(gt_kpts, volatile=True) 

  if args.cuda:
    cfg.CUDA = True

  if args.cuda:
    fasterRCNN.cuda()

  start = time.time()
  max_per_image = 100

  vis = True

  if vis:
    thresh = 0.01
  else:
    thresh = 0.01

  save_name = 'faster_rcnn_10'
  num_images = len(imdb.image_index)
  all_boxes_left = [[[] for _ in xrange(num_images)]
               for _ in xrange(imdb.num_classes)]
  all_boxes_right = [[[] for _ in xrange(num_images)]
               for _ in xrange(imdb.num_classes)]

  output_dir = get_output_dir(imdb, save_name)
  dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1, \
                        imdb.num_classes, training=False, normalize = False)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=0,
                            pin_memory=True)

  data_iter = iter(dataloader)

  _t = {'im_detect': time.time(), 'misc': time.time()}
  det_file = os.path.join(output_dir, 'detections.pkl')

  fasterRCNN.eval()
  empty_array = np.transpose(np.array([[],[],[],[],[]]), (1,0))
  for i in range(num_images):

    data = next(data_iter)
    im_left_data.data.resize_(data[0].size()).copy_(data[0])
    im_right_data.data.resize_(data[1].size()).copy_(data[1])
    im_info.data.resize_(data[2].size()).copy_(data[2])
    gt_boxes_left.data.resize_(data[3].size()).copy_(data[3])
    gt_boxes_right.data.resize_(data[4].size()).copy_(data[4])
    gt_boxes_merge.data.resize_(data[5].size()).copy_(data[5])
    gt_dis.data.resize_(data[6].size()).copy_(data[6])
    gt_dim_orien.data.resize_(data[7].size()).copy_(data[7])
    gt_kpts.data.resize_(data[8].size()).copy_(data[8])
    num_boxes.data.resize_(data[9].size()).copy_(data[9])
    
    det_tic = time.time()
    rois_left, cls_prob, bbox_pred, bbox_pred_dim, kpts_prob,\
    left_prob, right_prob, rpn_loss_cls, rpn_loss_box_left_right,\
    RCNN_loss_cls, RCNN_loss_bbox, RCNN_loss_dim_orien, RCNN_loss_kpts, rois_label =\
    fasterRCNN(im_left_data, im_info, gt_boxes_left, gt_dis, gt_dim_orien, gt_kpts, num_boxes)
    
    scores = cls_prob.data
    boxes_left = rois_left.data[:, :, 1:5]

    bbox_pred = bbox_pred.data
    box_delta_left = bbox_pred.new(bbox_pred.size()[1], 4*len(kitti_classes)).zero_()

    for keep_inx in range(box_delta_left.size()[0]):
      box_delta_left[keep_inx, 0::4] = bbox_pred[0,keep_inx,0::4]
      box_delta_left[keep_inx, 1::4] = bbox_pred[0,keep_inx,1::4]
      box_delta_left[keep_inx, 2::4] = bbox_pred[0,keep_inx,2::4]
      box_delta_left[keep_inx, 3::4] = bbox_pred[0,keep_inx,3::4]

    box_delta_left = box_delta_left.view(-1,4)

    dim_orien = bbox_pred_dim.data
    dim_orien = dim_orien.view(-1,5)

    kpts_prob = kpts_prob.data
    kpts_prob = kpts_prob.view(-1,4*cfg.KPTS_GRID)
    max_prob, kpts_delta = torch.max(kpts_prob,1)

    left_prob = left_prob.data
    left_prob = left_prob.view(-1,cfg.KPTS_GRID)
    _, left_delta = torch.max(left_prob,1)

    right_prob = right_prob.data
    right_prob = right_prob.view(-1,cfg.KPTS_GRID)
    _, right_delta = torch.max(right_prob,1)

    box_delta_left = box_delta_left * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
               + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
    dim_orien = dim_orien * torch.FloatTensor(cfg.TRAIN.DIM_NORMALIZE_STDS).cuda() \
               + torch.FloatTensor(cfg.TRAIN.DIM_NORMALIZE_MEANS).cuda()

    if args.class_agnostic:
      box_delta_left = box_delta_left.view(1,-1,4)
      dim_orien = dim_orien.view(1,-1,5)
      kpts_delta = kpts_delta.view(1,-1,1)
      left_delta = left_delta.view(1,-1,1)
      right_delta = right_delta.view(1,-1,1)
      max_prob = max_prob.view(1,-1,1)
    else:
      box_delta_left = box_delta_left.view(1,-1,4*len(kitti_classes))
      dim_orien = dim_orien.view(1, -1, 5*len(kitti_classes))
      kpts_delta = kpts_delta.view(1, -1, 1)
      left_delta = left_delta.view(1, -1, 1)
      right_delta = right_delta.view(1, -1, 1)
      max_prob = max_prob.view(1, -1, 1)

    pred_boxes_left = bbox_transform_inv(boxes_left, box_delta_left, 1)
    pred_kpts, kpts_type = kpts_transform_inv(boxes_left, kpts_delta,cfg.KPTS_GRID)
    pred_left = border_transform_inv(boxes_left, left_delta,cfg.KPTS_GRID)
    pred_right = border_transform_inv(boxes_left, right_delta,cfg.KPTS_GRID)

    pred_boxes_left = clip_boxes(pred_boxes_left, im_info.data, 1)

    pred_boxes_left /= im_info[0,2].data
    pred_kpts /= im_info[0,2].data
    pred_left /= im_info[0,2].data
    pred_right /= im_info[0,2].data

    scores = scores.squeeze()
    pred_boxes_left = pred_boxes_left.squeeze()

    pred_kpts = torch.cat((pred_kpts, kpts_type, max_prob, pred_left, pred_right),2)
    pred_kpts = pred_kpts.squeeze()
    dim_orien = dim_orien.squeeze()

    det_toc = time.time()
    detect_time = det_toc - det_tic
    misc_tic = time.time()

    img_path = imdb.img_left_path_at(i)
    split_path = img_path.split('/')
    image_number = split_path[len(split_path)-1].split('.')[0]
    calib_path = img_path.replace("image_2", "calib")
    calib_path = calib_path.replace("png", "txt")
    calib = kitti_utils.read_obj_calibration(calib_path)

    im_l = cv2.imread(imdb.img_left_path_at(i))
    im2show_left = np.copy(im_l)
    im_r = cv2.imread(imdb.img_right_path_at(i))
    im2show_right = np.copy(im_r)

    im_h = im2show_left.shape[0]*2
    res = float(im_h)/70.0
    im_w = int(res*60+0.5)
    im_box = np.zeros((im_h, im_w, 3), dtype = np.uint8)
    
    objects = []
    for gt_inx in range(len(objects)):
      pos = objects[gt_inx].pos
      dim = objects[gt_inx].dim
      theta = objects[gt_inx].orientation
      f = calib.p2[0,0]
      bl = (calib.p2[0,3] - calib.p3[0,3])/f
      #print('disparity', f*bl/pos[2])
      im_box = vis_utils.vis_box_in_bev(im_box, pos, dim, theta, res=0.1, gt=True)

    for j in xrange(1, imdb.num_classes):
      inds = torch.nonzero(scores[:,j] > thresh).view(-1)
      # if there is det
      if inds.numel() > 0:
        cls_scores = scores[:,j][inds]
        _, order = torch.sort(cls_scores, 0, True)
        if args.class_agnostic:
          cls_boxes_left = pred_boxes_left[inds, :]
          cls_dim_orien = dim_orien[inds, :]
        else:
          cls_boxes_left = pred_boxes_left[inds][:, j * 4:(j + 1) * 4]
          cls_dim_orien = dim_orien[inds][:, j * 5:(j + 1) * 5]
        
        cls_kpts = pred_kpts[inds]

        cls_dets_left = torch.cat((cls_boxes_left, cls_scores.unsqueeze(1)), 1)

        cls_dets_left = cls_dets_left[order]
        cls_dim_orien = cls_dim_orien[order]
        cls_kpts = cls_kpts[order] 

        keep = nms(cls_dets_left, cfg.TEST.NMS, force_cpu= not cfg.USE_GPU_NMS)
        keep = keep.view(-1).long()
        cls_dets_left = cls_dets_left[keep]
        cls_dim_orien = cls_dim_orien[keep]
        cls_kpts = cls_kpts[keep]
        infered_kpts = infer_boundary(cls_dets_left)
        for detect_idx in range(cls_dets_left.size()[0]):
            if cls_kpts[detect_idx,4] - cls_kpts[detect_idx,3] < 0.5*(infered_kpts[detect_idx,1]-infered_kpts[detect_idx,0]):
                cls_kpts[detect_idx,3:5] = infered_kpts[detect_idx]

        if vis:
          im2show_left = vis_detections(im2show_left, kitti_classes[j], \
                          cls_dets_left.cpu().numpy(), thresh, cls_kpts.cpu().numpy())

        if j == 1: # only calculate car
          for detect_idx in range(cls_dets_left.size()[0]):
            if cls_dets_left[detect_idx, -1] > thresh:
              # solve 3d box
              f = calib.p2[0,0]
              cx, cy = calib.p2[0,2], calib.p2[1,2]
              bl = (calib.p2[0,3] - calib.p3[0,3])/f

              box_left = cls_dets_left[detect_idx,0:4].cpu().numpy()  # based on origin image
              kpts_u = cls_kpts[detect_idx,0]
              dim = cls_dim_orien[detect_idx,0:3].cpu().numpy()
              sin_alpha = cls_dim_orien[detect_idx,3]
              cos_alpha = cls_dim_orien[detect_idx,4]
              alpha = m.atan2(sin_alpha, cos_alpha)
              status, state = optimizer.solve_x_y_z_theta_from_kpt(im2show_left.shape,\
                                          calib, alpha, dim, box_left, cls_kpts[detect_idx])
              if status > 0: # not faild
                poses = im_left_data.data.new(7).zero_()

                xyz = np.array([state[0], state[1], state[2]])
                theta = state[3]
                poses[0], poses[1], poses[2], poses[3], poses[4], poses[5], poses[6] = \
                            xyz[0], xyz[1], xyz[2], float(dim[0]), float(dim[1]), float(dim[2]), theta

                #im2show_right = vis_utils.vis_single_box_in_img(im2show_right, calib, xyz-np.array([bl, 0, 0]), dim, theta)
                
                # solve disparity by dense alignment (enlarged image)
                succ, dis_final = align_layer(calib, im_info.data[0,2], im_left_data.data, im_right_data.data, \
                                        cls_dets_left[detect_idx,0:4],cls_kpts[detect_idx], status, poses)
                if succ == True:
                  # resolve pose given the disparity
                  state_rect, z = optimizer.solve_x_y_theta_from_kpt(im2show_left.shape, calib, alpha, dim, box_left, \
                                          dis_final[0], cls_kpts[detect_idx])

                  # draw bird view
                  xyz[0] = state_rect[0]
                  xyz[1] = state_rect[1]
                  xyz[2] = z
                  theta = state_rect[2]
                  im_box = vis_utils.vis_box_in_bev(im_box, xyz, dim, theta, res=0.1)
                  im2show_left, box32 = vis_utils.vis_single_box_in_img(im2show_left, calib, xyz, dim, theta)

                  # visualize the keypoint in bev view
                  kpts_v = 100.0
                  kpts_d = (dis_final[0])
                  kpts_pos = kitti_utils.dis2point(f, bl, cx, cy, kpts_d, (box_left[0]+box_left[2])/2, kpts_v)
                  im_box = vis_utils.vis_kpts_in_bev(im_box, kpts_pos, res=0.1)

                  #write result into txt file
                  kitti_utils.write_detection_results(args.load_dir+'/result8/', image_number, box_left,\
                                                      xyz, dim, theta, cls_dets_left[detect_idx, -1], box32, calib.t_cam2_cam0[0])

              
    misc_toc = time.time()
    nms_time = misc_toc - misc_tic
    sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s     \r'\
              .format(i + 1, num_images, detect_time, nms_time))
    sys.stdout.flush() 

    #im2show = np.concatenate((im2show_left, im2show_right), axis=0)
    #im2show = np.concatenate((im2show, im_box), axis=1)
    #cv2.imwrite('result' + str(i) + '.png' + im2show)
    #cv2.imshow('result', im2show)

    k = cv2.waitKey(1)
    if k == 27:    # Esc key to stop
        print('exit!')
        sys.exit()
  end = time.time()


