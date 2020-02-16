
"""Detection visualizing.
This file helps visualize 3D detection result in 2D image format 
"""

import csv
import time
import argparse
import os
import sys
import numpy as np
import os.path
import cv2
import math as m
import torch
from model.utils import kitti_utils

def scale_to_255(a, min, max, dtype=np.uint8):
    """ Scales an array of values from specified min, max range to 0-255
        Optionally specify the data type of the output (default is uint8)
    """
    return (((a - min) / float(max - min)) * 255).astype(dtype)

def Space2Image(P0, pts3):
    pts2_norm = P0.dot(pts3)
    pts2 = np.array([int(pts2_norm[0]/pts2_norm[2]), int(pts2_norm[1]/pts2_norm[2])])
    return pts2

def Space2Bev(P0, side_range=(-40, 40),
                  fwd_range=(0,70),
                  res=0.1):
    x_img = (P0[0]/res).astype(np.int32)
    y_img = (-P0[2]/res).astype(np.int32)
    
    x_img -= int(np.floor(side_range[0]/res))
    y_img += int(np.floor(fwd_range[1]/res)) - 1

    return np.array([x_img, y_img])

def vis_lidar_box_in_bev(pointcloud, objects = None,
                          side_range=(-40, 40),
                          fwd_range=(0,70),
                          res=0.1,
                          min_height = -2.73,
                          max_height = 1.27,
                          saveto=None):
    """ Creates an 2D birds eye view representation of the point cloud data.
        You can optionally save the image to specified filename.
    Args:
        points:     3 x N in camera 2 frame
        side_range: (-left, right) in metres
        fwd_range:  (-behind, front) in metres
    """

    x_lidar = pointcloud[0, :]
    y_lidar = pointcloud[1, :]
    z_lidar = pointcloud[2, :]

    ff = np.logical_and((z_lidar > fwd_range[0]), (z_lidar < fwd_range[1]))
    ss = np.logical_and((x_lidar > side_range[0]), (x_lidar < side_range[1]))
    indices = np.argwhere(np.logical_and(ff,ss)).flatten()

    x_img = (x_lidar[indices]/res).astype(np.int32) 
    y_img = (-z_lidar[indices]/res).astype(np.int32)
    
    x_img -= int(np.floor(side_range[0]/res))
    y_img += int(np.floor(fwd_range[1]/res)) - 1

    pixel_values = np.clip(a = y_lidar[indices],
                           a_min=min_height,
                           a_max=max_height)

    pixel_values  = scale_to_255(pixel_values, min=min_height, max=max_height)

    x_max = int((side_range[1] - side_range[0])/res)
    y_max = int((fwd_range[1] - fwd_range[0])/res)
    im = np.zeros([y_max, x_max], dtype=np.uint8)
    im[y_img, x_img] = pixel_values

    im_rgb = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

    for i in range(len(objects)):
        pts3_c_o = []
        pts2_c_o = []
        pts3_c_o.append(objects[i].pos + objects[i].R.dot([-objects[i].dim[0], 0, -objects[i].dim[2]])/2.0)
        pts3_c_o.append(objects[i].pos + objects[i].R.dot([-objects[i].dim[0], 0, objects[i].dim[2]])/2.0) #-x z
        pts3_c_o.append(objects[i].pos + objects[i].R.dot([objects[i].dim[0], 0, objects[i].dim[2]])/2.0) # x, z
        pts3_c_o.append(objects[i].pos + objects[i].R.dot([objects[i].dim[0], 0, -objects[i].dim[2]])/2.0)

        pts3_c_o.append(objects[i].pos + objects[i].R.dot([0, 0, objects[i].dim[2]*2/3]))

        pts2_bev = []
        for index in range(5):
            pts2_bev.append(Space2Bev(pts3_c_o[index]))
        
        lineColor3d = (0, 255, 0)
        cv2.line(im_rgb,  (pts2_bev[0][0], pts2_bev[0][1]), (pts2_bev[1][0], pts2_bev[1][1]), lineColor3d, 1)
        cv2.line(im_rgb,  (pts2_bev[1][0], pts2_bev[1][1]), (pts2_bev[2][0], pts2_bev[2][1]), lineColor3d, 1)
        cv2.line(im_rgb,  (pts2_bev[2][0], pts2_bev[2][1]), (pts2_bev[3][0], pts2_bev[3][1]), lineColor3d, 1)
        cv2.line(im_rgb,  (pts2_bev[0][0], pts2_bev[0][1]), (pts2_bev[3][0], pts2_bev[3][1]), lineColor3d, 1)

        cv2.line(im_rgb,  (pts2_bev[1][0], pts2_bev[1][1]), (pts2_bev[4][0], pts2_bev[4][1]), lineColor3d, 1)
        cv2.line(im_rgb,  (pts2_bev[2][0], pts2_bev[2][1]), (pts2_bev[4][0], pts2_bev[4][1]), lineColor3d, 1)

    return im_rgb

def vis_box_in_bev(im_box, pos, dim, orien, res=0.1,gt=False,
                   side_range=(-30, 30),
                   fwd_range=(0,70),
                   min_height = -2.73,
                   max_height = 1.27,
                   saveto=None):

    R = kitti_utils.E2R(orien,0,0)
    pts3_c_o = []
    pts2_c_o = []
    pts3_c_o.append(pos + R.dot([-dim[0], 0, -dim[2]])/2.0)
    pts3_c_o.append(pos + R.dot([-dim[0], 0, dim[2]])/2.0) #-x z
    pts3_c_o.append(pos + R.dot([dim[0], 0, dim[2]])/2.0) # x, z
    pts3_c_o.append(pos + R.dot([dim[0], 0, -dim[2]])/2.0)

    pts3_c_o.append(pos + R.dot([0, 0, dim[2]*2/3]))
    
    pts2_bev = []
    for index in range(5):
        pts2_bev.append(Space2Bev(pts3_c_o[index], side_range=(-30, 30),
                                  fwd_range=(0,70), res=res)) 

    if gt is False:
        lineColor3d = (0, 255, 0)
    else:
        lineColor3d = (0, 0, 255)

    cv2.line(im_box, (pts2_bev[0][0], pts2_bev[0][1]), (pts2_bev[1][0], pts2_bev[1][1]), lineColor3d, 1)
    cv2.line(im_box, (pts2_bev[1][0], pts2_bev[1][1]), (pts2_bev[2][0], pts2_bev[2][1]), lineColor3d, 1)
    cv2.line(im_box, (pts2_bev[2][0], pts2_bev[2][1]), (pts2_bev[3][0], pts2_bev[3][1]), lineColor3d, 1)
    cv2.line(im_box, (pts2_bev[0][0], pts2_bev[0][1]), (pts2_bev[3][0], pts2_bev[3][1]), lineColor3d, 1)

    cv2.line(im_box, (pts2_bev[1][0], pts2_bev[1][1]), (pts2_bev[4][0], pts2_bev[4][1]), lineColor3d, 1)
    cv2.line(im_box, (pts2_bev[2][0], pts2_bev[2][1]), (pts2_bev[4][0], pts2_bev[4][1]), lineColor3d, 1)

    return im_box

def vis_kpts_in_bev(im_box, kpts_pos, res, 
                   side_range=(-30, 30),
                   fwd_range=(0,70)):

    kpts_bev = Space2Bev(kpts_pos, side_range=(-30,30),
                                   fwd_range=(0,70),
                                   res=res)

    cv2.circle(im_box, (kpts_bev[0],kpts_bev[1]),3,(0,0,255),-1)

    return im_box

def vis_box_in_img(img, objects, calib):

    for i in range(len(objects)):
        pts3_c_o = []
        pts2_c_o = []
        # 2D box
        lineColor3d = (0, 0, 255)
        '''
        cv2.line(img,  (objects[i].box[0], objects[i].box[1]), (objects[i].box[2], objects[i].box[1]), lineColor3d, 1)
        cv2.line(img,  (objects[i].box[0], objects[i].box[3]), (objects[i].box[2], objects[i].box[3]), lineColor3d, 1)
        cv2.line(img,  (objects[i].box[0], objects[i].box[1]), (objects[i].box[0], objects[i].box[3]), lineColor3d, 1)
        cv2.line(img,  (objects[i].box[2], objects[i].box[1]), (objects[i].box[2], objects[i].box[3]), lineColor3d, 1)
        '''
        # draw visible area
        lineColor3d = (255, 0, 0)
        cv2.line(img,  (objects[i].visible_left, objects[i].box[3]), (objects[i].visible_right, objects[i].box[3]), lineColor3d, 3)

        pts3_c_o.append(objects[i].pos + objects[i].R.dot([-objects[i].dim[0], 0, -objects[i].dim[2]])/2.0)
        pts3_c_o.append(objects[i].pos + objects[i].R.dot([-objects[i].dim[0], 0, objects[i].dim[2]])/2.0) #-x z
        pts3_c_o.append(objects[i].pos + objects[i].R.dot([objects[i].dim[0], 0, objects[i].dim[2]])/2.0) # x, z
        pts3_c_o.append(objects[i].pos + objects[i].R.dot([objects[i].dim[0], 0, -objects[i].dim[2]])/2.0)

        pts3_c_o.append(objects[i].pos + objects[i].R.dot([-objects[i].dim[0], -objects[i].dim[1]*2, -objects[i].dim[2]])/2.0)
        pts3_c_o.append(objects[i].pos + objects[i].R.dot([-objects[i].dim[0], -objects[i].dim[1]*2, objects[i].dim[2]])/2.0) #-x z
        pts3_c_o.append(objects[i].pos + objects[i].R.dot([objects[i].dim[0], -objects[i].dim[1]*2, objects[i].dim[2]])/2.0) # x, z
        pts3_c_o.append(objects[i].pos + objects[i].R.dot([objects[i].dim[0], -objects[i].dim[1]*2, -objects[i].dim[2]])/2.0)
        depth_invalid = False
        for i in range(8):
            pts2_c_o.append(Space2Image(calib.p2[:,0:3], pts3_c_o[i]))
            if(pts3_c_o[i][2] < 0):
                depth_invalid = True
            if(depth_invalid):
                continue

        lineColor3d = (0, 255, 0)
        cv2.line(img,  (pts2_c_o[0][0], pts2_c_o[0][1]), (pts2_c_o[1][0], pts2_c_o[1][1]), lineColor3d, 1)
        cv2.line(img,  (pts2_c_o[1][0], pts2_c_o[1][1]), (pts2_c_o[2][0], pts2_c_o[2][1]), lineColor3d, 1)
        cv2.line(img,  (pts2_c_o[2][0], pts2_c_o[2][1]), (pts2_c_o[3][0], pts2_c_o[3][1]), lineColor3d, 1)
        cv2.line(img,  (pts2_c_o[0][0], pts2_c_o[0][1]), (pts2_c_o[3][0], pts2_c_o[3][1]), lineColor3d, 1)

        cv2.line(img,  (pts2_c_o[4][0], pts2_c_o[4][1]), (pts2_c_o[5][0], pts2_c_o[5][1]), lineColor3d, 1)
        cv2.line(img,  (pts2_c_o[5][0], pts2_c_o[5][1]), (pts2_c_o[6][0], pts2_c_o[6][1]), lineColor3d, 1)
        cv2.line(img,  (pts2_c_o[6][0], pts2_c_o[6][1]), (pts2_c_o[7][0], pts2_c_o[7][1]), lineColor3d, 1)
        cv2.line(img,  (pts2_c_o[7][0], pts2_c_o[7][1]), (pts2_c_o[4][0], pts2_c_o[4][1]), lineColor3d, 1)

        cv2.line(img,  (pts2_c_o[4][0], pts2_c_o[4][1]), (pts2_c_o[0][0], pts2_c_o[0][1]), lineColor3d, 1)
        cv2.line(img,  (pts2_c_o[5][0], pts2_c_o[5][1]), (pts2_c_o[1][0], pts2_c_o[1][1]), lineColor3d, 1)
        cv2.line(img,  (pts2_c_o[6][0], pts2_c_o[6][1]), (pts2_c_o[2][0], pts2_c_o[2][1]), lineColor3d, 1)
        cv2.line(img,  (pts2_c_o[7][0], pts2_c_o[7][1]), (pts2_c_o[3][0], pts2_c_o[3][1]), lineColor3d, 1)

    return img

def vis_single_box_in_img(img, calib, pos, dim, theta):

    pts3_c_o = []
    pts2_c_o = []
    # 2D box
    R = kitti_utils.E2R(theta,0,0)
    pts3_c_o.append(pos + R.dot([-dim[0], 0, -dim[2]])/2.0)
    pts3_c_o.append(pos + R.dot([-dim[0], 0, dim[2]])/2.0) #-x z
    pts3_c_o.append(pos + R.dot([dim[0], 0, dim[2]])/2.0) # x, z
    pts3_c_o.append(pos + R.dot([dim[0], 0, -dim[2]])/2.0)

    pts3_c_o.append(pos + R.dot([-dim[0], -dim[1]*2, -dim[2]])/2.0)
    pts3_c_o.append(pos + R.dot([-dim[0], -dim[1]*2, dim[2]])/2.0) #-x z
    pts3_c_o.append(pos + R.dot([dim[0], -dim[1]*2, dim[2]])/2.0) # x, z
    pts3_c_o.append(pos + R.dot([dim[0], -dim[1]*2, -dim[2]])/2.0)

    box2d = np.array([5000,5000,0,0])
    for i in range(8):
        pts2_c_o.append(Space2Image(calib.p2[:,0:3], pts3_c_o[i]))
        if(pts3_c_o[i][2] < 0):
            continue
        box2d[0] = min(box2d[0],pts2_c_o[i][0])
        box2d[1] = min(box2d[1],pts2_c_o[i][1])
        box2d[2] = max(box2d[2],pts2_c_o[i][0])
        box2d[3] = max(box2d[3],pts2_c_o[i][1])

    lineColor3d = (0, 255, 0)
    cv2.line(img,  (pts2_c_o[0][0], pts2_c_o[0][1]), (pts2_c_o[1][0], pts2_c_o[1][1]), lineColor3d, 1)
    cv2.line(img,  (pts2_c_o[1][0], pts2_c_o[1][1]), (pts2_c_o[2][0], pts2_c_o[2][1]), lineColor3d, 1)
    cv2.line(img,  (pts2_c_o[2][0], pts2_c_o[2][1]), (pts2_c_o[3][0], pts2_c_o[3][1]), lineColor3d, 1)
    cv2.line(img,  (pts2_c_o[0][0], pts2_c_o[0][1]), (pts2_c_o[3][0], pts2_c_o[3][1]), lineColor3d, 1)

    cv2.line(img,  (pts2_c_o[4][0], pts2_c_o[4][1]), (pts2_c_o[5][0], pts2_c_o[5][1]), lineColor3d, 1)
    cv2.line(img,  (pts2_c_o[5][0], pts2_c_o[5][1]), (pts2_c_o[6][0], pts2_c_o[6][1]), lineColor3d, 1)
    cv2.line(img,  (pts2_c_o[6][0], pts2_c_o[6][1]), (pts2_c_o[7][0], pts2_c_o[7][1]), lineColor3d, 1)
    cv2.line(img,  (pts2_c_o[7][0], pts2_c_o[7][1]), (pts2_c_o[4][0], pts2_c_o[4][1]), lineColor3d, 1)

    cv2.line(img,  (pts2_c_o[4][0], pts2_c_o[4][1]), (pts2_c_o[0][0], pts2_c_o[0][1]), (0, 255, 0), 1)
    cv2.line(img,  (pts2_c_o[5][0], pts2_c_o[5][1]), (pts2_c_o[1][0], pts2_c_o[1][1]), (0, 255, 255), 1)
    cv2.line(img,  (pts2_c_o[6][0], pts2_c_o[6][1]), (pts2_c_o[2][0], pts2_c_o[2][1]), (0, 0, 255), 1)
    cv2.line(img,  (pts2_c_o[7][0], pts2_c_o[7][1]), (pts2_c_o[3][0], pts2_c_o[3][1]), (255, 0, 0), 1)

    return img,box2d

def vis_error_in_img(im2show_left, valid_uve):
    for i in range(valid_uve.shape[0]):
        u = int(valid_uve[i,0])
        v = int(valid_uve[i,1])
        color = (0,0,int(valid_uve[i,2])*3)
        cv2.circle(im2show_left, (u,v),1,color,-1)
    
    return im2show_left
