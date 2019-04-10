
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

def Space2Image(P0, pts3):
    pts2_norm = P0.dot(pts3)
    pts2 = np.array([int(pts2_norm[0]/pts2_norm[2]), int(pts2_norm[1]/pts2_norm[2])])
    return pts2

def Space2Bev(P0, side_range=(-20, 20),
                  fwd_range=(0,70),
                  res=0.1):
    x_img = (P0[0]/res).astype(np.int32)
    y_img = (-P0[2]/res).astype(np.int32)
    
    x_img -= int(np.floor(side_range[0]/res))
    y_img += int(np.floor(fwd_range[1]/res)) - 1

    return np.array([x_img, y_img])

def vis_lidar_in_bev(pointcloud, width=750, side_range=(-20, 20), fwd_range=(0,70),
                    min_height=-2.5, max_height=1.5):
    ''' Project pointcloud to bev image for simply visualization

        Inputs:
            pointcloud:     3 x N in camera 2 frame
        Return:
            cv color image

    '''
    res = float(fwd_range[1]-fwd_range[0])/width
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

    x_max = int((side_range[1] - side_range[0])/res)
    y_max = int((fwd_range[1] - fwd_range[0])/res)
    im = np.zeros([y_max, x_max], dtype=np.uint8)
    x_img[x_img>x_max-1] = x_max-1
    y_img[y_img>y_max-1] = y_max-1

    im[:,:] = 255
    im[y_img, x_img] = 100
    im_rgb = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    
    return im_rgb

def vis_box_in_bev(im_bev, pos, dim, orien, width=750, gt=False,
                   side_range=(-20, 20), fwd_range=(0,70),
                   min_height=-2.73, max_height=1.27):
    ''' Project 3D bounding box to bev image for simply visualization
        It should use consistent width and side/fwd range input with 
        the function: vis_lidar_in_bev

        Inputs:
            im_bev:         cv image
            pos, dim, orien: params of the 3D bounding box
        Return:
            cv color image

    '''
    res = float(fwd_range[1]-fwd_range[0])/width

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
        pts2_bev.append(Space2Bev(pts3_c_o[index], side_range=side_range,
                                  fwd_range=fwd_range, res=res)) 

    if gt is False:
        lineColor3d = (0, 200, 0)
    else:
        lineColor3d = (0, 0, 255)

    cv2.line(im_bev, (pts2_bev[0][0], pts2_bev[0][1]), (pts2_bev[1][0], pts2_bev[1][1]), lineColor3d, 2)
    cv2.line(im_bev, (pts2_bev[1][0], pts2_bev[1][1]), (pts2_bev[2][0], pts2_bev[2][1]), lineColor3d, 2)
    cv2.line(im_bev, (pts2_bev[2][0], pts2_bev[2][1]), (pts2_bev[3][0], pts2_bev[3][1]), lineColor3d, 2)
    cv2.line(im_bev, (pts2_bev[0][0], pts2_bev[0][1]), (pts2_bev[3][0], pts2_bev[3][1]), lineColor3d, 2)

    cv2.line(im_bev, (pts2_bev[1][0], pts2_bev[1][1]), (pts2_bev[4][0], pts2_bev[4][1]), lineColor3d, 2)
    cv2.line(im_bev, (pts2_bev[2][0], pts2_bev[2][1]), (pts2_bev[4][0], pts2_bev[4][1]), lineColor3d, 2)

    return im_bev

def vis_single_box_in_img(img, calib, pos, dim, theta):

    ''' Project 3D bounding box to rgb frontview for simply visualization

        Inputs:
            img:         cv image
            calib:       FrameCalibrationData
            pos, dim, orien: params of the 3D bounding box
        Return:
            cv color image

    '''

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

    for i in range(8):
        pts2_c_o.append(Space2Image(calib.p2[:,0:3], pts3_c_o[i]))
        if(pts3_c_o[i][2] < 0):
            return img

    lineColor3d = (0, 200, 0)
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