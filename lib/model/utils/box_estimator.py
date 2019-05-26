import numpy as np
import csv
import time
import os
import sys
import os.path
import math as m
import shutil
import cv2
from copy import copy
import scipy
from scipy.optimize import minimize
from model.utils import kitti_utils as utils
    
def BB2Viewpoint(alpha):
    '''Convert the viewpoint angle to discrete viewpoint
    '''
    alpha = alpha*180.0/m.pi
    if alpha > 360:
        alpha = alpha-360
    elif alpha < -360:
        alpha = alpha+360
    viewpoint = -1
    threshold = 4.0
    if alpha >= -90.0 - threshold and alpha <= -90.0 + threshold :
        viewpoint = 0
    elif alpha >= -180.0 + threshold and alpha <= -90.0 - threshold :
        viewpoint = 1
    elif alpha >= 180.0 - threshold or alpha <= -180.0 + threshold  :
        viewpoint = 2
    elif alpha >= 90.0 + threshold and alpha <= 180.0 - threshold :
        viewpoint = 3
    elif alpha >= 90.0 - threshold and alpha <= 90.0 + threshold :
        viewpoint = 4
    elif alpha >= 0.0 + threshold and alpha <= 90.0 - threshold :
        viewpoint = 5
    elif alpha >= 0.0 - threshold and alpha <= 0.0 + threshold :
        viewpoint = 6    
    elif alpha >= -90.0 + threshold and alpha <= 0.0 - threshold :
        viewpoint = 7
    return viewpoint

def viewpoint2vertex(view_point, w, l):
    '''Obtain the 3D vertex that corresponds to the left, right, bottom side of the 2D box

    vertex define
    
       6  ____  7
      2 /|___/|            / z(l)
       / /  / / 3         /
    5 /_/_4/ /           /----->x(w)
      |/___|/            |
    1     0              | y(h)

    viewpoint define
      3 __4__ 5
       /|___/|
    2 / /  / /
     /_/_ / / 6
    |/___|/
  1   0   7

    orientation define
          180
         ____   head
       /|___/|
  -90 / /  / /
     /_/_ / /   90
    |/___|/
       0    back   

    kpt define
          
      1  ____  2  head
       /|___/|
      / /  / /
     /_/_ / / 
    |/___|/
   0      3 back  

    viewpoint angle (alpha) define
          90
         ____   head
       /|___/|
+-180 / /  / /
     /_/_ / /  0
    |/___|/
     -90    back 

    '''
    if view_point == 0:
        left_vertex = np.array([-w, 0, -l])/2
        right_vertex = np.array([w, 0, -l])/2
        bottom_vertex = np.array([w, 0, -l])/2 
    elif view_point == 1:
        left_vertex = np.array([-w, 0, l])/2
        right_vertex = np.array([w, 0, -l])/2
        bottom_vertex = np.array([-w, 0, -l])/2  
    elif view_point == 2:
        left_vertex = np.array([-w, 0, l])/2
        right_vertex = np.array([-w, 0, -l])/2
        bottom_vertex = np.array([-w, 0, -l])/2
    elif view_point == 3:
        left_vertex = np.array([w, 0, l])/2
        right_vertex = np.array([-w, 0, -l])/2
        bottom_vertex = np.array([-w, 0, l])/2
    elif view_point == 4:
        left_vertex = np.array([w, 0, l])/2
        right_vertex = np.array([-w, 0, l])/2
        bottom_vertex = np.array([-w, 0, l])/2
    elif view_point == 5:
        left_vertex = np.array([w, 0, -l])/2
        right_vertex = np.array([-w, 0, l])/2
        bottom_vertex = np.array([w, 0, l])/2
    elif view_point == 6:
        left_vertex = np.array([w, 0, -l])/2
        right_vertex = np.array([w, 0, l])/2
        bottom_vertex = np.array([w, 0, l])/2
    else:
        left_vertex = np.array([-w, 0, -l])/2
        right_vertex = np.array([w, 0, l])/2
        bottom_vertex = np.array([w, 0, -l])/2

    return left_vertex, right_vertex, bottom_vertex

def kpt2vertex(kpt_type, w, l):
    ''' Obtain the 3D vertex that corresponds to the keypoint

    kpt define
          
      1  ____  2  head
       /|___/|
      / /  / /
     /_/_ / / 
    |/___|/
   0      3 back  
    '''
    if kpt_type == 0:
        kpt_vertex = np.array([-w, 0, -l])/2
    elif kpt_type == 1:
        kpt_vertex = np.array([-w, 0, l])/2 
    elif kpt_type == 2:
        kpt_vertex = np.array([w, 0, l])/2
    elif kpt_type == 3:
        kpt_vertex = np.array([w, 0, -l])/2
    
    return kpt_vertex

def kpt2alpha(kpt_pos, kpt_type, box):
    '''Convert keypoint to viewpoint angle approximately.
       It is only used for get discrete viewpoint, so we dont
       need the accurate viewpoint angle.
    '''
    def clamp(n, minn, maxn):
        return max(min(maxn, n), minn)

    box_width = box[2]-box[0]
    if kpt_type == 0:
        alpha = -m.pi/2 - m.asin(clamp((kpt_pos-box[0])/box_width,-1,1))   # 0 -> -90, 1 -> -180
    elif kpt_type == 1:
        alpha = m.pi - m.asin(clamp((kpt_pos-box[0])/box_width,-1,1))      # 0 -> 180, 1 -> 90
    elif kpt_type == 2:
        alpha = m.pi/2 - m.asin(clamp((kpt_pos-box[0])/box_width,-1,1))    # 0 -> 90, 1 -> 0
    elif kpt_type == 3:
        alpha = - m.asin(clamp((kpt_pos-box[0])/box_width,-1,1))           # 0 -> 0, 1 -> -90
    
    return alpha

def solve_x_y_z_theta_from_kpt(im_shape, calib, alpha, dim, box_left, box_right, kpts):
    ''' Solve initial 3D bounding box use the 2D bounding boxes, keypoints/alpha angle

        Inputs:
            alpha: viewpoint angle
            dim:   regressed object dimension in w,h,l
            box_left:
            box_right: left and right 2D box
            kpts: contain u coordinates of the perspective keypoint,
                          type of the perspective keypoint (0,1,2,3),
                          confidence (unused here),
                          u coordinates of the left borderline,
                          u coordinates of the right borderline
        Return:
            status: 0: faild, 1: normal
            x: solved object pose (x, y, z, theta)
    '''
    if kpts[4] - kpts[3] < 3 or box_left[2]-box_left[0]< 10 or box_left[3]-box_left[1]< 10:
        return 0, 0
    kpt_pos = kpts[0]
    kpt_type = int(kpts[1])
    h_max, w_max = im_shape[0], im_shape[1]
    truncate_border = 10
    w, h, l = dim[0], dim[1], dim[2] 
    ul, ur, vt, vb = box_left[0], box_left[2], box_left[1], box_left[3]
    ul_r, ur_r = box_right[0], box_right[2]

    f = calib.p2[0,0]
    cx, cy = calib.p2[0,2], calib.p2[1,2]
    bl = (calib.p2[0,3] - calib.p3[0,3])/f

    # normalize image plane
    left_u = (ul - cx)/f
    right_u = (ur - cx)/f 
    top_v = (vt - cy)/f
    bottom_v = (vb - cy)/f 
    kpt_u = (kpt_pos - cx)/f

    left_u_right = (ul_r - cx)/f
    right_u_right = (ur_r - cx)/f 
    
    if ul < 2.0*truncate_border or ur > w_max - 2.0*truncate_border:
        truncation = True
    else:
        truncation = False

    if not truncation: # in truncation case, we use alpha instead of keypoints
        alpha = kpt2alpha(kpt_pos, kpt_type, box_left)

    # get 3d vertex
    view_point = BB2Viewpoint(alpha)
    left_vertex_o, right_vertex_o, bottom_vertex_o = viewpoint2vertex(view_point, w, l)
    kpt_vertex_o = kpt2vertex(kpt_type, w, l)
    left_w = left_vertex_o[0]
    left_l = left_vertex_o[2]

    right_w = right_vertex_o[0]
    right_l = right_vertex_o[2]

    bottom_w = bottom_vertex_o[0]
    bottom_l = bottom_vertex_o[2]

    kpt_w = kpt_vertex_o[0]
    kpt_l = kpt_vertex_o[2]

    def f_kpt(states): # x, y, theta
        x = states[0]
        y = states[1]
        z = states[2]
        theta = states[3]
        
        res_ul = (x + np.cos(theta)*left_w + np.sin(theta)*left_l)/(z-np.sin(theta)*left_w+np.cos(theta)*left_l) - left_u
        res_ur = (x + np.cos(theta)*right_w + np.sin(theta)*right_l)/(z-np.sin(theta)*right_w+np.cos(theta)*right_l) - right_u
        res_uk = (x + np.cos(theta)*kpt_w + np.sin(theta)*kpt_l)/(z-np.sin(theta)*kpt_w+np.cos(theta)*kpt_l) - kpt_u
        res_uk = 2*res_uk

        res_vb = y/(z -np.sin(theta)*bottom_w + np.cos(theta)*bottom_l) - bottom_v
        res_vt = (y - h)/(z +np.sin(theta)*bottom_w -np.cos(theta)*bottom_l) - top_v
        
        res_ul_right = (x -bl + np.cos(theta)*left_w + np.sin(theta)*left_l)/(z-np.sin(theta)*left_w+np.cos(theta)*left_l) - left_u_right
        res_ur_right = (x -bl + np.cos(theta)*right_w + np.sin(theta)*right_l)/(z-np.sin(theta)*right_w+np.cos(theta)*right_l) - right_u_right
        
        res_alpha = theta - m.pi/2 + m.atan2(-x, z) - alpha
        
        if truncation:
            res_uk = 0
        else:
            res_alpha = 0
            res_ul_right = 0
            res_ur_right = 0
        
        if ul < 2.0*truncate_border:
            res_ul = 0.0
        if ul_r < 2.0*truncate_border:
            res_ul_right = 0.0
        if ur > w_max - 2.0*truncate_border:
            res_ur = 0.0
        if ur_r > w_max - 2.0*truncate_border:
            res_ur_right = 0.0
        if vt < truncate_border:
            res_vt = 0.0
        if vb > h_max - truncate_border:
            res_vb = 0.0

        return res_ul**2 + res_ur**2 + res_uk**2 + res_vb**2 + res_vt**2 + res_alpha**2 + res_ul_right**2 + res_ur_right**2

    def j_kpt(states):
        x = states[0]
        y = states[1]
        z = states[2]
        theta = states[3]
        
        res_ul = (x + np.cos(theta)*left_w + np.sin(theta)*left_l)/(z-np.sin(theta)*left_w+np.cos(theta)*left_l) - left_u
        res_ur = (x + np.cos(theta)*right_w + np.sin(theta)*right_l)/(z-np.sin(theta)*right_w+np.cos(theta)*right_l) - right_u
        res_uk = (x + np.cos(theta)*kpt_w + np.sin(theta)*kpt_l)/(z-np.sin(theta)*kpt_w+np.cos(theta)*kpt_l) - kpt_u
        res_uk = 2*res_uk

        res_vb = y/(z-np.sin(theta)*bottom_w + np.cos(theta)*bottom_l) - bottom_v
        res_vt = (y - h)/(z +np.sin(theta)*bottom_w -np.cos(theta)*bottom_l) - top_v

        res_ul_right = (x -bl + np.cos(theta)*left_w + np.sin(theta)*left_l)/(z-np.sin(theta)*left_w+np.cos(theta)*left_l) - left_u_right
        res_ur_right = (x -bl + np.cos(theta)*right_w + np.sin(theta)*right_l)/(z-np.sin(theta)*right_w+np.cos(theta)*right_l) - right_u_right
        
        res_alpha = theta - m.pi/2 + m.atan2(-x, z) - alpha
        
        if truncation:
            res_uk = 0
        else:
            res_alpha = 0
            res_ul_right = 0
            res_ur_right = 0

        if ul < 2.0*truncate_border:
            res_ul = 0.0
        if ul_r < 2.0*truncate_border:
            res_ul_right = 0.0
        if ur > w_max - 2.0*truncate_border:
            res_ur = 0.0
        if ur_r > w_max - 2.0*truncate_border:
            res_ur_right = 0.0
        if vt < truncate_border:
            res_vt = 0.0
        if vb > h_max - truncate_border:
            res_vb = 0.0

        # jacobian of left u
        dul_dx = 2.0*res_ul/(z + left_l*np.cos(theta) - left_w*np.sin(theta))
        dul_dy = 0.0
        dul_dz = -2.0*res_ul*(x + left_w*np.cos(theta) + left_l*np.sin(theta))/\
                        ((z + left_l*np.cos(theta) - left_w*np.sin(theta))**2)
        dul_dth = 2.0*res_ul*((left_l*np.cos(theta) - left_w*np.sin(theta))/(z + left_l*np.cos(theta) - left_w*np.sin(theta)) +\
                  ((left_w*np.cos(theta) + left_l*np.sin(theta))*(x + left_w*np.cos(theta) + left_l*np.sin(theta)))/((z + left_l*np.cos(theta) - left_w*np.sin(theta))**2))
        # jacobian of right u
        dur_dx = 2.0*res_ur/(z + right_l*np.cos(theta) - right_w*np.sin(theta))
        dur_dy = 0.0
        dur_dz = -2.0*res_ur*(x + right_w*np.cos(theta) + right_l*np.sin(theta))/\
                        ((z + right_l*np.cos(theta) - right_w*np.sin(theta))**2)
        dur_dth = 2.0*res_ur*((right_l*np.cos(theta) - right_w*np.sin(theta))/(z + right_l*np.cos(theta) - right_w*np.sin(theta)) +\
                  ((right_w*np.cos(theta) + right_l*np.sin(theta))*(x + right_w*np.cos(theta) + right_l*np.sin(theta)))/((z + right_l*np.cos(theta) - right_w*np.sin(theta))**2))
        # jacobian of keypoint
        duk_dx = 2.0*res_uk/(z + kpt_l*np.cos(theta) - kpt_w*np.sin(theta))
        duk_dy = 0.0
        duk_dz = -2.0*res_uk*(x + kpt_w*np.cos(theta) + kpt_l*np.sin(theta))/\
                        ((z + kpt_l*np.cos(theta) - kpt_w*np.sin(theta))**2)
        duk_dth = 2.0*res_uk*((kpt_l*np.cos(theta) - kpt_w*np.sin(theta))/(z + kpt_l*np.cos(theta) - kpt_w*np.sin(theta)) +\
                  ((kpt_w*np.cos(theta) + kpt_l*np.sin(theta))*(x + kpt_w*np.cos(theta) + kpt_l*np.sin(theta)))/((z + kpt_l*np.cos(theta) - kpt_w*np.sin(theta))**2))
        # jacobian of bottom v
        dvb_dx = 0.0
        dvb_dy = 2.0*res_vb/(z + bottom_l*np.cos(theta) - bottom_w*np.sin(theta))
        dvb_dz = -2.0*res_vb*y/((z + bottom_l*np.cos(theta) - bottom_w*np.sin(theta))**2)
        dvb_dth = 2.0*res_vb*(y*(bottom_w*np.cos(theta) + bottom_l*np.sin(theta)))/((z + bottom_l*np.cos(theta) - bottom_w*np.sin(theta))**2)
        # jacobian of top v
        dvt_dx = 0.0
        dvt_dy = 2.0*res_vt/(z - bottom_l*np.cos(theta) + bottom_w*np.sin(theta))
        dvt_dz = 2.0*res_vt*(h - y)/((z - bottom_l*np.cos(theta) + bottom_w*np.sin(theta))**2)
        dvt_dth = 2.0*res_vt*((h - y)*(bottom_w*np.cos(theta) + bottom_l*np.sin(theta)))/((z - bottom_l*np.cos(theta) + bottom_w*np.sin(theta))**2)

        # jacobian of left u of right
        dul_r_dx = 2.0*res_ul_right/(z + left_l*np.cos(theta) - left_w*np.sin(theta))
        dul_r_dy = 0.0
        dul_r_dz = -2.0*res_ul_right*(x -bl + left_w*np.cos(theta) + left_l*np.sin(theta))/\
                        ((z + left_l*np.cos(theta) - left_w*np.sin(theta))**2)
        dul_r_dth = 2.0*res_ul_right*((left_l*np.cos(theta) - left_w*np.sin(theta))/(z + left_l*np.cos(theta) - left_w*np.sin(theta)) +\
                  ((left_w*np.cos(theta) + left_l*np.sin(theta))*(x- bl + left_w*np.cos(theta) + left_l*np.sin(theta)))/((z + left_l*np.cos(theta) - left_w*np.sin(theta))**2))
        # jacobian of right u of right
        dur_r_dx = 2.0*res_ur_right/(z + right_l*np.cos(theta) - right_w*np.sin(theta))
        dur_r_dy = 0.0
        dur_r_dz = -2.0*res_ur_right*(x - bl + right_w*np.cos(theta) + right_l*np.sin(theta))/\
                        ((z + right_l*np.cos(theta) - right_w*np.sin(theta))**2)
        dur_r_dth = 2.0*res_ur_right*((right_l*np.cos(theta) - right_w*np.sin(theta))/(z + right_l*np.cos(theta) - right_w*np.sin(theta)) +\
                  ((right_w*np.cos(theta) + right_l*np.sin(theta))*(x -bl + right_w*np.cos(theta) + right_l*np.sin(theta)))/((z + right_l*np.cos(theta) - right_w*np.sin(theta))**2))
        
        # jacobian of alpha
        dalpha_dx = 2.0*res_alpha /(1.0+(-x/z)**2) * (-1.0/z)
        dalpha_dy = 0.0
        dalpha_dz = 2.0*res_alpha /(1.0+(-x/z)**2) * (x/(z*z))
        dalpha_dth = 2.0*res_alpha 

        J = scipy.array([dul_dx + dur_dx + duk_dx + dvb_dx + dvt_dx + dul_r_dx + dur_r_dx + dalpha_dx, \
                         dul_dy + dur_dy + duk_dy + dvb_dy + dvt_dy + dul_r_dy + dur_r_dy + dalpha_dy, \
                         dul_dz + dur_dz + duk_dz + dvb_dz + dvt_dz + dul_r_dz + dur_r_dz + dalpha_dz, \
                         dul_dth + dur_dth + duk_dth + dvb_dth + dvt_dth + dul_r_dth + dur_r_dth + dalpha_dth])

        return J

    disparity = (box_left[0]+box_left[2])/2 - (box_right[0]+box_right[2])/2
    init_z = f*bl/disparity
    init_x = init_z*(left_u+right_u)/2.0
    init_y = init_z*(bottom_v+top_v)/2.0 + h/2.0
    
    init_theta = alpha + m.pi/2 - m.atan2(-init_x, init_z) 
    
    res = minimize(f_kpt, [init_x,init_y,init_z, init_theta], method ='Newton-CG', jac=j_kpt, options={'disp': False}) 

    if res.x[2] > 100:
        return 0, res.x
    return 1, res.x

def solve_x_y_theta_from_kpt(im_shape, calib, alpha, dim, box_left, disparity, kpts):
    ''' 3D rectify using the aligned disparity and the 2D bounding box, keypoints/alpha angle

        Inputs:
            alpha: viewpoint angle
            dim:   regressed object dimension in w,h,l
            box_left: left 2D box
            disparity: object center disparity given by the dense alignment
            kpts: contain u coordinates of the perspective keypoint,
                          type of the perspective keypoint (0,1,2,3),
                          confidence (unused here),
                          u coordinates of the left borderline,
                          u coordinates of the right borderline
        Return:
            x: solved object pose (x, y, theta)
            z: object center depth
    '''
    kpt_pos = kpts[0]
    kpt_type = int(kpts[1])

    h_max, w_max = im_shape[0], im_shape[1]
    truncate_border = 10
    w, h, l = dim[0], dim[1], dim[2] 
    ul, ur, vt, vb = box_left[0], box_left[2], box_left[1], box_left[3]

    f = calib.p2[0,0]
    cx, cy = calib.p2[0,2], calib.p2[1,2]
    bl = (calib.p2[0,3] - calib.p3[0,3])/f

    z = f*bl/disparity

    # normalize image plane
    left_u = (ul - cx)/f
    right_u = (ur - cx)/f 
    top_v = (vt - cy)/f
    bottom_v = (vb - cy)/f 
    kpt_u = (kpt_pos - cx)/f

    if ul < 2.0*truncate_border or ur > w_max - 2.0*truncate_border:
        truncation = True
    else:
        truncation = False

    if not truncation: # in truncation case, we use alpha instead of keypoints
        alpha = kpt2alpha(kpt_pos, kpt_type, box_left)

    # get 3d vertex
    view_point = BB2Viewpoint(alpha)
    left_vertex_o, right_vertex_o, bottom_vertex_o = viewpoint2vertex(view_point, w, l)
    kpt_vertex_o = kpt2vertex(kpt_type, w, l)
    left_w = left_vertex_o[0]
    left_l = left_vertex_o[2]

    right_w = right_vertex_o[0]
    right_l = right_vertex_o[2]

    bottom_w = bottom_vertex_o[0]
    bottom_l = bottom_vertex_o[2]
    kpt_w = kpt_vertex_o[0]
    kpt_l = kpt_vertex_o[2]

    def f_rect(states): # x, y, theta
        x = states[0]
        y = states[1]
        theta = states[2]
        
        res_ul = (x + np.cos(theta)*left_w + np.sin(theta)*left_l)/(z-np.sin(theta)*left_w+np.cos(theta)*left_l) - left_u
        res_ur = (x + np.cos(theta)*right_w + np.sin(theta)*right_l)/(z-np.sin(theta)*right_w+np.cos(theta)*right_l) - right_u
        res_uk = (x + np.cos(theta)*kpt_w + np.sin(theta)*kpt_l)/(z-np.sin(theta)*kpt_w+np.cos(theta)*kpt_l) - kpt_u
        res_uk = 2*res_uk

        res_vb = y/(z -np.sin(theta)*bottom_w + np.cos(theta)*bottom_l) - bottom_v
        res_vt = (y - h)/(z +np.sin(theta)*bottom_w -np.cos(theta)*bottom_l) - top_v

        res_alpha = theta - m.pi/2 + m.atan2(-x, z) - alpha
        
        if truncation:
            res_uk = 0
        else:
            res_alpha = 0
         
        if ul < 2.0*truncate_border:
            res_ul = 0.0
        if ur > w_max - 2.0*truncate_border:
            res_ur = 0.0
        if vt < truncate_border:
            res_vt = 0.0
        if vb > h_max - truncate_border:
            res_vb = 0.0

        return res_ul**2 + res_ur**2 + res_uk**2 + res_vb**2 + res_vt**2 + res_alpha**2

    def j_rect(states):
        x = states[0]
        y = states[1]
        theta = states[2] 
        
        res_ul = (x + np.cos(theta)*left_w + np.sin(theta)*left_l)/(z-np.sin(theta)*left_w+np.cos(theta)*left_l) - left_u
        res_ur = (x + np.cos(theta)*right_w + np.sin(theta)*right_l)/(z-np.sin(theta)*right_w+np.cos(theta)*right_l) - right_u
        res_uk = (x + np.cos(theta)*kpt_w + np.sin(theta)*kpt_l)/(z-np.sin(theta)*kpt_w+np.cos(theta)*kpt_l) - kpt_u
        res_uk = 2*res_uk

        res_vb = y/(z-np.sin(theta)*bottom_w + np.cos(theta)*bottom_l) - bottom_v
        res_vt = (y - h)/(z +np.sin(theta)*bottom_w -np.cos(theta)*bottom_l) - top_v
        res_alpha = theta - m.pi/2 + m.atan2(-x, z) - alpha
        
        if truncation:
            res_uk = 0
        else:
            res_alpha = 0
        
        if ul < 2.0*truncate_border:
            res_ul = 0.0
        if ur > w_max - 2.0*truncate_border:
            res_ur = 0.0
        if vt < truncate_border:
            res_vt = 0.0
        if vb > h_max - truncate_border:
            res_vb = 0.0

        dul_dx = 2.0*res_ul/(z + left_l*np.cos(theta) - left_w*np.sin(theta))
        dul_dy = 0.0
        dul_dth = 2.0*res_ul*((left_l*np.cos(theta) - left_w*np.sin(theta))/(z + left_l*np.cos(theta) - left_w*np.sin(theta)) +\
                  ((left_w*np.cos(theta) + left_l*np.sin(theta))*(x + left_w*np.cos(theta) + left_l*np.sin(theta)))/((z + left_l*np.cos(theta) - left_w*np.sin(theta))**2))
        # jacobian of right u
        dur_dx = 2.0*res_ur/(z + right_l*np.cos(theta) - right_w*np.sin(theta))
        dur_dy = 0.0
        dur_dth = 2.0*res_ur*((right_l*np.cos(theta) - right_w*np.sin(theta))/(z + right_l*np.cos(theta) - right_w*np.sin(theta)) +\
                  ((right_w*np.cos(theta) + right_l*np.sin(theta))*(x + right_w*np.cos(theta) + right_l*np.sin(theta)))/((z + right_l*np.cos(theta) - right_w*np.sin(theta))**2))
        # jacobian of keypoint
        duk_dx = 2.0*res_uk/(z + kpt_l*np.cos(theta) - kpt_w*np.sin(theta))
        duk_dy = 0.0
        duk_dth = 2.0*res_uk*((kpt_l*np.cos(theta) - kpt_w*np.sin(theta))/(z + kpt_l*np.cos(theta) - kpt_w*np.sin(theta)) +\
                  ((kpt_w*np.cos(theta) + kpt_l*np.sin(theta))*(x + kpt_w*np.cos(theta) + kpt_l*np.sin(theta)))/((z + kpt_l*np.cos(theta) - kpt_w*np.sin(theta))**2))
        # jacobian of bottom v
        dvb_dx = 0.0
        dvb_dy = 2.0*res_vb/(z + bottom_l*np.cos(theta) - bottom_w*np.sin(theta))
        dvb_dth = 2.0*res_vb*(y*(bottom_w*np.cos(theta) + bottom_l*np.sin(theta)))/((z + bottom_l*np.cos(theta) - bottom_w*np.sin(theta))**2)
        # jacobian of top v
        dvt_dx = 0.0
        dvt_dy = 2.0*res_vt/(z - bottom_l*np.cos(theta) + bottom_w*np.sin(theta))
        dvt_dth = 2.0*res_vt*((h - y)*(bottom_w*np.cos(theta) + bottom_l*np.sin(theta)))/((z - bottom_l*np.cos(theta) + bottom_w*np.sin(theta))**2)

        dalpha_dx = 2.0*res_alpha /(1.0+(-x/z)**2) * (-1.0/z)
        dalpha_dy = 0.0
        dalpha_dth = 2.0*res_alpha

        J = scipy.array([dul_dx + dur_dx + duk_dx + dvb_dx + dvt_dx + dalpha_dx, \
                         dul_dy + dur_dy + duk_dy + dvb_dy + dvt_dy + dalpha_dy, \
                         dul_dth + dur_dth + duk_dth + dvb_dth + dvt_dth + dalpha_dth])

        return J

    init_x = z*(left_u+right_u)/2.0
    init_y = z*(bottom_v+top_v)/2.0 + h/2.0
    init_theta = alpha + m.pi/2 - m.atan2(-init_x, z) 
    
    res = minimize(f_rect, [init_x,init_y,init_theta], method ='Newton-CG', jac=j_rect, options={'disp': False})
    return res.x, z


