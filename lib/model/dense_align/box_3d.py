import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils import kitti_utils
import math as m

class Box3d(nn.Module): 
    def __init__(self, poses):
        super(Box3d, self).__init__()
        self.T_c_o = poses[0:3]
        self.size = poses[3:6]
        self.R_c_o = torch.FloatTensor([[ m.cos(poses[6]), 0 ,m.sin(poses[6])],
                                        [ 0,         1 ,     0],
                                        [-m.sin(poses[6]), 0 ,m.cos(poses[6])]]).type_as(self.T_c_o)

        self.P_o = poses.new(8,3).zero_()
        self.P_o[0,0],self.P_o[0,1], self.P_o[0,2] = -self.size[0]/2, 0, -self.size[2]/2.0
        self.P_o[1,0],self.P_o[1,1], self.P_o[1,2] = -self.size[0]/2, 0, self.size[2]/2.0
        self.P_o[2,0],self.P_o[2,1], self.P_o[2,2] = self.size[0]/2, 0, self.size[2]/2.0         #max
        self.P_o[3,0],self.P_o[3,1], self.P_o[3,2] = self.size[0]/2, 0, -self.size[2]/2.0

        self.P_o[4,0],self.P_o[4,1], self.P_o[4,2] = -self.size[0]/2, -self.size[1], -self.size[2]/2.0 # min
        self.P_o[5,0],self.P_o[5,1], self.P_o[5,2] = -self.size[0]/2, -self.size[1], self.size[2]/2.0
        self.P_o[6,0],self.P_o[6,1], self.P_o[6,2] = self.size[0]/2, -self.size[1], self.size[2]/2.0
        self.P_o[7,0],self.P_o[7,1], self.P_o[7,2] = self.size[0]/2, -self.size[1], -self.size[2]/2.0

        P_c = poses.new(8,3).zero_()
        for i in range(8):
            P_c[i] = torch.mm(self.R_c_o, self.P_o[i].unsqueeze(1)).squeeze(1) + self.T_c_o

        def creatPlane(p1, p2, p3):
            arrow1 = p2 - p1
            arrow2 = p3 - p1
            normal = torch.cross(arrow1, arrow2)
            plane = p1.new((4)).zero_()
            plane[0] = normal[0]
            plane[1] = normal[1]
            plane[2] = normal[2]
            plane[3] = -normal[0] * p1[0] - normal[1] * p1[1] - normal[2] * p1[2]
            return plane

        self.planes_c = poses.new(6,4).zero_()
        self.planes_c[0] = creatPlane(P_c[0], P_c[3], P_c[4])  #front 0 
        self.planes_c[1] = creatPlane(P_c[2], P_c[3], P_c[6])  #right 1
        self.planes_c[2] = creatPlane(P_c[1], P_c[2], P_c[5])  #back 2
        self.planes_c[3] = creatPlane(P_c[0], P_c[1], P_c[4])  #left 3
        self.planes_c[4] = creatPlane(P_c[0], P_c[1], P_c[2])  #botom 4
        self.planes_c[5] = creatPlane(P_c[4], P_c[5], P_c[6])  #top 5

        # compute the nearest vertex
        self.nearest_dist = 100000000
        for i in range(P_c.size()[0]):
            if torch.norm(P_c[i]) < self.nearest_dist:
                self.nearest_dist = torch.norm(P_c[i])
                self.nearest_vertex = i  # find the nearest vertex with camera canter

    def mask_out_box(self, valid_insec, insection_c):
        DOUBLE_EPS = 0.01
        R_c_o_t = self.R_c_o.permute(1,0)
        insection_c = insection_c[:,:,0:3] - self.T_c_o
        insection_o = insection_c.new(insection_c.size()).zero_()
        insection_o[:,:,0] = R_c_o_t[0,0]*insection_c[:,:,0] + R_c_o_t[0,1]*insection_c[:,:,1] + R_c_o_t[0,2]*insection_c[:,:,2]
        insection_o[:,:,1] = R_c_o_t[1,0]*insection_c[:,:,0] + R_c_o_t[1,1]*insection_c[:,:,1] + R_c_o_t[1,2]*insection_c[:,:,2]
        insection_o[:,:,2] = R_c_o_t[2,0]*insection_c[:,:,0] + R_c_o_t[2,1]*insection_c[:,:,1] + R_c_o_t[2,2]*insection_c[:,:,2]
        
        mask = ((insection_o[:,:,0] >= self.P_o[4,0] - DOUBLE_EPS) &\
                (insection_o[:,:,1] >= self.P_o[4,1] - DOUBLE_EPS) &\
                (insection_o[:,:,2] >= self.P_o[4,2] - DOUBLE_EPS) &\
                (insection_o[:,:,0] <= self.P_o[2,0] + DOUBLE_EPS) &\
                (insection_o[:,:,1] <= self.P_o[2,1] + DOUBLE_EPS) &\
                (insection_o[:,:,2] <= self.P_o[2,2] + DOUBLE_EPS)).type_as(insection_o)
        #print('valid_insec',valid_insec[valid_insec[:,:,3]==0])
        #print('insection_o',insection_o[valid_insec[:,:,3]==0])
        valid_insec[:,:,0][valid_insec[:,:,3]==0] = insection_c[:,:,0][valid_insec[:,:,3]==0]
        valid_insec[:,:,1][valid_insec[:,:,3]==0] = insection_c[:,:,1][valid_insec[:,:,3]==0]
        valid_insec[:,:,2][valid_insec[:,:,3]==0] = insection_c[:,:,2][valid_insec[:,:,3]==0]
        valid_insec[:,:,3][valid_insec[:,:,3]==0] = mask[valid_insec[:,:,3]==0]

        return valid_insec

    def BoxRayInsec(self, pt2):

        plane_group = torch.IntTensor([[0, 3, 4],
                                [2, 3, 4],
                                [1, 2, 4],
                                [0, 1, 4],

                                [0, 3, 5],
                                [2, 3, 5],
                                [1, 2, 5],
                                [0, 1, 5]])
        homo_pt3 = torch.cat((pt2, torch.ones_like(pt2[:,:,0]).unsqueeze(2)),2)
        valid_insec = homo_pt3.new(homo_pt3.size()[0],homo_pt3.size()[1], 4).zero_() # x_o, y_o, z_o, mask
        for i in range(3):
            plane = self.planes_c[plane_group[self.nearest_vertex,i]]
            # get insection, t is a scalar
            t = homo_pt3[:,:,0]*plane[0] +  homo_pt3[:,:,1]*plane[1] + homo_pt3[:,:,2]*plane[2]
            t = -t.reciprocal()*plane[3]
            insection_c = homo_pt3 * t.unsqueeze(2)
            valid_insec = self.mask_out_box(valid_insec, insection_c)
        return valid_insec