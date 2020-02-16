import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
import math as m
from model.align_layer.box_3d import Box3d


class AlignLayer(nn.Module): 
	def __init__(self, sample_size):
		super(AlignLayer, self).__init__()
		self.ss = sample_size

	def sample(self, calib, scale, f_h, f_w, box_left, status, poses, borders):
		'''
		Return sample pixel for both left and right feature map, and object jacobian for each pixel
		given evaluation disparity
		box_left: rois x 4
		status: 1: truncation, w/o pose data, simply use all the pixel
		poses: rois x 7: x, y, z, w, h, l, theta
		f_w, f_h: width and height of the rescaled image or feature map
		borders: left and right border rois x 2
		'''
		f = calib.p2[0,0]*scale
		cx, cy = calib.p2[0,2]*scale, calib.p2[1,2]*scale

		u_axis = torch.Tensor(np.reshape(np.array(range(f_w)),[1,f_w,1]))   # 1 x f_w x 1
		u_axis = u_axis.repeat(f_h,1,1).type(torch.cuda.FloatTensor).type_as(box_left) 	  # f_h x f_w x 1
		v_axis = torch.Tensor(np.reshape(np.array(range(f_h)),[f_h,1,1]))   # f_h x 1 x 1
		v_axis = v_axis.repeat(1,f_w,1).type(torch.cuda.FloatTensor).type_as(box_left)	  # f_h x f_w x 1
		uv_axis = torch.cat((u_axis, v_axis),2) # f_h x f_w x 2

		for i in range(box_left.size()[0]):
			box_it = box_left[i]

			if status == 1: # truncation
				local_uv = uv_axis[int((box_it[1]+box_it[3])/2.0+0.5):int(box_it[3]-(box_it[3]-box_it[1])*0.1+0.5), \
								int(box_it[0]+0.5):int(box_it[2]+0.5)] # object roi which contains the uv coordinates
			else:
				local_uv = uv_axis[int((box_it[1]+box_it[3])/2.0+0.5):int(box_it[3]-(box_it[3]-box_it[1])*0.1+0.5), \
								int(borders[i,0]+0.5):int(borders[i,1]+0.5)]

			norm_uv = local_uv.new(local_uv.size()).zero_()
			norm_uv[:,:,0] = (local_uv[:,:,0]-cx)/f
			norm_uv[:,:,1] = (local_uv[:,:,1]-cy)/f
			
			# use guassion weight
			std = 0.2
			sqrt_2pi = 2.5066
			weight = (local_uv[:,:,0] - (borders[i,0]+borders[i,1])/2.0)/(borders[i,1]-borders[i,0])
			weight = torch.exp(-(weight*weight)/(2*std*std))/(std*sqrt_2pi)

			if status == 1: # truncation
				valid_u = local_uv[:,:,0].contiguous().view(-1) # num
				valid_v = local_uv[:,:,1].contiguous().view(-1) # num
				valid_delta_z = valid_u.new(valid_u.size()[0]).zero_()
			else:
				box_3d = Box3d(poses[i])
				valid_insec = box_3d.BoxRayInsec(norm_uv)
				valid_u = local_uv[:,:,0][valid_insec[:,:,3]==1]
				valid_v = local_uv[:,:,1][valid_insec[:,:,3]==1]
				valid_delta_z = valid_insec[:,:,2][valid_insec[:,:,3]==1]
				weight = weight[valid_insec[:,:,3]==1]
			
		return valid_u, valid_v, valid_delta_z, weight
	
	def gaussian_blur(self, features_left, features_right):
		# Set these to whatever you want for your gaussian filter
		kernel_size = 7
		sigma = 0.5

		# Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
		x_cord = torch.arange(kernel_size)
		x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
		y_grid = x_grid.t()
		xy_grid = torch.stack([x_grid, y_grid], dim=-1)

		mean = (kernel_size - 1)/2.
		variance = sigma**2.

		# Calculate the 2-dimensional gaussian kernel which is
		# the product of two gaussian distributions for two different
		# variables (in this case called x and y)
		gaussian_kernel = (1./(2.*m.pi*variance)) *\
						torch.exp(
							-torch.sum((xy_grid - mean)**2., dim=-1) /\
							(2*variance)
						)
		# Make sure sum of values in gaussian kernel equals 1.
		gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
		#print('gaussian_kernel',gaussian_kernel) 0.0838  0.6193  0.0838

		# Reshape to 2d depthwise convolutional weight
		gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
		gaussian_kernel = gaussian_kernel.repeat(3, 1, 1, 1).cuda()

		gaussian_filter = nn.Conv2d(in_channels=3, out_channels=3,
									kernel_size=kernel_size, padding=3, groups=3, bias=False)

		gaussian_filter.weight.data = gaussian_kernel
		gaussian_filter.weight.requires_grad = False

		features_left = gaussian_filter(Variable(features_left))
		features_right = gaussian_filter(Variable(features_right))
		return features_left, features_right

	def forward(self, calib, scale, features_left, features_right, \
					  box_left, keypoints, status, poses):
		'''
		scale: H_feature/H_origin_img
		features_left, features_right: 1 x 3 x H x W
		box_left, box_right: 4 (origin image)
		keypoints: 5 (kpt, kpt_type, prob, left_border, right_border) (origin image)
		status: 1: truncation, w/o pose data; 2: normal case, with pose data
		poses: 7 (x, y, z, w, h, l, theta)
		'''
		scale = scale * 2
		features_left = F.upsample(features_left, scale_factor=2, mode='bilinear')
		features_right = F.upsample(features_right, scale_factor=2, mode='bilinear')
		f = calib.p2[0,0]*scale
		bl = (calib.p2[0,3] - calib.p3[0,3])*scale/f
		box_left = box_left.unsqueeze(0)*scale
		keypoints = keypoints.unsqueeze(0)*scale
		poses = poses.unsqueeze(0)

		dis_init = box_left.new(box_left.size()[0]).zero_()  # in 1/4 feature map 
		dis_init[0] = f*bl/poses[0,2]
		dis_init = dis_init.cuda()

		valid_u, valid_v, valid_delta_z, pixel_weight = self.sample(calib, scale, features_left.size()[2], features_left.size()[3], \
										box_left, status, poses, keypoints[:,3:5])
		
		if len(valid_u.size()) == 0:
			return False, 0
		
		f_h = float(features_left.size()[2])-1
		f_w = float(features_left.size()[3])-1

		valid_uv = torch.cat((valid_u.unsqueeze(1), valid_v.unsqueeze(1)),1) # num x 2
		grid_left = valid_uv.new(1, 1, valid_uv.size()[0], 2).zero_() # 1 x 1 x num x 2
		grid_right = valid_uv.new(1, 1, valid_uv.size()[0], 2).zero_() # 1 x 1 x num x 2
		
		grid_left[0,0,:,0] = (valid_uv[:,0] - f_w/2)/(f_w/2)
		grid_left[0,0,:,1] = (valid_uv[:,1] - f_h/2)/(f_h/2)
		grid_right[0,0,:,1] = (valid_uv[:,1] - f_h/2)/(f_h/2)
		
		pixel_weight = pixel_weight.repeat(features_left.size()[1], box_left.size()[0], 1)       # (3L, 1, num)
		pixel_weight = pixel_weight.permute(1,0,2).contiguous() 
		pixel_weight = pixel_weight.view(box_left.size()[0], -1)

		iter_num = 50
		depth_interval = 0.5
		error_minimum = 100000000
		depth = dis_init.reciprocal()*f*bl - iter_num*depth_interval/2
		best_depth = depth
		if status == 1: # for truncate case, we simply enum 1.5 to 7.5
			depth = dis_init.new(1).zero_()
			depth[0] = 1.5
			depth_interval = (-depth+7.5)/iter_num
			#print('dis_init', dis_init[0], 'depth init',f*bl/dis_init[0])
		
		for i in range(iter_num):
			dis = depth.reciprocal()*f*bl
			delta_d = (valid_delta_z/(f*bl) + dis.reciprocal()).reciprocal() # num calculate delta disparity for each pixel
			grid_right[0,0,:,0] = (valid_uv[:,0] - delta_d - f_w/2)/(f_w/2)
			error = F.grid_sample(features_left,grid_left, padding_mode='border')-\
					F.grid_sample(features_right,grid_right, padding_mode='border')  # (1L, 3L, 1, num)
			error = error.squeeze(0).data 					 		 # (3L, 1, num)
			error_pixel = error.permute(1,0,2).contiguous() 	 	 # (1, 3L, num)
			error = error_pixel.view(error_pixel.size()[0],-1) 		 # 1 x (3 x num)
			
			# use a Tanh function to limit the intensity error
			#error = torch.tanh(error/6.0)*6.0
			error_sum = torch.sum(torch.abs(error),1)[0]
			'''
			diff_num = error[torch.abs(error)>5].size()
			if len(diff_num) == 0:
				error_sum = 0
			else:
				error_sum = diff_num[0]
			'''
			if error_sum < error_minimum and dis[0] - dis_init[0] <= 50 and dis[0] - dis_init[0] >= -50:
				best_depth = depth
				error_minimum = error_sum
			#print('init dis',dis[0]/scale, 'depth', depth[0], 'error', error_sum)

			depth = depth+depth_interval 
		
		tune_num = 20
		tune_depth_interval = depth_interval*2/tune_num
		depth = best_depth - depth_interval
		for i in range(tune_num):

			dis = depth.reciprocal()*f*bl
			delta_d = (valid_delta_z/(f*bl) + dis.reciprocal()).reciprocal() # num calculate delta disparity for each pixel
			grid_right[0,0,:,0] = (valid_uv[:,0] - delta_d - f_w/2)/(f_w/2)
			error = F.grid_sample(features_left,grid_left, padding_mode='border')-\
					F.grid_sample(features_right,grid_right, padding_mode='border')  # (1L, 3L, 1, num)
			error = error.squeeze(0).data 					 		 # (3L, 1, num)
			error_pixel = error.permute(1,0,2).contiguous() 	 	 # (1, 3L, num)
			error = error_pixel.view(error_pixel.size()[0],-1) 		 # 1 x (3 x num)
			#error = error * pixel_weight
			
			# use a Tanh function to limit the intensity error
			# error = torch.tanh(error/6.0)*6.0
			error_sum = torch.sum(torch.abs(error),1)[0]
			'''
			diff_num = error[torch.abs(error)>5].size()
			if len(diff_num) == 0:
				error_sum = 0
			else:
				error_sum = diff_num[0]
            '''
			if error_sum < error_minimum:
				best_depth = depth
				error_minimum = error_sum

			#print('tune dis',dis[0]/scale, 'depth', depth[0], 'error', error_sum)
			depth = depth+tune_depth_interval

		best_dis = f*bl/best_depth
		#print('final dis',best_dis[0]/scale, 'depth', best_depth[0], 'error', error_minimum)
		return True, best_dis/scale + 0.5
 
	def backward(self, top, propagare_down, bottom):
		pass 
