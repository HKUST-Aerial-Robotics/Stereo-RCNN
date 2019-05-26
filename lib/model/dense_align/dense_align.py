import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
import math as m
from model.dense_align.box_3d import Box3d


def sample(calib, scale, f_h, f_w, box_left, poses, borders):
	''' Return sample pixel for the left image in the valid RoI region
		Inputs:
			box_left: rois x 4
			poses: x, y, z, w, h, l, theta (rois x 7)
			f_w, f_h: width and height of the rescaled image
			borders: left and right border of the valid RoI (rois x 2)
		Return:
			all_uvz: sample u locations, sample v locations, delta z w.r.t the object center
					 rois x pixels x 3
			all_weight: we sample same number pixels for all object RoI,
						As a result, 0 denotes unused pixels in all_weight
						1 denotes useful pixels in all_weight. (rois x pixels)
			
	'''
	f = calib.p2[0,0]*scale
	cx, cy = calib.p2[0,2]*scale, calib.p2[1,2]*scale

	u_axis = torch.Tensor(np.reshape(np.array(range(f_w)),[1,f_w,1]))   # 1 x f_w x 1
	u_axis = u_axis.expand(f_h,-1,-1).type(torch.cuda.FloatTensor).type_as(box_left) 	  # f_h x f_w x 1
	v_axis = torch.Tensor(np.reshape(np.array(range(f_h)),[f_h,1,1]))   # f_h x 1 x 1
	v_axis = v_axis.expand(-1,f_w,-1).type(torch.cuda.FloatTensor).type_as(box_left)	  # f_h x f_w x 1
	uv_axis = torch.cat((u_axis, v_axis),2) # f_h x f_w x 2
	
	all_uvzs= []
	max_pixels = 0
	for i in range(box_left.size(0)):
		box_it = box_left[i]

		width = max(int((borders[i,1]-borders[i,0])/56.0),1)
		height = max(int((box_it[3]-box_it[1])/56.0),1)
		local_uv = uv_axis[int((box_it[1]+box_it[3])/2.0+0.5):int(box_it[3]-(box_it[3]-box_it[1])*0.1+0.5):height, \
						int(borders[i,0]+0.5):int(borders[i,1]+0.5):width]

		norm_uv = local_uv.new(local_uv.size()).zero_()
		norm_uv[:,:,0] = (local_uv[:,:,0]-cx)/f
		norm_uv[:,:,1] = (local_uv[:,:,1]-cy)/f

		box_3d = Box3d(poses[i])
		valid_insec = box_3d.BoxRayInsec(norm_uv)
		valid_uvz = torch.cat((local_uv[:,:,0][valid_insec[:,:,3]==1].view(-1,1),\
								local_uv[:,:,1][valid_insec[:,:,3]==1].view(-1,1),\
								valid_insec[:,:,2][valid_insec[:,:,3]==1].view(-1,1)),1)
		
		if valid_uvz.dim() >0 and valid_uvz.size(0) > max_pixels:
			max_pixels = valid_uvz.size(0)
		
		all_uvzs.append(valid_uvz)

	all_uvz = uv_axis.new(box_left.size(0), max_pixels, 3).zero_()
	all_weight = uv_axis.new(box_left.size(0), max_pixels).zero_()
	for i in range(len(all_uvzs)):
		if all_uvzs[i].dim() > 0:
			all_uvz[i,:all_uvzs[i].size(0),:] = all_uvzs[i]
			all_weight[i,:all_uvzs[i].size(0)] = 1.0

	return all_uvz, all_weight

def align(calib, scale, im_left, im_right, \
		  box_left, keypoints, poses):
	''' Dense alignment for multiple objects

	Inputs:
		scale: H_im_left/H_origin_img
		im_left, im_right: 1 x 3 x H x W
		box_left: rois x 4 in origin image
		keypoints: rois x 5 (kpt, kpt_type, prob, left_border, right_border in origin image
		status: rois x 1: truncation, w/o; 2: normal case
		poses: rois x 7 (x, y, z, w, h, l, theta)
	Return:
		solve_status: 1 denotes success, 0 denotes faild due to no valid pixel (rois x 1)
		best_dis: aligned disparity in origin image (rois x 1)

	'''
	scale = scale * 2
	im_left = F.interpolate(im_left, scale_factor=2, mode='bilinear', align_corners=False)
	im_right = F.interpolate(im_right, scale_factor=2, mode='bilinear', align_corners=False)

	f = calib.p2[0,0]*scale
	bl = (calib.p2[0,3] - calib.p3[0,3])*scale/f
	box_left = box_left*scale
	keypoints = keypoints*scale

	dis_init = f*bl/poses[:,2]
	dis_init = dis_init.cuda()

	all_uvz, all_weight = sample(calib, scale, im_left.size(2), im_left.size(3), \
								 box_left, poses, keypoints[:,3:5])

	solve_status = box_left.new(box_left.size(0)).zero_()
	if torch.sum(all_weight) == 0:
		return solve_status, dis_init

	solve_status = solve_status + 1.0
	solve_status[torch.sum(all_weight,1)==0] = 0
	
	f_h = float(im_left.size()[2])-1
	f_w = float(im_left.size()[3])-1

	grid_left = all_uvz.new(1, all_uvz.size(0), all_uvz.size(1), 2).zero_() # 1 x rois x pixels x 2
	grid_right = all_uvz.new(1, all_uvz.size(0), all_uvz.size(1), 2).zero_() # 1 x rois x pixels x 2
	
	grid_left[0,:,:,0] = (all_uvz[:,:,0] - f_w/2)/(f_w/2)
	grid_left[0,:,:,1] = (all_uvz[:,:,1] - f_h/2)/(f_h/2)
	grid_right[0,:,:,1] = (all_uvz[:,:,1] - f_h/2)/(f_h/2)
	
	all_weight = all_weight.unsqueeze(1).expand(-1, im_left.size(1), -1)   # (rois, 3L, num)

	iter_num = 50
	error_minimum = box_left.new(box_left.size(0)).zero_() +  100000000
	depth_interval = box_left.new(box_left.size(0)).zero_() + 0.5
	depth = dis_init.reciprocal()*f*bl - iter_num*depth_interval/2

	best_depth = depth
	
	for i in range(iter_num):
		dis = depth.reciprocal()*f*bl
		delta_d = (all_uvz[:,:,2]/(f*bl) + dis.reciprocal().unsqueeze(1).expand(-1,all_uvz.size(1))).reciprocal() # num calculate delta disparity for each pixel
		grid_right[0,:,:,0] = (all_uvz[:,:,0] - delta_d - f_w/2)/(f_w/2)
		error = F.grid_sample(im_left,grid_left, padding_mode='border')-\
				F.grid_sample(im_right,grid_right, padding_mode='border')  # (1L, 3L, rois, num)
		
		error = error.squeeze(0).data 					 		 # (3L, rois, num)
		error_pixel = error.permute(1,0,2).contiguous() 	 	 # (rois, 3L, num)
		
		error = error * all_weight
		error = error_pixel.view(error_pixel.size(0),-1) 		 # rois x (3 x num)
		
		error_sum = torch.sum(torch.abs(error),1) 				 # rois 
		update_inds = (error_sum < error_minimum) &\
						(dis - dis_init <= 50.0) & (dis - dis_init >= -50.0)
		
		best_depth[update_inds] = depth[update_inds]
		error_minimum[update_inds] = error_sum[update_inds]

		depth = depth+depth_interval 
	
	tune_num = 20
	tune_depth_interval = depth_interval*2/tune_num
	depth = best_depth - depth_interval
	for i in range(tune_num):

		dis = depth.reciprocal()*f*bl
		delta_d = (all_uvz[:,:,2]/(f*bl) + dis.reciprocal().unsqueeze(1).expand(-1,all_uvz.size(1))).reciprocal() # num calculate delta disparity for each pixel
		grid_right[0,:,:,0] = (all_uvz[:,:,0] - delta_d - f_w/2)/(f_w/2)
		error = F.grid_sample(im_left,grid_left, padding_mode='border')-\
				F.grid_sample(im_right,grid_right, padding_mode='border')  # (1L, 3L, rois, num)
		error = error.squeeze(0).data 					 		 # (3L, rois, num)
		error_pixel = error.permute(1,0,2).contiguous() 	 	 # (rois, 3L, num)
		error = error * all_weight
		error = error_pixel.view(error_pixel.size(0),-1) 		 # rois x (3 x num)
		
		error_sum = torch.sum(torch.abs(error),1)
		update_inds = error_sum < error_minimum
		best_depth[update_inds] = depth[update_inds]
		error_minimum[update_inds] = error_sum[update_inds]

		depth = depth+tune_depth_interval

	best_dis = f*bl/(best_depth*scale) + 0.5
	return solve_status, best_dis
  
def enumeration_depth(im_left, im_right, all_uvz, all_weight, depth_enum, fb):
	''' Given the depth enumeration, return the best depth according to photometric error

	Inputs:
		im_left, im_right: 1 x 3 x H x W
		all_uvz: sample u locations, sample v locations and delta z w.r.t the object center
				 (rois x pixels x 3)
		all_weight: 0 denotes unused pixels in all_weight
					1 denotes useful pixels in all_weight (rois x pixels)
		depth_enum: depths for all enumration and all objects (iter_num x rois)
		fb: focal_legnth (in scales image) * baseline
	Return:
		best_depth: aligned depth

	'''
	iter_num = depth_enum.size(0)
	rois_num = all_uvz.size(0)
	pixels_num = all_uvz.size(1)
	
	f_h = float(im_left.size(2))-1
	f_w = float(im_left.size(3))-1

	grid_left = all_uvz.new(1, rois_num, pixels_num, 2).zero_()                           # 1 x rois x pixels x 2
	grid_right = all_uvz.new(1, rois_num, pixels_num, 2).zero_()                          # 1 x rois x pixels x 2
	
	grid_left[0,:,:,0] = (all_uvz[:,:,0] - f_w/2)/(f_w/2)
	grid_left[0,:,:,1] = (all_uvz[:,:,1] - f_h/2)/(f_h/2)
	grid_right[0,:,:,1] = (all_uvz[:,:,1] - f_h/2)/(f_h/2)
	
	grid_left = grid_left.expand(iter_num,-1,-1,-1).contiguous().view(1,-1,pixels_num,2)  # 1 x (iter x rois) x pixels x 2
	grid_right = grid_right.expand(iter_num,-1,-1,-1).contiguous().view(1,-1,pixels_num,2)
	all_weight = all_weight.unsqueeze(1).expand(-1, im_left.size(1), -1)   	      		  # rois, 3, pixels
	all_weight = all_weight.unsqueeze(0).expand(iter_num, -1, -1, -1).contiguous()        # iter x rois x 3 x pixels
	all_weight = all_weight.view(-1, im_left.size(1), pixels_num)     	          		  # (iter x rois) x 3 x pixels

	depth_enum = depth_enum.view(-1).unsqueeze(1).expand(-1,pixels_num)  		          # (iter x rois) x pixels
	dis_enum = depth_enum.reciprocal()*fb

	local_delta_d = all_uvz[:,:,2].unsqueeze(0).expand(iter_num,-1,-1).contiguous()       # iter x rois x pixels
	local_delta_d = local_delta_d.view(-1,pixels_num)               			          # (iter x rois) x pixels
	global_delta_d = (local_delta_d/(fb) + dis_enum.reciprocal()).reciprocal()            # (iter x rois) x pixels
	
	all_u = all_uvz[:,:,0].unsqueeze(0).expand(iter_num,-1,-1).contiguous()   		      # iter x rois x pixels
	all_u = all_u.view(-1,pixels_num)               			  				          # (iter x rois) x pixels
	grid_right[0,:,:,0] = (all_u - global_delta_d - f_w/2)/(f_w/2)
	
	error = F.grid_sample(im_left,grid_left, padding_mode='border')-\
				F.grid_sample(im_right,grid_right, padding_mode='border')  		  		  # (1L, 3L, iter x rois, pixels)
	
	error = error.squeeze(0).data 					 		 							  # (3L, iter x rois, pixels)
	error = error.permute(1,0,2).contiguous() 	 	 									  # ((iter x rois), 3L, pixels)
	
	error = error * all_weight
	error = error.view(error.size(0), -1) 		 	 									  # (iter x rois) x (3 x pixels)
	error = error.view(iter_num, rois_num, -1)		                                      # iter x rois x (3 x pixels)
	
	error_sum = torch.sum(torch.abs(error),2) 				                              # iter x rois 
	_, depth_idx = torch.min(error_sum, 0)
	rois_idx = torch.Tensor(np.array(range(rois_num))).type_as(depth_idx)
	best_depth = depth_enum[:,0].contiguous().view(iter_num,-1)                           # iter x rois
	
	best_depth = best_depth[depth_idx, rois_idx]                                          # rois

	return best_depth

def align_parallel(calib, scale, im_left, im_right, \
		  			box_left, keypoints, poses):
	''' Dense alignment for multiple objects in parallel for depth enumeration

	Inputs:
		scale: H_im_left/H_origin_img
		im_left, im_right: 1 x 3 x H x W
		box_left: rois x 4 in origin image
		keypoints: rois x 5 (kpt, kpt_type, prob, left_border, right_border in origin image
		poses: rois x 7 (x, y, z, w, h, l, theta)
	Return:
		solve_status: 1 denotes success, 0 denotes faild due to no valid pixel (rois x 1)
		best_dis: aligned disparity in origin image (rois x 1)

	'''
	scale = scale * 2
	im_left = F.interpolate(im_left, scale_factor=2, mode='bilinear', align_corners=False)
	im_right = F.interpolate(im_right, scale_factor=2, mode='bilinear', align_corners=False)

	f = calib.p2[0,0]*scale
	bl = (calib.p2[0,3] - calib.p3[0,3])*scale/f
	box_left = box_left*scale
	keypoints = keypoints*scale
	poses = poses
	dis_init = box_left.new(box_left.size(0)).zero_()  # in 1/4 feature map 
	dis_init = f*bl/poses[:,2]
	dis_init = dis_init.cuda()

	# rois x pixels x 3, rois x pixels
	all_uvz, all_weight = sample(calib, scale, im_left.size(2), im_left.size(3), \
									box_left, poses, keypoints[:,3:5])

	solve_status = box_left.new(box_left.size(0)).zero_()
	if torch.sum(all_weight) == 0:
		return solve_status, dis_init

	solve_status = solve_status + 1.0
	solve_status[torch.sum(all_weight,1)==0] = 0

	# initial enumeration
	iter_num = 50
	depth_interval = 0.5
	depth_enum = box_left.new(iter_num, box_left.size(0)).zero_() # 50 x rois
	for i in range(iter_num):
		depth_enum[i] = dis_init.reciprocal()*f*bl - iter_num*depth_interval/2 + depth_interval*i

	depth_enum[depth_enum<1.5] = 1.5
	best_depth = enumeration_depth(im_left, im_right, all_uvz, all_weight, depth_enum, f*bl)
	
	# depth tune
	tune_num = 20
	tune_interval = depth_interval*2.0/tune_num
	tune_depth_enum = box_left.new(tune_num, box_left.size(0)).zero_() # 50 x rois
	for i in range(tune_num):
		tune_depth_enum[i] = best_depth - tune_num*tune_interval/2 + tune_interval*i
		
	best_depth = enumeration_depth(im_left, im_right, all_uvz, all_weight, tune_depth_enum, f*bl)
	
	best_dis = f*bl/(best_depth*scale) + 0.5

	return solve_status, best_dis

    