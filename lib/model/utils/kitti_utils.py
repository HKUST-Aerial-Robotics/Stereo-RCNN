import numpy as np
import csv
import time
import os
import sys
import os.path
import math as m
import shutil
import cv2

# process all the data in camera 2 frame, the kitti raw data is on camera 0 frame
#------------------------------------------------------ Class define ------------------------------------------------------#
class Box2d:
    def __init__(self):
        self.box = []               # left, top, right bottom in 2D image
        self.keypoints = []         # holds the u coordinates of 4 keypoints, -1 denotes the invisible one
        self.visible_left = 0       # The left side is visible (not occluded) by other object
        self.visible_right = 0      # The right side is visible (not occluded) by other object

class KittiObject:
    def __init__(self):
        self.cls = ''               # Car, Van, Truck
        self.truncate = 0           # float 0(non-truncated) - 1(totally truncated)
        self.occlusion = 0          # integer 0, 1, 2, 3
        self.alpha = 0              # viewpoint angle -pi - pi
        self.boxes = (Box2d(),\
             Box2d(), Box2d())      # Box2d list, default order: box_left, box_right, box_merge
        self.pos = []               # x, y, z in cam2 frame
        self.dim = []               # width(x), height(y), length(z)    
        self.orientation = 0        # [-pi - pi]     
        self.R = []                 # rotation matrix in cam2 frame

class FrameCalibrationData:
    '''Frame Calibration Holder
        p0-p3      Camera P matrix. Contains extrinsic 3x4    
                   and intrinsic parameters.
        r0_rect    Rectification matrix, required to transform points 3x3    
                   from velodyne to camera coordinate frame.
        tr_velodyne_to_cam0     Used to transform from velodyne to cam 3x4    
                                coordinate frame according to:
                                Point_Camera = P_cam * R0_rect *
                                                Tr_velo_to_cam *
                                                Point_Velodyne.
    '''

    def __init__(self):
        self.p0 = []
        self.p1 = []
        self.p2 = []
        self.p3 = []
        self.p2_3 = []
        self.r0_rect = []
        self.t_cam2_cam0 = []
        self.tr_velodyne_to_cam0 = []

#------------------------------------------------------ Math opreation ------------------------------------------------------#
def E2R(Ry, Rx, Rz):
    '''Combine Euler angles to the rotation matrix (right-hand)
       
        Inputs:
            Ry, Rx, Rz : rotation angles along  y, x, z axis
                         only has Ry in the KITTI dataset
        Returns:
            3 x 3 rotation matrix

    '''
    R_yaw = np.array([[ m.cos(Ry), 0 ,m.sin(Ry)],
                      [ 0,         1 ,     0],
                      [-m.sin(Ry), 0 ,m.cos(Ry)]])
    R_pitch = np.array([[1, 0, 0],
                        [0, m.cos(Rx), -m.sin(Rx)],
                        [0, m.sin(Rx), m.cos(Rx)]])
    #R_roll = np.array([[[m.cos(Rz), -m.sin(Rz), 0],
    #                    [m.sin(Rz), m.cos(Rz), 0],
    #                    [ 0,         0 ,     1]])
    return (R_pitch.dot(R_yaw))

def Space2Image(P0, pts3):
    ''' Project a 3D point to the image
        
        Inputs:
            P0 : Camera intrinsic matrix 3 x 4   
            pts3 : 4-d homogeneous coordinates
        Returns:
            image uv coordinates

    '''
    pts2_norm = P0.dot(pts3)
    pts2 = np.array([(pts2_norm[0]/pts2_norm[2]), (pts2_norm[1]/pts2_norm[2])])
    return pts2

def NormalizeVector(P):
    return np.append(P, [1])

#------------------------------------------------------ Data reading ------------------------------------------------------#

def read_obj_calibration(CALIB_PATH):
    ''' Reads in Calibration file from Kitti Dataset.
        
        Inputs:
        CALIB_PATH : Str PATH of the calibration file.
        
        Returns:
        frame_calibration_info : FrameCalibrationData
                                Contains a frame's full calibration data.
        ^ z        ^ z                                      ^ z         ^ z
        | cam2     | cam0                                   | cam3      | cam1
        |-----> x  |-----> x                                |-----> x   |-----> x

    '''
    frame_calibration_info = FrameCalibrationData()

    data_file = open(CALIB_PATH, 'r')
    data_reader = csv.reader(data_file, delimiter=' ')
    data = []

    for row in data_reader:
        data.append(row)

    data_file.close()

    p_all = []

    for i in range(4):
        p = data[i]
        p = p[1:]
        p = [float(p[i]) for i in range(len(p))]
        p = np.reshape(p, (3, 4))
        p_all.append(p)

    # based on camera 0
    frame_calibration_info.p0 = p_all[0]
    frame_calibration_info.p1 = p_all[1]
    frame_calibration_info.p2 = p_all[2]
    frame_calibration_info.p3 = p_all[3]
    
    # based on camera 2
    frame_calibration_info.p2_2 = np.copy(p_all[2]) 
    frame_calibration_info.p2_2[0,3] = frame_calibration_info.p2_2[0,3] - frame_calibration_info.p2[0,3]

    frame_calibration_info.p2_3 = np.copy(p_all[3]) 
    frame_calibration_info.p2_3[0,3] = frame_calibration_info.p2_3[0,3] - frame_calibration_info.p2[0,3]

    frame_calibration_info.t_cam2_cam0 = np.zeros(3)
    frame_calibration_info.t_cam2_cam0[0] = (frame_calibration_info.p2[0,3] - frame_calibration_info.p0[0,3])/frame_calibration_info.p2[0,0]

    # Read in rectification matrix
    tr_rect = data[4]
    tr_rect = tr_rect[1:]
    tr_rect = [float(tr_rect[i]) for i in range(len(tr_rect))]
    frame_calibration_info.r0_rect = np.reshape(tr_rect, (3, 3))

    # Read in velodyne to cam matrix
    tr_v2c = data[5]
    tr_v2c = tr_v2c[1:]
    tr_v2c = [float(tr_v2c[i]) for i in range(len(tr_v2c))]
    frame_calibration_info.tr_velodyne_to_cam0 = np.reshape(tr_v2c, (3, 4))

    return frame_calibration_info

def read_obj_data(LABEL_PATH, calib=None, im_shape=None):
    '''Reads in object label file from Kitti Object Dataset.

        Inputs:
            LABEL_PATH : Str PATH of the label file.
        Returns:
            List of KittiObject : Contains all the labeled data

    '''
    used_cls = ['Car', 'Van' ,'Truck', 'Misc']
    objects = []

    detection_data = open(LABEL_PATH, 'r')
    detections = detection_data.readlines()

    for object_index in range(len(detections)):
        
        data_str = detections[object_index]
        data_list = data_str.split()
        
        if data_list[0] not in used_cls:
            continue

        object_it = KittiObject()

        object_it.cls = data_list[0]
        object_it.truncate = float(data_list[1])
        object_it.occlusion = int(data_list[2])
        object_it.alpha = float(data_list[3])

        #                            width          height         lenth
        object_it.dim = np.array([data_list[9], data_list[8], data_list[10]]).astype(float)

        # The KITTI GT 3D box is on cam0 frame, while we deal with cam2 frame
        object_it.pos = np.array(data_list[11:14]).astype(float) + calib.t_cam2_cam0 # 0.062
        # The orientation definition is inconsitent with right-hand coordinates in kitti
        object_it.orientation = float(data_list[14]) + m.pi/2  
        object_it.R = E2R(object_it.orientation, 0, 0)

        pts3_c_o = []  # 3D location of 3D bounding box corners
        pts3_c_o.append(object_it.pos + object_it.R.dot([-object_it.dim[0], 0, -object_it.dim[2]])/2.0)
        pts3_c_o.append(object_it.pos + object_it.R.dot([-object_it.dim[0], 0, object_it.dim[2]])/2.0)
        pts3_c_o.append(object_it.pos + object_it.R.dot([object_it.dim[0], 0, object_it.dim[2]])/2.0)
        pts3_c_o.append(object_it.pos + object_it.R.dot([object_it.dim[0], 0, -object_it.dim[2]])/2.0)

        pts3_c_o.append(object_it.pos + object_it.R.dot([-object_it.dim[0 ], -2.0*object_it.dim[1], -object_it.dim[2]])/2.0)
        pts3_c_o.append(object_it.pos + object_it.R.dot([-object_it.dim[0], -2.0*object_it.dim[1], object_it.dim[2]])/2.0)
        pts3_c_o.append(object_it.pos + object_it.R.dot([object_it.dim[0], -2.0*object_it.dim[1], object_it.dim[2]])/2.0)
        pts3_c_o.append(object_it.pos + object_it.R.dot([object_it.dim[0], -2.0*object_it.dim[1], -object_it.dim[2]])/2.0)

        object_it.boxes[0].box = np.array([10000, 10000, 0, 0]).astype(float)
        object_it.boxes[1].box = np.array([10000, 10000, 0, 0]).astype(float)
        object_it.boxes[2].box = np.array([0.0, 0.0, 0.0, 0.0]).astype(float)
        object_it.boxes[0].keypoints = np.array([-1.0, -1.0, -1.0, -1.0]).astype(float)
        object_it.boxes[1].keypoints = np.array([-1.0, -1.0, -1.0, -1.0]).astype(float)
        for j in range(2): # left and right boxes
            for i in range(8):
                if pts3_c_o[i][2] < 0:
                    continue
                if j == 0:    # project 3D corner to left image
                    pt2 = Space2Image(calib.p2_2, NormalizeVector(pts3_c_o[i]))
                elif j == 1:  # project 3D corner to right image
                    pt2 = Space2Image(calib.p2_3, NormalizeVector(pts3_c_o[i]))
                if i < 4:
                    object_it.boxes[j].keypoints[i] = pt2[0] 

                object_it.boxes[j].box[0] = min(object_it.boxes[j].box[0], pt2[0])
                object_it.boxes[j].box[1] = min(object_it.boxes[j].box[1], pt2[1]) 
                object_it.boxes[j].box[2] = max(object_it.boxes[j].box[2], pt2[0])
                object_it.boxes[j].box[3] = max(object_it.boxes[j].box[3], pt2[1]) 

            object_it.boxes[j].box[0] = max(object_it.boxes[j].box[0], 0)
            object_it.boxes[j].box[1] = max(object_it.boxes[j].box[1], 0) 

            if im_shape is not None:
                object_it.boxes[j].box[2] = min(object_it.boxes[j].box[2], im_shape[1]-1)
                object_it.boxes[j].box[3] = min(object_it.boxes[j].box[3], im_shape[0]-1)

            # deal with unvisible keypoints
            left_keypoint, right_keypoint = 5000, 0
            left_inx, right_inx = -1, -1
            # 1. Select keypoints that lie on the left and right side of the 2D box
            for i in range(4):
                if object_it.boxes[j].keypoints[i] < left_keypoint:
                    left_keypoint = object_it.boxes[j].keypoints[i]
                    left_inx = i
                if object_it.boxes[j].keypoints[i] > right_keypoint:
                    right_keypoint = object_it.boxes[j].keypoints[i]
                    right_inx = i
            # 2. For keypoints between left and right side, select the visible one
            for i in range(4):
                if i == left_inx or i == right_inx:
                    continue
                if pts3_c_o[i][2] > object_it.pos[2]:
                    object_it.boxes[j].keypoints[i] = -1

        # calculate the union of the left and right box
        object_it.boxes[2].box[0] = min(object_it.boxes[1].box[0], object_it.boxes[0].box[0])
        object_it.boxes[2].box[1] = min(object_it.boxes[1].box[1], object_it.boxes[0].box[1])
        object_it.boxes[2].box[2] = max(object_it.boxes[1].box[2], object_it.boxes[0].box[2])
        object_it.boxes[2].box[3] = max(object_it.boxes[1].box[3], object_it.boxes[0].box[3])

        objects.append(object_it)

    return objects

def project_to_image(point_cloud, p):
    ''' Projects a 3D point cloud to 2D points for plotting

        Inputs:
            point_cloud: 3D point cloud (3, N)
            p: Camera matrix (3, 4)
        Return: 
            pts_2d: the image coordinates of the 3D points in the shape (2, N)

    '''

    pts_2d = np.dot(p, np.append(point_cloud,
                                 np.ones((1, point_cloud.shape[1])),
                                 axis=0))

    pts_2d[0, :] = pts_2d[0, :] / pts_2d[2, :]
    pts_2d[1, :] = pts_2d[1, :] / pts_2d[2, :]
    pts_2d = np.delete(pts_2d, 2, 0)
    return pts_2d

def point_in_2Dbox(points_im, obj):
    '''Select points contained in object 2D box
    
        Inputs:
            points_im: N x 2 numpy array in image
            obj: KittiObject
        Return:
            pointcloud indexes

    '''
    point_filter = (points_im[:, 0] > obj.box[0]) & \
                    (points_im[:, 0] < obj.box[2]) & \
                    (points_im[:, 1] > obj.box[1]) & \
                    (points_im[:, 1] < obj.box[3])
    return point_filter

def lidar_to_cam_frame(xyz_lidar, frame_calib):
    '''Transforms the pointclouds to the camera 2 frame.
        
        Inputs:
            xyz_lidar : N x 3  x,y,z coordinates of the pointcloud in lidar frame
            frame_calib : FrameCalibrationData
        Returns:
            ret_xyz : N x 3  x,y,z coordinates of the pointcloud in cam2 frame
    
    '''
    # Pad the r0_rect matrix to a 4x4
    r0_rect_mat = frame_calib.r0_rect
    r0_rect_mat = np.pad(r0_rect_mat, ((0, 1), (0, 1)),
                         'constant', constant_values=0)
    r0_rect_mat[3, 3] = 1

    # Pad the tr_vel_to_cam matrix to a 4x4
    tf_mat = frame_calib.tr_velodyne_to_cam0
    tf_mat = np.pad(tf_mat, ((0, 1), (0, 0)),
                    'constant', constant_values=0)
    tf_mat[3, 3] = 1

    # Pad the t_cam2_cam0 matrix to a 4x4
    t_cam2_cam0 = np.identity(4)
    t_cam2_cam0[0:3, 3] = frame_calib.t_cam2_cam0

    # Pad the pointcloud with 1's for the transformation matrix multiplication
    one_pad = np.ones(xyz_lidar.shape[0]).reshape(-1, 1)
    xyz_lidar = np.append(xyz_lidar, one_pad, axis=1)

    # p_cam = P2 * R0_rect * Tr_velo_to_cam * p_velo
    rectified = np.dot(r0_rect_mat, tf_mat)
    
    to_cam2 = np.dot(t_cam2_cam0, rectified)
    ret_xyz = np.dot(to_cam2, xyz_lidar.T)

    # Change to N x 3 array for consistency.
    return ret_xyz[0:3].T

def get_point_cloud(LIDAR_PATH, frame_calib, image_shape=None, objects=None):
    ''' Calculates the lidar point cloud, and optionally returns only the
        points that are projected to the image.

        Inputs:
            LIDAR_PATH: string specify the lidar file path
            frame_calib: FrameCalibrationData
        Return:
            (3, N) point_cloud in the form [[x,...][y,...][z,...]]

    '''
    if not os.path.isfile(LIDAR_PATH):
        return np.array([[0],[0],[0]])
    if image_shape is not None:
        im_size = [image_shape[1], image_shape[0]]
    else:
        im_size = [1242, 375]

    with open(LIDAR_PATH, 'rb') as fid:
        data_array = np.fromfile(fid, np.single)
    xyzi = data_array.reshape(-1, 4)
    x = xyzi[:, 0]
    y = xyzi[:, 1]
    z = xyzi[:, 2]
    i = xyzi[:, 3]

    # Calculate the point cloud
    point_cloud = np.vstack((x, y, z))
    point_cloud = lidar_to_cam_frame(point_cloud.T, frame_calib)  # N x 3

    # Only keep points in front of camera (positive z)
    point_cloud = point_cloud[point_cloud[:,2] > 0].T   # camera frame 3 x N

    # Project to image frame
    point_in_im = project_to_image(point_cloud, p=frame_calib.p2).T

    # Filter based on the given image size
    
    image_filter = (point_in_im[:, 0] > 0) & \
                    (point_in_im[:, 0] < im_size[0]) & \
                    (point_in_im[:, 1] > 0) & \
                    (point_in_im[:, 1] < im_size[1])
     
    object_filter = np.zeros(point_in_im.shape[0], dtype=bool)
    if objects is not None:
        for i in range(len(objects)):
            object_filter = np.logical_or(point_in_2Dbox(point_in_im, objects[i]), object_filter)

        object_filter = np.logical_and(image_filter, object_filter)
    else:
        object_filter = image_filter

    point_cloud = point_cloud.T[object_filter].T

    return point_cloud

def infer_boundary(im_shape, boxes_left):
    ''' Approximately infer the occlusion border for all objects
        accoording to the 2D bounding box
        
        Inputs:
            im_shape: H x W x 3
            boxes_left: rois x 4
        Return:
            left_right: left and right borderline for each object rois x 2
    '''
    left_right = np.zeros((boxes_left.shape[0],2), dtype=np.float32)
    depth_line = np.zeros(im_shape[1]+1, dtype=float)
    for i in range(boxes_left.shape[0]):
        for col in range(int(boxes_left[i,0]), int(boxes_left[i,2])+1):
            pixel = depth_line[col]
            depth = 1050.0/boxes_left[i,3]
            if pixel == 0.0:
                depth_line[col] = depth
            elif depth < depth_line[col]:
                depth_line[col] = (depth+pixel)/2.0

    for i in range(boxes_left.shape[0]):
        left_right[i,0] = boxes_left[i,0]
        left_right[i,1] = boxes_left[i,2]
        left_visible = True
        right_visible = True
        if depth_line[int(boxes_left[i,0])] < 1050.0/boxes_left[i,3]:
            left_visible = False
        if depth_line[int(boxes_left[i,2])] < 1050.0/boxes_left[i,3]:
            right_visible = False

        if right_visible == False and left_visible == False:
            left_right[i,1] = boxes_left[i,0]

        for col in range(int(boxes_left[i,0]), int(boxes_left[i,2])+1):
            if left_visible and depth_line[col] >= 1050.0/boxes_left[i,3]:
                left_right[i,1] = col
            elif right_visible and depth_line[col] < 1050.0/boxes_left[i,3]:
                left_right[i,0] = col
    return left_right

#---------------------------------------------------- Data Writing -------------------------------------------------#
def write_detection_results(result_dir, file_number, calib, box_left, pos, dim, orien, score):
    '''One by one write detection results to KITTI format label files.
    '''
    if result_dir is None: return
    result_dir = result_dir + '/data'
    # convert the object from cam2 to the cam0 frame
    dis_cam02 = calib.t_cam2_cam0[0]
    
    output_str = 'Car -1 -1 '
    alpha = orien - m.pi/2 + m.atan2(-pos[0], pos[2])
    output_str += '%f %f %f %f %f ' % (alpha, box_left[0],box_left[1],box_left[2],box_left[3])
    output_str += '%f %f %f %f %f %f %f %f \n' % (dim[1],dim[0],dim[2],pos[0]-dis_cam02,pos[1],\
                                                  pos[2],orien-1.57,score) 

    # Write TXT files
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    pred_filename = result_dir + '/' + file_number + '.txt'
    with open(pred_filename, 'a') as det_file:
        det_file.write(output_str)














    
