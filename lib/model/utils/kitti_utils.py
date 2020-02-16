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
class KittiObject:

    def __init__(self):
        self.cls = ''         # Car, Van, Truck, Pedistrian, Person_sitting, Cyclist, Tram, Misc, DontCare
        self.truncate = 0     # float 0(non-truncated) - 1(truncated)
        self.occlusion = 0    # integer 0, 1, 2, 3
        self.alpha = 0        # Observe angle -pi - pi
        self.box = []         # left, top, right bottom in 2D image
        self.box_left = []    # left, top, right bottom in 2D image
        self.box_right = []   # left, top, right bottom in 2D image
        self.box_merge = []   # left, top, right bottom in 2D image
        self.keypoints = []  
        self.keypoints_right = []  
        self.pos = []         # x, y, z in camera frame
        self.dim = []         # width(x), height(y), length(z), 
        self.orientation = 0  # -pi - pi
        self.R = []         # rotation in camera frame
        self.visible_left = 0
        self.visible_right = 0
        self.visible_left_right = 0
        self.visible_right_right = 0

class FrameCalibrationData:
    """Frame Calibration Holder
        3x4    p0-p3      Camera P matrix. Contains extrinsic
                          and intrinsic parameters.
        3x3    r0_rect    Rectification matrix, required to transform points
                          from velodyne to camera coordinate frame.
        3x4    tr_velodyne_to_cam0    Used to transform from velodyne to cam
                                     coordinate frame according to:
                                     Point_Camera = P_cam * R0_rect *
                                                    Tr_velo_to_cam *
                                                    Point_Velodyne.
        """

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
    pts2_norm = P0.dot(pts3)
    pts2 = np.array([(pts2_norm[0]/pts2_norm[2]), (pts2_norm[1]/pts2_norm[2])])
    return pts2

def dis2point(focal_length, base_line, u0, v0, dispatiry, u, v):
    z = focal_length * base_line / dispatiry
    x = (u - u0) * z /focal_length
    y = (v - v0) * z /focal_length
    return x,y,z

def depth2point(focal_length, u0, v0, u, v, depth):
    z = depth
    x = (u - u0) * z /focal_length
    y = (v - v0) * z /focal_length
    return x,y,z

def depth2local_point(focal_length, u0, v0, u, v, depth, obj, plane_data):
    z = depth
    x = (u - u0) * z /focal_length
    y = (v - v0) * z /focal_length
    pts_c = np.array([x,y,z])

    y_ground = (-plane_data[0]*x - plane_data[2]*z -plane_data[3])/plane_data[1]
    if y >= y_ground - 0.07:
        return False, np.array([0,0,0])   #point is under or on ground, mark it
    pts_o = obj.R.T.dot(pts_c - obj.pos)
    dim_expand = obj.dim * 1.2
    if ((pts_o[0] < -dim_expand[0]/2) or \
                    (pts_o[0] > +dim_expand[0]/2) or \
                    (pts_o[1] < -(dim_expand[1]/1.2 - 0.05)) or \
                    (pts_o[1] > 0) or \
                    (pts_o[2] < -dim_expand[2]/2) or \
                    (pts_o[2] > +dim_expand[2]/2)):
        return False, np.array([0,0,0])   #point is not in 3D box, mark it
    else:
        return True, pts_o + np.array([dim_expand[0]/2,dim_expand[1],dim_expand[2]/2])

def NormalizeVector(P):
    return np.append(P, [1])

#------------------------------------------------------ Array opreation ------------------------------------------------------#
def depth_image2dis(depth_im, frame_calib):
    '''
    Convert depth image to disparity image
    depth_im: H x W uint16 np.array
    frame_calib: Full calib information
    Return: Disparity image: H x W uint16 np.array
    '''
    fb = frame_calib.p2[0,3] - frame_calib.p3[0,3] # focal_length x baseline
    depth = (depth_im).astype(float)/256
    depth[depth[:,:] == 0] = 10000
    dis = fb/depth
    dis_img = (dis*256).astype(np.uint16)

    return dis_img

def lidar_to_cam_frame(xyz_lidar, frame_calib):
    """Transforms the pointclouds to the camera 2 frame.
        Keyword Arguments:
        ------------------
        xyz_lidar : N x 3 Numpy Array
                  Contains the x,y,z coordinates of the lidar pointcloud
        frame_calib : FrameCalibrationData
                  Contains calibration information for a given frame
        Returns:
        --------
        ret_xyz : Numpy Array
                   Contains the xyz coordinates of the transformed pointcloud.
        """

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

def project_to_image(point_cloud, p):
    """ Projects a 3D point cloud to 2D points for plotting
    :param point_cloud: 3D point cloud (3, N)
    :param p: Camera matrix (3, 4)
    :return: pts_2d: the image coordinates of the 3D points in the shape (2, N)
    """

    pts_2d = np.dot(p, np.append(point_cloud,
                                 np.ones((1, point_cloud.shape[1])),
                                 axis=0))

    pts_2d[0, :] = pts_2d[0, :] / pts_2d[2, :]
    pts_2d[1, :] = pts_2d[1, :] / pts_2d[2, :]
    pts_2d = np.delete(pts_2d, 2, 0)
    return pts_2d

def points_inverse_transform(points, T_src_des):
    '''
    Transforming points from src frame to destination frame 
    
    points: N x 3 array in src frame
    T_src_des: 3 x 4 transform matrix of des frame in src frame
    '''

    # calsulate inverse transformation
    R_des_src = T_src_des[:,0:3].T
    P_des_src = np.dot(-R_des_src, T_src_des[:,3])
    T_des_src = np.c_[R_des_src, P_des_src]

    T_des_src = np.pad(T_des_src, ((0, 1), (0, 0)),
                    'constant', constant_values=0)
    T_des_src[3, 3] = 1

    # Pad the pointcloud with 1's for the transformation matrix multiplication
    one_pad = np.ones(points.shape[0]).reshape(-1, 1)
    xyz_lidar = np.append(points, one_pad, axis=1)

    # p_cam = P2 * R0_rect * Tr_velo_to_cam * p_velo
    ret_xyz = np.dot(T_des_src, xyz_lidar.T)

    # Change to N x 3 array for consistency.
    return ret_xyz[0:3].T

def point_in_3Dbox(point_cloud, obj):
    '''
    Function: select points contained in object 3D box
    
    pointcloud: N x 3 numpy array in camera frame
    
    obj: KittiObject
    Return: pointcloud indexes
    '''
    T_c_o = np.c_[obj.R,obj.pos]
    points_obj = points_inverse_transform(point_cloud, T_c_o)

    point_filter = (points_obj[:, 0] >= -obj.dim[0]/2) & \
                    (points_obj[:, 0] <= +obj.dim[0]/2) & \
                    (points_obj[:, 1] >= -obj.dim[1]) & \
                    (points_obj[:, 1] < -0.05) & \
                    (points_obj[:, 2] >= -obj.dim[2]/2) & \
                    (points_obj[:, 2] <= +obj.dim[2]/2)
    return point_filter

def point_in_2Dbox(points_im, obj):
    '''
    Function: select points contained in object 2D box
    
    pointcloud: N x 2 numpy array in image
    
    obj: KittiObject
    Return: pointcloud indexes
    '''
    point_filter = (points_im[:, 0] > obj.box[0]) & \
                    (points_im[:, 0] < obj.box[2]) & \
                    (points_im[:, 1] > obj.box[1]) & \
                    (points_im[:, 1] < obj.box[3])
    return point_filter

#------------------------------------------------------ Data reading ------------------------------------------------------#

def read_obj_calibration(CALIB_PATH):
    """Reads in Calibration file from Kitti Dataset.
    Keyword Arguments:
    ------------------
    CALIB_PATH : Str
                PATH of the calibration file.
    Returns:
    --------
    frame_calibration_info : FrameCalibrationData
                             Contains a frame's full calibration data.
    ^ z        ^ z                                      ^ z         ^ z
    | cam2     | cam0                                   | cam3      | cam1
    |-----> x  |-----> x                                |-----> x   |-----> x
    """
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

def read_plane(PLANE_PATH):
    """Reads in Plane file from Kitti Dataset.
    Keyword Arguments:
    ------------------
    CALIB_PATH : Str
                PATH of the calibration file.
    Returns:
    --------
    Plane data in 4 x 1 array Ax + By + Cz + D = 0
    """

    data_file = open(PLANE_PATH, 'r')
    data_reader = csv.reader(data_file, delimiter=' ')
    data = []

    for row in data_reader:
        data.append(row)

    data_file.close()

    ABCD = data[3]
    ABCD = [float(ABCD[i]) for i in range(len(ABCD))]
    ABCD = np.reshape(ABCD, 4)

    return ABCD

def read_obj_data(LABEL_PATH, calib = None, used_cls = ['Car', 'Van' ,'Truck', 'Misc'], im_shape=None):
    """Reads in object label file from Kitti Object Dataset.
    Keyword Arguments:
    ------------------
    LABEL_PATH : Str
                PATH of the label file.
    Returns:
    --------
    List of KittiObject : Contains a all the labeled data
    """
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
        object_it.box = np.array(data_list[4:8]).astype(float)
        object_it.box = object_it.box.astype(int)
        #                            width          height         lenth
        object_it.dim = np.array([data_list[9], data_list[8], data_list[10]]).astype(float)
        #convert it to camera 2 frame
        # TODO TO CHECK
        object_it.pos = np.array(data_list[11:14]).astype(float) + calib.t_cam2_cam0
        object_it.orientation = float(data_list[14]) + m.pi/2  # orientation definition inconsitent in kitti
        object_it.R = E2R(object_it.orientation, 0, 0)
        if calib is not None:
            pts3_c_o = []
            pts3_c_o.append(object_it.pos + object_it.R.dot([-object_it.dim[0], 0, -object_it.dim[2]])/2.0)
            pts3_c_o.append(object_it.pos + object_it.R.dot([-object_it.dim[0], 0, object_it.dim[2]])/2.0)
            pts3_c_o.append(object_it.pos + object_it.R.dot([object_it.dim[0], 0, object_it.dim[2]])/2.0) ##max
            pts3_c_o.append(object_it.pos + object_it.R.dot([object_it.dim[0], 0, -object_it.dim[2]])/2.0)

            pts3_c_o.append(object_it.pos + object_it.R.dot([-object_it.dim[0 ], -2.0*object_it.dim[1], -object_it.dim[2]])/2.0) ##min
            pts3_c_o.append(object_it.pos + object_it.R.dot([-object_it.dim[0], -2.0*object_it.dim[1], object_it.dim[2]])/2.0)
            pts3_c_o.append(object_it.pos + object_it.R.dot([object_it.dim[0], -2.0*object_it.dim[1], object_it.dim[2]])/2.0)
            pts3_c_o.append(object_it.pos + object_it.R.dot([object_it.dim[0], -2.0*object_it.dim[1], -object_it.dim[2]])/2.0)

            # Project 3D to right
            object_it.box_left = np.array([10000, 10000, 0, 0]).astype(float) # left top,right, bottom
            object_it.box_right = np.array([10000, 10000, 0, 0]).astype(float) # left top,right, bottom
            object_it.box_merge = np.array([0.0, 0.0, 0.0, 0.0]).astype(float) # left top,right, bottom
            object_it.keypoints = np.array([-1.0, -1.0, -1.0, -1.0]).astype(float)
            object_it.keypoints_right = np.array([-1.0, -1.0, -1.0, -1.0]).astype(float)
            for i in range(8):
                if pts3_c_o[i][2] < 0:
                    continue
                pt2_left = Space2Image(calib.p2_2, NormalizeVector(pts3_c_o[i]))
                if i < 4:
                    object_it.keypoints[i] = pt2_left[0] 

                object_it.box_left[0] = min(object_it.box_left[0], pt2_left[0])
                object_it.box_left[1] = min(object_it.box_left[1], pt2_left[1]) 
                
                object_it.box_left[2] = max(object_it.box_left[2], pt2_left[0])
                object_it.box_left[3] = max(object_it.box_left[3], pt2_left[1]) 

                pt2_right = Space2Image(calib.p2_3, NormalizeVector(pts3_c_o[i]))
                if i < 4:
                    object_it.keypoints_right[i] = pt2_right[0] 

                object_it.box_right[0] = min(object_it.box_right[0], pt2_right[0])
                object_it.box_right[1] = min(object_it.box_right[1], pt2_right[1])

                object_it.box_right[2] = max(object_it.box_right[2], pt2_right[0])
                object_it.box_right[3] = max(object_it.box_right[3], pt2_right[1])

            object_it.box_left[0] = max(object_it.box_left[0], 0)
            object_it.box_left[1] = max(object_it.box_left[1], 0) 
            object_it.box_right[0] = max(object_it.box_right[0], 0)
            object_it.box_right[1] = max(object_it.box_right[1], 0) 

            if im_shape is not None:
                object_it.box_left[2] = min(object_it.box_left[2], im_shape[1]-1)
                object_it.box_left[3] = min(object_it.box_left[3], im_shape[0]-1)
                object_it.box_right[2] = min(object_it.box_right[2], im_shape[1]-1)
                object_it.box_right[3] = min(object_it.box_right[3], im_shape[0]-1)

            object_it.box_merge[0] = min(object_it.box_right[0], object_it.box_left[0])
            object_it.box_merge[1] = min(object_it.box_right[1], object_it.box_left[1])
            object_it.box_merge[2] = max(object_it.box_right[2], object_it.box_left[2])
            object_it.box_merge[3] = max(object_it.box_right[3], object_it.box_left[3])

            # deal with unvisible left keypoints
            left_keypoint = 5000
            left_inx = -1
            right_keypoint = 0
            right_inx = -1
            # 1. Select left and right keypoints
            for i in range(4):
                if object_it.keypoints[i] < left_keypoint:
                    left_keypoint = object_it.keypoints[i]
                    left_inx = i
                if object_it.keypoints[i] > right_keypoint:
                    right_keypoint = object_it.keypoints[i]
                    right_inx = i
            # 2. For keypointss between left and right, select the visible one
            for i in range(4):
                if i == left_inx or i == right_inx:
                    continue
                if pts3_c_o[i][2] > object_it.pos[2]:
                    object_it.keypoints[i] = -1

            # deal with unvisible right keypoints
            left_keypoint = 5000
            left_inx = -1
            right_keypoint = 0
            right_inx = -1
            # 1. Select left and right keypoints
            for i in range(4):
                if object_it.keypoints_right[i] < left_keypoint:
                    left_keypoint = object_it.keypoints_right[i]
                    left_inx = i
                if object_it.keypoints_right[i] > right_keypoint:
                    right_keypoint = object_it.keypoints_right[i]
                    right_inx = i
            # 2. For keypoints_rights between left and right, select the visible one
            for i in range(4):
                if i == left_inx or i == right_inx:
                    continue
                if pts3_c_o[i][2] > object_it.pos[2]:
                    object_it.keypoints_right[i] = -1
            
        objects.append(object_it)

    return objects

def get_point_cloud(LIDAR_PATH, frame_calib, image_shape=None, objects=None):

    """ Calculates the lidar point cloud, and optionally returns only the
    points that are projected to the image.
    :param img_idx: image index
    :param calib_dir: directory with calibration files
    :param velo_dir: directory with velodyne files
    :param im_size: (optional) 2 x 1 list containing the size of the image
                      to filter the point cloud [w, h]
    :param min_intensity: (optional) minimum intensity required to keep a point
    :return: (3, N) point_cloud in the form [[x,...][y,...][z,...]]
    """
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

    # Creat depth image
    point_in_im = point_in_im[object_filter]
    #                        height     width
    pts_z = point_cloud[2,:]
    depth_image = np.zeros((im_size[1], im_size[0]), dtype=np.uint16)
    depth_image[(point_in_im[:,1]).astype(int), point_in_im[:,0].astype(int)] = (pts_z * 256).astype('uint16')

    return point_cloud, depth_image

def data_collector(data_path, image_number):
    calib_path = data_path + 'training/calib/' + image_number + '.txt'
    label_path = data_path + 'training/label_2/' + image_number + '.txt'
    point_cloud_path = data_path + 'training/velodyne/' + image_number + '.bin'
    calib_it = read_obj_calibration(calib_path)
    objects = read_obj_data(label_path,calib_it)
    pointcloud_lidar_c, depth_lidar = get_point_cloud(point_cloud_path, calib_it)
    return calib_it, objects, pointcloud_lidar_c

#---------------------------------------------------- Data Writing -------------------------------------------------#

def write_estimation_results(result_dir, fiLe_number, objects):
    '''
    Write detection results to KITTI format label files.
    '''
    if result_dir is None:
        return
    results = [] # each string is a line (without \n)
    for i in range(len(objects)):
        output_str = 'Car' + " -1 -1 -10 "
        output_str += "%f %f %f %f " %(objects[i].box[0],objects[i].box[1],objects[i].box[2],objects[i].box[3]) 

        box_score = 0.006*(objects[i].box[3] -objects[i].box[1])+0.12
        box_score = max(1.0, box_score)
        box_score = min(0.3, box_score) 
        
        status_score = 1.0
        if objects[i].box[0] < 5 or objects[i].box[2] > 1235:
            status_score = 0.5 

        score = (status_score - box_score)/2.0
        output_str += "%f %f %f %f %f %f %f %f" % (objects[i].dim[1],objects[i].dim[0],objects[i].dim[2],\
                      objects[i].pos[0],objects[i].pos[1],objects[i].pos[2],objects[i].orientation,score)
        results.append(output_str) 
    # Write TXT files
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    pred_filename = result_dir + '/' +  file_number + '.txt'
    print('save in', pred_filename)
    fout = open(pred_filename, 'w')
    for line in results:
        fout.write(line+ 'n')
    fout.close()

def write_detection_results(result_dir, file_number, box_left, pos, dim, orien, score, box32, ex_t):
    '''
    Write detection results to KITTI format label files.
    '''
    if result_dir is None: return
    output_str = 'Car -1 -1 -10 '
    output_str += '%f %f %f %f ' % (box_left[0],box_left[1],box_left[2],box_left[3])
    #score = score*30.0/(max(30.0,pos[2]))
    score = score*60.0/(max(30.0,pos[2])+30.0)

    insect_w = min(box32[2], box_left[2]) - max(box32[0], box_left[0]) + 1
    if insect_w<0:
        insect_w = 0
    insect_h = min(box32[3], box_left[3]) - max(box32[1], box_left[1]) + 1
    if insect_h<0:
        insect_h = 0
    area_3d = (box32[2]-box32[0]+1)*(box32[3]-box32[1]+1)
    area_2d = (box_left[2]-box_left[0]+1)*(box_left[3]-box_left[1]+1)
    iou = insect_w*insect_h/(area_3d+area_2d-insect_w*insect_h)
    score = score*(min(0.9,iou)*3.0-1.7)
    output_str += '%f %f %f %f %f %f %f %f \n' % (dim[1],dim[0],dim[2],pos[0]-ex_t,pos[1],pos[2],orien-1.57,score) 

    # Write TXT files
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    pred_filename = result_dir + '/' + file_number + '.txt'
    with open(pred_filename, 'a') as det_file:
        det_file.write(output_str)

def depth_dis_table(calib, depth_range = np.array([0,80]), res = 0.1):
    '''
    generate depth and disparity table,
    res for depth is given by res
    res for dispatiry is 1 pixel
    '''
    fb = (calib.p2[0,3] - calib.p3[0,3])
    table = []
    table_range = (depth_range/res).astype(int)

    for i in range(5, 193): #dispatiry range
        depth = fb/i
        depth_index = int(depth/res + 0.5)

        last_depth = fb/(i-1)
        last_depth_index = int(last_depth/res + 0.5)
        if last_depth_index - depth_index < 1:
            continue
        table.append((int(depth_range[1]/res - depth_index), i))
    return table

def depth_im_to_local_im(depth_im, frame_calib, objects, plane_data, save_pth):

    """ Convert the depth image of object area to object local frame.
    :param depth_im: entire left depth image
    :param frame_calib: calibration information
    :param objects: all the objects from ground truth
    :return: 3 channels uint16 image R(x)G(y)B(z) which encodes the local frame of obejct pixel
    """
    focal_length = frame_calib.p2[0,0]
    u0, v0 = frame_calib.p2[0,2], frame_calib.p2[1,2]

    objs_local = np.zeros((depth_im.shape[0], depth_im.shape[1], 3), dtype=np.uint8)

    if os.path.exists(save_pth):
        #shutil.rmtree(save_pth)
        return objs_local
    os.makedirs(save_pth)

    for i in range(len(objects)):
        obj_local = np.zeros((objects[i].box[3]-objects[i].box[1], objects[i].box[2]-objects[i].box[0], 3), dtype=np.uint8)
        for row in range(objects[i].box[1],objects[i].box[3]):
            for col in range(objects[i].box[0],objects[i].box[2]):
                if depth_im[row, col] != 0:
                    in_box, pts_o = depth2local_point(focal_length, u0, v0, col, row, ((float)(depth_im[row, col]))/256, objects[i], plane_data)
                    if in_box == True:
                        pts_o = pts_o * 50
                        objs_local[row, col, 2] = (pts_o[0]).astype('uint8') if (pts_o[0]).astype('uint8') <=255 else 255  # X -> R 
                        objs_local[row, col, 1] = (pts_o[1]).astype('uint8') if (pts_o[1]).astype('uint8') <=255 else 255  # Y -> G
                        objs_local[row, col, 0] = (pts_o[2]).astype('uint8') if (pts_o[2]).astype('uint8') <=255 else 255  # Z -> B

                        obj_local[row-objects[i].box[1], col-objects[i].box[0], 2] = (pts_o[0]).astype('uint8') if (pts_o[0]).astype('uint8') <=255 else 255  # X -> R 
                        obj_local[row-objects[i].box[1], col-objects[i].box[0], 1] = (pts_o[1]).astype('uint8') if (pts_o[1]).astype('uint8') <=255 else 255  # Y -> G
                        obj_local[row-objects[i].box[1], col-objects[i].box[0], 0] = (pts_o[2]).astype('uint8') if (pts_o[2]).astype('uint8') <=255 else 255  # Z -> B
                        
        im_path = save_pth + '/{:d}_{:d}.png'.format(objects[i].box[0], objects[i].box[1])
        cv2.imwrite(im_path, obj_local)

    return objs_local

def depth_im_to_object_im(depth_im, frame_calib, objects, plane_data, save_pth):

    """ Convert the depth image to seperate object depth.
    :param depth_im: entire left depth image
    :param frame_calib: calibration information
    :param objects: all the objects from ground truth
    :return: 1 channels uint16 depth image
    """
    focal_length = frame_calib.p2[0,0]
    u0, v0 = frame_calib.p2[0,2], frame_calib.p2[1,2]

    objs_depth = np.zeros((depth_im.shape[0], depth_im.shape[1]), dtype=np.uint16)

    if os.path.exists(save_pth):
        #shutil.rmtree(save_pth)
        return objs_depth
    os.makedirs(save_pth)

    for i in range(len(objects)):
        obj_depth = np.zeros((objects[i].box[3]-objects[i].box[1], objects[i].box[2]-objects[i].box[0]), dtype=np.uint16)
        for row in range(objects[i].box[1],objects[i].box[3]):
            for col in range(objects[i].box[0],objects[i].box[2]):
                if depth_im[row, col] != 0:
                    in_box, pts_o = depth2local_point(focal_length, u0, v0, col, row, ((float)(depth_im[row, col]))/256, objects[i], plane_data)
                    if in_box == True:
                        obj_depth[row-objects[i].box[1], col-objects[i].box[0]] = depth_im[row, col]
                        
        im_path = save_pth + '/{:d}_{:d}.png'.format(objects[i].box[0], objects[i].box[1])
        cv2.imwrite(im_path, obj_depth)

    return objs_depth














    
