"""
Test the 6DOF-GraspNet model for grasp generation
with input from a segmentation algorithm

Referenced the ~/demo/main.py script for this
"""
import os, sys, glob
import argparse
import threading
import datetime

import numpy as np
import cv2
from scipy.io import savemat
import torch
import torch.nn as nn
import torch.utils.data

import rospy
import tf
import rosnode
import message_filters
import tf2_ros
from tf.transformations import quaternion_matrix
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose, PoseStamped, PoseArray
from cv_bridge import CvBridge, CvBridgeError
from transforms3d.quaternions import mat2quat, quat2mat

import grasp_estimator
from utils.visualization_utils import *
from utils import utils

lock = threading.lock()


def ros_qt_to_rt(rot, trans):
    qt = np.zeros((4,), dtype=np.float32)
    qt[0] = rot[3]
    qt[1] = rot[0]
    qt[2] = rot[1]
    qt[3] = rot[2]
    obj_T = np.eye(4)
    obj_T[:3, :3] = quat2mat(qt)
    obj_T[:3, 3] = trans

    return obj_T


def rt_to_ros_qt(rt):
    """ 
    Returns (quat_xyzw, trans) from a 4x4 transform
    """
    quat = mat2quat(rt[:3, :3])
    quat = [quat[1], quat[2], quat[3], quat[0]]
    trans = rt[:3, 3]
    return quat, trans


def set_ros_pose(pose, quat, trans):
    """
    pose is a mutable reference to a Pose() object
    quat is in (x,y,z,w) format
    Sets the fields in pose var and modifies it
    """
    pose.position.x = trans[0]
    pose.position.x = trans[1]
    pose.position.x = trans[2]
    pose.orientation.x = quat[0]
    pose.orientation.x = quat[1]
    pose.orientation.x = quat[2]
    pose.orientation.x = quat[3]


def compute_xyz(depth_img, fx, fy, px, py, height, width):
    indices = np.indices((height, width), dtype=np.float32).transpose(1, 2, 0)
    z_e = depth_img
    x_e = (indices[..., 1] - px) * z_e / fx
    y_e = (indices[..., 0] - py) * z_e / fy
    xyz_img = np.stack([x_e, y_e, z_e], axis=-1)  # Shape: [H x W x 3]
    return xyz_img


def get_color_for_pc(pc, K, color_image):
    proj = pc.dot(K.T)
    proj[:, 0] /= proj[:, 2]
    proj[:, 1] /= proj[:, 2]

    pc_colors = np.zeros((pc.shape[0], 3), dtype=np.uint8)
    for i, p in enumerate(proj):
        x = int(p[0])
        y = int(p[1])
        pc_colors[i, :] = color_image[y, x, :]

    return pc_colors


def backproject(
    depth_cv, intrinsic_matrix, return_finite_depth=True, return_selection=False
):
    depth = depth_cv.astype(np.float32, copy=True)

    # get intrinsic matrix
    K = intrinsic_matrix
    Kinv = np.linalg.inv(K)

    # compute the 3D points
    width = depth.shape[1]
    height = depth.shape[0]

    # construct the 2D points matrix
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    ones = np.ones((height, width), dtype=np.float32)
    x2d = np.stack((x, y, ones), axis=2).reshape(width * height, 3)

    # backprojection
    R = np.dot(Kinv, x2d.transpose())

    # compute the 3D points
    X = np.multiply(np.tile(depth.reshape(1, width * height), (3, 1)), R)
    X = np.array(X).transpose()
    if return_finite_depth:
        selection = np.isfinite(X[:, 0])
        X = X[selection, :]

    if return_selection:
        return X, selection

    return X


def extract_points(depth_cv, xyz_image, label, camera_pose):
    kernel = np.ones((3, 3), np.uint8)
    mask_ids = np.unique(label)
    assert len(mask_ids) == 1
    mask_id = mask_ids[0]

    mask = np.array(label == mask_id).astype(np.uint8)
    mask2 = cv2.erode(mask, kernel)
    mask = (mask2 > 0) & (depth_cv > 0)
    points = xyz_image[mask, :]

    # convert points to robot base
    points_base = np.matmul(camera_pose[:3, :3], points.T) + camera_pose[:3, 3].reshape((3, 1))
    points_base = points_base.T
    selection = np.isfinite(points_base[:, 0])
    points_base = points_base[selection, :]
    points_cent = np.mean(points_base, axis=0)
    return points_base, points_cent


class GraspPosePubSub:

    def __init__(self, grasp_estimator):

        self.estimator = grasp_estimator
        self.cv_bridge = CvBridge()

        self.im = None
        self.depth = None
        self.depth_frame_id = None
        self.rgb_frame_id = None
        self.rgb_frame_stamp = None
        self.xyz_image = None
        self.label = None

        self.counter = 0
        self.output_dir = 'output/real_world'

        # initialize a node
        rospy.init_node("pose_6dof")
        self.pose_pub = rospy.Publisher('6dof_pose', PoseArray, queue_size=10)

        self.base_frame = 'base_link'
        rgb_sub = message_filters.Subscriber('/selected_rgb', Image, queue_size=10)
        depth_sub = message_filters.Subscriber('/selected_depth', Image, queue_size=10)
        label_sub = message_filters.Subscriber('/selected_label', Image, queue_size=10)
        msg = rospy.wait_for_message('/head_camera/rgb/camera_info', CameraInfo)
        self.camera_frame = 'head_camera_rgb_optical_frame'
        self.target_frame = self.base_frame

        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(100.0))  # tf buffer length    
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # update camera intrinsics
        intrinsics = np.array(msg.K).reshape(3, 3)
        self.fx = intrinsics[0, 0]
        self.fy = intrinsics[1, 1]
        self.px = intrinsics[0, 2]
        self.py = intrinsics[1, 2]
        print("Camera Intrinsics:", intrinsics)

        # camera pose in base
        transform = self.tf_buffer.lookup_transform(self.base_frame,
                                           # source frame:
                                           self.camera_frame,
                                           # get the tf at the time the pose was valid
                                           rospy.Time.now(),
                                           # wait for at most 1 second for transform, otherwise throw
                                           rospy.Duration(1.0)).transform
        quat = [transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w]
        RT = quaternion_matrix(quat)
        RT[0, 3] = transform.translation.x
        RT[1, 3] = transform.translation.y        
        RT[2, 3] = transform.translation.z
        self.camera_pose = RT
        # print(self.camera_pose)

        queue_size = 1
        slop_seconds = 0.1
        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub, label_sub], queue_size, slop_seconds)
        ts.registerCallback(self.callback_rgbd)


    def callback_rgbd(self, rgb, depth, label):

        if depth.encoding == '32FC1':
            depth_cv = self.cv_bridge.imgmsg_to_cv2(depth)
        elif depth.encoding == '16UC1':
            depth_cv = self.cv_bridge.imgmsg_to_cv2(depth).copy().astype(np.float32)
            depth_cv /= 1000.0
        else:
            rospy.logerr_throttle(
                1, 'Unsupported depth type. Expected 16UC1 or 32FC1, got {}'.format(
                    depth.encoding))
            return

        im = self.cv_bridge.imgmsg_to_cv2(rgb, 'bgr8')
        label = self.cv_bridge.imgmsg_to_cv2(label)
        # compute xyz image
        height = depth_cv.shape[0]
        width = depth_cv.shape[1]
        xyz_image = compute_xyz(depth_cv, self.fx, self.fy, self.px, self.py, height, width)

        # kernel = np.ones((3, 3), np.uint8)
        # mask_ids = np.unique(label)
        # assert len(mask_ids) == 1
        # mask_id = mask_ids[0]

        # mask = np.array(label == mask_id).astype(np.uint8)
        # mask2 = cv2.erode(mask, kernel)
        # mask = (mask2 > 0) & (depth_cv > 0)
        # points = xyz_image[mask, :]

        # # convert points to robot base
        # points_base = np.matmul(self.camera_pose[:3, :3], points.T) + self.camera_pose[:3, 3].reshape((3, 1))
        # points_base = points_base.T
        # selection = np.isfinite(points_base[:, 0])
        # points_base = points_base[selection, :]
        # points_cent = np.mean(points_base, axis=0)
        points_base, points_cent = extract_points(depth_cv, xyz_image, label, 
                                                  self.camera_pose)

        with lock:
            self.im = im.copy()
            self.label = label.copy()
            self.depth = depth_cv.copy()
            self.depth_frame_id = depth.header.frame_id
            self.rgb_frame_id = rgb.header.frame_id
            self.rgb_frame_stamp = rgb.header.stamp
            self.xyz_image = xyz_image
            self.points_base = points_base.copy()
            self.points_cent = points_cent.copy()


    def run_network(self):

        with lock:
            if listener.im is None:
              return
            im_color = self.im.copy()
            depth_img = self.depth.copy()
            xyz_image = self.xyz_image.copy()
            points_base = self.points_base.copy()
            points_cent = self.points_cent.copy()
            label = self.label.copy()
            rgb_frame_id = self.rgb_frame_id
            depth_frame_id = self.depth_frame_id
            rgb_frame_stamp = self.rgb_frame_stamp

        print('===================================================')
        # run the network
        gen_grasps, gen_scores = self.estimator.generate_and_refine_grasps(
            points_base)

        prob_index = np.argsort(gen_scores)
        sorted_graps = gen_grasps[prob_index, :] # list of [rt_grasps]

        parray = PoseArray()
        parray.header.frame_id = "/base_link"
        parray.header.stamp = rgb_frame_stamp #rospy.Time.now()
        for grasp in sorted_graps:
            quat, trans = rt_to_ros_qt(grasp)
            p = Pose()
            set_ros_pose(p, quat, trans)
            parray.poses.append(p)
        
        self.pose_pub.publish(parray)


def make_parser():
    parser = argparse.ArgumentParser(
        description="6-DoF GraspNet Demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--grasp_sampler_folder", type=str, default="checkpoints/gan_pretrained/"
    )
    parser.add_argument(
        "--grasp_evaluator_folder",
        type=str,
        default="checkpoints/evaluator_pretrained/",
    )
    parser.add_argument(
        "--refinement_method", choices={"gradient", "sampling"}, default="sampling"
    )
    parser.add_argument("--refine_steps", type=int, default=25)

    # parser.add_argument("--npy_folder", type=str, default="demo/data/")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="When choose_fn is something else than all, all grasps with a score given by the evaluator notwork less than the threshold are removed",
    )
    parser.add_argument(
        "--choose_fn",
        choices={"all", "better_than_threshold", "better_than_threshold_in_sequence"},
        default="better_than_threshold",
        help="If all, no grasps are removed. If better than threshold, only the last refined grasps are considered while better_than_threshold_in_sequence consideres all refined grasps",
    )

    parser.add_argument("--target_pc_size", type=int, default=1024)
    parser.add_argument("--num_grasp_samples", type=int, default=200)
    parser.add_argument(
        "--generate_dense_grasps",
        action="store_true",
        help="If enabled, it will create a [num_grasp_samples x num_grasp_samples] dense grid of latent space values and generate grasps from these.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=30,
        help="Set the batch size of the number of grasps we want to process and can fit into the GPU memory at each forward pass. The batch_size can be increased for a GPU with more memory.",
    )
    # parser.add_argument("--train_data", action="store_true")
    opts, _ = parser.parse_known_args()
    return parser


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    grasp_sampler_args = utils.read_checkpoint_args(args.grasp_sampler_folder)
    grasp_sampler_args.is_train = False
    grasp_evaluator_args = utils.read_checkpoint_args(args.grasp_evaluator_folder)
    grasp_evaluator_args.continue_train = True
    
    estimator = grasp_estimator.GraspEstimator(
        grasp_sampler_args, grasp_evaluator_args, args
    )

    listener = GraspPosePubSub(estimator)
    
    while not rospy.is_shutdown():
        try:
           listener.run_network()
        except KeyboardInterrupt:
            break
    print("Exiting 6dof-pose generation ros node")

