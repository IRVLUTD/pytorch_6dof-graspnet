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
from sensor_msgs.msg import Image, CameraInfo, PointCloud
from geometry_msgs.msg import Pose, PoseArray, Point
from cv_bridge import CvBridge, CvBridgeError
from transforms3d.quaternions import mat2quat, quat2mat

import _init_paths
import grasp_estimator
from utils import utils
from utils.visualization_utils import *
import mayavi.mlab as mlab


lock = threading.Lock()


def rt_to_ros_qt(rt):
    """ 
    Returns (quat_xyzw, trans) from a 4x4 transform
    """
    quat = mat2quat(rt[:3, :3])
    quat = [quat[1], quat[2], quat[3], quat[0]]
    trans = rt[:3, 3]
    return quat, trans


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


def set_ros_pose(pose, quat, trans):
    """
    pose is a mutable reference to a Pose() object
    quat is in (x,y,z,w) format
    Sets the fields in pose var and modifies it
    """
    pose.position.x = trans[0]
    pose.position.y = trans[1]
    pose.position.z = trans[2]
    pose.orientation.x = quat[0]
    pose.orientation.y = quat[1]
    pose.orientation.z = quat[2]
    pose.orientation.w = quat[3]


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


# class to receive pointcloud and publish grasp pose array for it
class PointToGraspPubSub:

    def __init__(self, grasp_estimator):
        self.estimator = grasp_estimator
        self.points_base = None
        self.points_cent = None
        self.frame_stamp = None
        self.frame_id = None
        self.base_frame = 'base_link'
        self.SCALING_FACTOR = 0.8
        self.prev_step = None
        self.step = 0 # indicator for whether a new pc is registered
        # Create the transform the aligns Fetch with Panda Grasp
        # Apply the generated grasp pose on this transform to get pose for Fetch Gripper
        # i.e pose_fetch = pose_panda @ transform
        # _quat_tf = [0, -0.7071068, 0, 0.7071068]
        # _tran_tf = [0, 0, -0.08]
        _quat_tf = [ 0.5, -0.5, 0.5, 0.5 ]
        _tran_tf = [0, 0, -0.1]
        self._transform_grasp = ros_qt_to_rt(_quat_tf, _tran_tf)

        # initialize a node
        rospy.init_node("pose_graspnet")
        self.pose_pub = rospy.Publisher('pose_6dof', PoseArray, queue_size=10)
        point_sub = message_filters.Subscriber('/selected_objpts', PointCloud, queue_size=5)
        queue_size = 1
        slop_seconds = 0.1
        ts = message_filters.ApproximateTimeSynchronizer([point_sub], queue_size, slop_seconds)
        ts.registerCallback(self.callback_points)


    def callback_points(self, points_pc):
        pc_header = points_pc.header
        pc_frame_id = pc_header.frame_id
        pc_frame_stamp = pc_header.stamp
        n = len(points_pc.points)
        assert n > 0
        points_base = np.zeros((n, 3))
        for i, objpt in enumerate(points_pc.points):
            points_base[i, :] = [objpt.x, objpt.y, objpt.z]
        points_cent = np.mean(points_base, axis=0)
        # with lock:
        self.points_base = points_base.copy()
        self.points_cent = points_cent.copy()
        self.frame_id = pc_frame_id
        self.frame_stamp = pc_frame_stamp
        self.step += 1

    def run_network(self, viz=False):
        # with lock:
        if listener.points_base is None:
            return
        
        if self.prev_step == self.step:
            return
        self.prev_step = self.step

        points_base = self.points_base.copy()
        points_cent = self.points_cent.copy()
        frame_id = self.frame_id
        frame_stamp = self.frame_stamp
        print('===================================================')
        # run the network

        # Scale the points using scaling factor before passing through the network
        points_viz = points_base.copy() # no scaling applied to these
        center = np.mean(points_base, axis=0)
        points_base -= center
        points_base *= self.SCALING_FACTOR
        points_base += center
        pc_to_network = points_base.copy()

        gen_grasps, gen_scores = self.estimator.generate_and_refine_grasps(
            points_base)

        # Invert the scaling for the translation part of grasp pose
        # Rotation is not afffected
        gg = []
        for g in gen_grasps:
            g[:3, 3] -= center 
            g[:3, 3] * (1.0/self.SCALING_FACTOR)
            g[:3, 3] += center
            gg.append(g)

        # arg sort the grasps using scores with highest score first
        sort_index = sorted(range(len(gen_scores)), key=lambda i: gen_scores[i], reverse=True)
        # Go along the highest grasps first and convert to Fetch using the saved transform
        sorted_graps_fetch = [gg[i] @ self._transform_grasp for i in sort_index]
        parray = PoseArray()
        parray.header.frame_id = frame_id
        parray.header.stamp = frame_stamp #rospy.Time.now()
        for grasp in sorted_graps_fetch:
            quat, trans = rt_to_ros_qt(grasp)
            p = Pose()
            set_ros_pose(p, quat, trans)
            parray.poses.append(p)
        
        while True:
            if self.pose_pub.get_num_connections() > 0:
                rospy.loginfo(f"Publishing Grasp Pose Array of len {len(parray.poses)}")
                self.pose_pub.publish(parray)
                rospy.loginfo("Finished publishing pose array")
                break
        if viz:
            # Panda Gripper
            mlab.figure(bgcolor=(1, 1, 1))
            draw_scene(
                pc_to_network,
                grasps=gen_grasps,
                grasp_scores=gen_scores,
                # show_gripper_mesh=True,
                # gripper='panda'
            )
            # Fetch Gripper 
            # mlab.figure(bgcolor=(1, 1, 1))
            # draw_scene(
            #     points_viz,
            #     pc_color=None,
            #     grasps=sorted_graps_fetch,
            #     grasp_scores=[gen_scores[i] for i in sort_index],
            #     show_gripper_mesh=True,
            #     gripper='fetch_real_world'
            # )
            mlab.show() # show after publishing
        print("Returning from run_network")


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

    parser.add_argument("--npy_folder", type=str, default="demo/data/")
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

    parser.add_argument("--viz", action="store_true", help="flag for vizualizing the generated grasps")
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
    # listener = ImgToGraspPubSub(estimator)
    listener = PointToGraspPubSub(estimator)
    
    while not rospy.is_shutdown():
        try:
            listener.run_network(viz=args.viz)
        except KeyboardInterrupt:
            break

    # listener = GraspPublisher(estimator) 
    # flag_break = False
    # while (not flag_break) and (not rospy.is_shutdown()):
    #     for npy_file in glob.glob(os.path.join(args.npy_folder, '*.npy')):
    #         data = np.load(npy_file, allow_pickle=True, encoding="latin1").item()
    #         object_pc = data['smoothed_object_pc']
    #         try:
    #             listener.run_network(object_pc)
    #         except KeyboardInterrupt:
    #             flag_break = True
    #             break
    
    print("Exiting 6dof-pose generation ros node")

