"""
    For panda (two-finger) gripper: pushing, pushing-left, pushing-up, pulling, pulling-left, pulling-up
        50% all parts closed, 50% middle (for each part, 50% prob. closed, 50% prob. middle)
        Simulate until static before starting
"""

import os
import sys
import shutil
import numpy as np
from PIL import Image
import cv2
import json
from argparse import ArgumentParser
from utils import control_joints_to_target, get_robot_ee_pose, save_h5, create_orthogonal_vectors, ContactError, length_to_plane, move_to_target_positions_array
from sapien.core import Pose
from env_robotiq import Env
from camera import ornshowAxes, Camera, CameraIntrinsic, update_camera_image_to_base, point_cloud_flter, camera_setup
import pyvista as pv
import pcl
import pcl.pcl_visualization
import os
import time
import pdb
import math
import pybullet as p
import pybullet_data
from collections import namedtuple
from attrdict import AttrDict
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
sys.path.append('../')
from pybullet_planning import get_joint_limits, get_max_velocity, get_max_force, get_link_pose
from pybullet_planning import get_movable_joints, set_joint_positions, get_joint_positions, disconnect

'''
defaut body id:
ground -> 1
object -> 2
robot  -> 3
'''
parser = ArgumentParser()
parser.add_argument('category', type=str) # StorageFurniture
parser.add_argument('--out_dir', type=str)
parser.add_argument('--trial_id', type=int, default=0, help='trial id')
parser.add_argument('--random_seed', type=int, default=None)
parser.add_argument('--no_gui', action='store_true', default=False, help='no_gui [default: False]')
args = parser.parse_args()

HERE = os.path.dirname(__file__)
ROBOT_URDF = os.path.join(HERE, 'data', 'robotiq_85', 'urdf', 'ur5_robotiq_85.urdf')
OBJECT_URDF = os.path.join(HERE, 'datasets', 'grasp', '%s', 'model.urdf') % args.category
# OBJECT_URDF = os.path.join(HERE, 'datasets', 'dataset_single', '%s', 'model.sdf') % args.category
# OBJECT_URDF = os.path.join(HERE, '../../../../', 'media/zhou/软件/博士/具身/data_small', '%s', 'model.sdf') % args.category

robotStartPos0 = [0, 0, 0.2]
robotStartPos1 = [0.1, 0, 0.4]
robotStartPos2 = [0.1, 0, 0.6]

# gripper_main_control_joint_name = ["left_outer_knuckle_joint",
#                                 "left_inner_knuckle_joint",
#                                 "right_outer_knuckle_joint",
#                                 "right_inner_knuckle_joint",
#                                 ]
# mimic_joint_name = ["left_inner_finger_joint",
#                     "right_inner_finger_joint",
#                     ]
# gripper_main_control_joint_name = ["robotiq_2f_85_right_driver_joint",
#                                 "robotiq_2f_85_left_driver_joint",
#                                 "robotiq_2f_85_right_spring_link_joint",
#                                 "robotiq_2f_85_left_spring_link_joint",
#                                 ]

# mimic_joint_name = ["robotiq_2f_85_left_follower_joint",
#                     "robotiq_2f_85_right_follower_joint",
#                     ]

gripper_main_control_joint_name = ["right_outer_knuckle_joint",
                                "left_inner_knuckle_joint",
                                "right_inner_knuckle_joint",
                                "left_inner_finger_joint",
                                "right_inner_finger_joint"
                                ]

mimic_joint_name = ["finger_joint",
                    ]


jointInfo = namedtuple("jointInfo",
                       ["id","name","type","lowerLimit","upperLimit","maxForce","maxVelocity"])

joints = AttrDict()
print("start collect data")
trial_id = args.trial_id
if args.no_gui:
    out_dir = os.path.join(args.out_dir, '%s_%d' % (args.category, trial_id))
else:
    out_dir = os.path.join('datasets', 'results', '%s_%d' % (args.category, trial_id))
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
# print(out_dir)
os.makedirs(out_dir)
flog = open(os.path.join(out_dir, 'log.txt'), 'w')
out_info = dict() # 创建一个空字典

# set random seed
if args.random_seed is not None:
    np.random.seed(args.random_seed)
    out_info['random_seed'] = args.random_seed

dist = 0.5
camera_config = "setup.json"

theta, phi, pose, config = camera_setup(camera_config, dist)
relative_offset_pose = pose

camera_intrinsic = CameraIntrinsic.from_dict(config["intrinsic"])  # 相机内参数据
# setup camera
cam = Camera(camera_intrinsic, dist=0.5, phi=phi, theta=theta, fixed_position=False)

# setup env
env = Env()

# load shape
flog.write('object_urdf_fn: %s\n' % OBJECT_URDF)
state = 'random-closed-middle'
if np.random.random() < 0.5:
    state = 'closed'
flog.write('Object State: %s\n' % state)
out_info['object_state'] = state
objectID = env.load_object(OBJECT_URDF)
objectLinkid = -1
# simulate some steps for the object to stay rest
still_timesteps = 0
wait_timesteps = 0
while still_timesteps < 1000:
    env.step()
    still_timesteps += 1

rgb, depth, pc, cwT = update_camera_image_to_base(relative_offset_pose, cam)
out_info['camera_metadata'] = cam.get_metadata_json()

cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts = point_cloud_flter(pc, depth)
### use the GT vision
# rgb, depth, _ = cam.shot()
Image.fromarray((rgb).astype(np.uint8)).save(os.path.join(out_dir, 'rgb.png'))

# 根据深度图（depth）和相机的内参矩阵来计算相机坐标系中的三维点
''' show
pv.plot(
    cam_XYZA_pts,
    scalars=cam_XYZA_pts[:, 2],
    render_points_as_spheres=True,
    point_size=5,
    show_scalar_bar=False,
)
'''
positive_mask = cam_XYZA_pts > 0  # 创建布尔掩码
positive_numbers = cam_XYZA_pts[positive_mask] # 选择正数元素

cloud = pcl.PointCloud(cam_XYZA_pts.astype(np.float32))
# 创建SAC-IA分割对象
seg = cloud.make_segmenter()
seg.set_optimize_coefficients(True)
seg.set_model_type(pcl.SACMODEL_PLANE)
seg.set_method_type(pcl.SAC_RANSAC)
seg.set_distance_threshold(0.02)
# 执行分割
inliers, coefficients = seg.segment()
# 获取地面点云和非地面点云
ground_points = cloud.extract(inliers, negative=False)
non_ground_points = cloud.extract(inliers, negative=True)
# 转换为array
cam_XYZA_filter_pts = non_ground_points.to_array()
''' show
pv.plot(
    cam_XYZA_filter_pts,
    scalars=cam_XYZA_filter_pts[:, 2],
    render_points_as_spheres=True,
    point_size=5,
    show_scalar_bar=False,
)
'''
positive_mask = cam_XYZA_filter_pts > 0  # 创建布尔掩码
positive_numbers = cam_XYZA_filter_pts[positive_mask] # 选择正数元素

cam_XYZA_pts_tmp = np.array(cam_XYZA_pts).astype(np.float32)
cam_XYZA_filter_pts_tem = np.array(cam_XYZA_filter_pts).astype(np.float32)

index_inliers_set = set(inliers)
cam_XYZA_filter_idx = []
cam_XYZA_pts_idx = np.arange(cam_XYZA_pts.shape[0])
for idx in range(len(cam_XYZA_pts_idx)):
    if idx not in index_inliers_set:
        cam_XYZA_filter_idx.append(cam_XYZA_pts_idx[idx])
cam_XYZA_filter_idx = np.array(cam_XYZA_filter_idx)
cam_XYZA_filter_idx = cam_XYZA_filter_idx.astype(int)
cam_XYZA_filter_id1 = cam_XYZA_id1[cam_XYZA_filter_idx]
cam_XYZA_filter_id2 = cam_XYZA_id2[cam_XYZA_filter_idx]

# 将计算出的三维点信息组织成一个矩阵格式。
cam_XYZA = cam.compute_XYZA_matrix(cam_XYZA_filter_id1, cam_XYZA_filter_id2, cam_XYZA_filter_pts, depth.shape[0], depth.shape[1])
save_h5(os.path.join(out_dir, 'cam_XYZA.h5'), \
        [(cam_XYZA_filter_id1.astype(np.uint64), 'id1', 'uint64'), \
         (cam_XYZA_filter_id2.astype(np.uint64), 'id2', 'uint64'), \
         (cam_XYZA_filter_pts.astype(np.float32), 'pc', 'float32'), \
        ])

gt_nor = cam.get_normal_map(relative_offset_pose, cam, cwT)[0]
Image.fromarray(((gt_nor)*255).astype(np.uint8)).save(os.path.join(out_dir, 'gt_nor.png'))

object_link_ids = env.movable_link_ids

gt_movable_link_mask = cam.get_grasp_regien_mask(cam_XYZA_filter_id1, cam_XYZA_filter_id2, depth.shape[0], depth.shape[1]) # gt_movable_link_mask 表示为：像素图中可抓取link对应其link_id
Image.fromarray((gt_movable_link_mask>0).astype(np.uint8)*255).save(os.path.join(out_dir, 'interaction_mask.png')) # 将gt_movable_link_mask转为二值图进行保存

# sample a pixel to interact
# object_mask = cam.get_object_mask()
xs, ys = np.where(gt_movable_link_mask==1)
if len(xs) == 0:
    flog.write('No Movable Pixel! Quit!\n')
    flog.close()
    env.close()
    exit(1)
idx = np.random.randint(len(xs)) # sample interaction pixels random
x, y = xs[idx], ys[idx]
out_info['pixel_locs'] = [int(x), int(y)] # 采样到的像素位置
# 随机设置一个可移动关节作为 actor_id
env.set_target_object_part_actor_id(object_link_ids[gt_movable_link_mask[x, y]-1], custom=False) # [gt_movable_link_mask[x, y]-1] represent pixel coordinate(x,y) correspond to movable link id
out_info['target_object_part_actor_id'] = env.target_object_part_actor_id
out_info['target_object_part_joint_id'] = env.target_object_part_joint_id

# get pixel 3D pulling direction (cam/world)
# direction_cam = gt_nor[x, y, :3]

idx_ = np.random.randint(cam_XYZA_filter_pts.shape[0])
x, y = cam_XYZA_filter_id1[idx_], cam_XYZA_filter_id2[idx_]
# direction_cam = normalpoint[idx_][:3]
direction_cam = gt_nor[x, y, :3]

direction_cam /= np.linalg.norm(direction_cam)
out_info['direction_camera'] = direction_cam.tolist()
flog.write('Direction Camera: %f %f %f\n' % (direction_cam[0], direction_cam[1], direction_cam[2]))

# get pixel 3D position (cam/world)
position_world_xyz1 = cam_XYZA[x, y, :3]
position_world = position_world_xyz1[:3]
# p.addUserDebugLine(position_world, position_world + direction_cam*1, [1, 0, 0])
# p.addUserDebugText(str("gt_nor"), position_world, [1, 0, 0])

# direction_world = cwT[:3, :3] @ direction_cam
# out_info['direction_world'] = direction_world.tolist()
# flog.write('Direction World: %f %f %f\n' % (direction_world[0], direction_world[1], direction_world[2]))
flog.write('mat44: %s\n' % str(cwT))

# sample a random direction in the hemisphere (cam/world)
action_direction_cam = np.random.randn(3).astype(np.float32)
action_direction_cam /= np.linalg.norm(action_direction_cam)
if action_direction_cam @ direction_cam > 0: # 两个向量的夹角小于90度
    action_direction_cam = -action_direction_cam
out_info['gripper_direction_camera'] = action_direction_cam.tolist() # position p
action_direction_world = action_direction_cam
out_info['gripper_direction_world'] = action_direction_world.tolist()
env.action_direction_world = action_direction_world

# compute final pose
# 初始化 gripper坐标系，默认gripper正方向朝向-z轴
robotStartOrn = p.getQuaternionFromEuler([0, 0, 0])
# gripper坐标系绕y轴旋转-pi/2, 使其正方向朝向+x轴
robotStartOrn1 = p.getQuaternionFromEuler([0, -np.pi/2, 0])
robotStartrot3x3 = R.from_quat(robotStartOrn).as_matrix()
robotStart2rot3x3 = R.from_quat(robotStartOrn1).as_matrix()
# gripper坐标变换
basegrippermatZTX = robotStartrot3x3@robotStart2rot3x3
robotStartOrn2 = R.from_matrix(basegrippermatZTX).as_quat()

# 建立gripper朝向向量relative_offset，[0，0，1]为+z轴方向，由于默认gripper正方向朝向-z轴，所以x轴为-relative_offset
relative_offset = np.array(action_direction_world)
p.addUserDebugLine(robotStartPos2, robotStartPos2 + relative_offset*1, [0, 1, 0])
p.addUserDebugText(str("action_direction_world"), robotStartPos2, [0, 1, 0])

# 以 -relative_offset 为x轴建立正交坐标系
forward, up, left = create_orthogonal_vectors(-relative_offset)
fg = np.vstack([forward, up, left]).T
robotStartOrnfg = R.from_matrix(fg).as_quat()
# print("res: ", np.cross(fg[:, 0], relative_offset))
out_info['gripper_forward_direction_camera'] = up.tolist()

# gripper坐标变换
basegrippermatT = fg@basegrippermatZTX
robotStartOrn3 = R.from_matrix(basegrippermatT).as_quat()
theta_x, theta_y, theta_z = p.getEulerFromQuaternion(robotStartOrn3)
ornshowAxes(robotStartPos2, robotStartOrn3)
# print("res: ", np.cross(basegrippermatT[:, 2], relative_offset))

rotmat = np.eye(4).astype(np.float32) # 旋转矩阵
rotmat[:3, :3] = basegrippermatT

h = length_to_plane(position_world, action_direction_world, plane_height=0.05)
if h > 0.05:
    d_gsp = 0.13
else:
    d_gsp = 0.15 - h
# final_dist = 0.13 # ur5 grasp
final_dist = d_gsp
### main steps
out_info['grasp_width'] = final_dist


final_rotmat = np.array(rotmat, dtype=np.float32)
final_rotmat[:3, 3] = position_world - action_direction_world * final_dist # 以齐次坐标形式添加 平移向量
final_pose = Pose().from_transformation_matrix(final_rotmat) # 变换矩阵转位置和旋转（四元数）
out_info['target_rotmat_world'] = final_rotmat.tolist()
p.addUserDebugPoints([[position_world[0], position_world[1], position_world[2]]], [[0, 1, 0]], pointSize=8)

p.addUserDebugPoints([[final_rotmat[:3, 3][0], final_rotmat[:3, 3][1], final_rotmat[:3, 3][2]]], [[0, 0, 1]], pointSize=8)
p.addUserDebugText(str("final_pose"), final_pose.p, [0, 0, 1])

start_rotmat = np.array(rotmat, dtype=np.float32)
# start_rotmat[:3, 3] = position_world - action_direction_world * 0.2 # 以齐次坐标形式添加 平移向量  ur5 grasp
start_rotmat[:3, 3] = position_world - action_direction_world * 0.38 # 以齐次坐标形式添加 平移向量
start_pose = Pose().from_transformation_matrix(start_rotmat) # 变换矩阵转位置和旋转（四元数）
out_info['start_rotmat_world'] = start_rotmat.tolist()
# ornshowAxes(start_pose.p, start_pose.q)
p.addUserDebugPoints([[start_rotmat[:3, 3][0], start_rotmat[:3, 3][1], start_rotmat[:3, 3][2]]], [[1, 0, 0]], pointSize=8)
p.addUserDebugText(str("start_pose"), start_pose.p, [0, 0, 1])

action_direction = None

if action_direction is not None:
    end_rotmat = np.array(rotmat, dtype=np.float32)
    end_rotmat[:3, 3] = position_world - action_direction_world * final_dist + action_direction * 0.05
    out_info['end_rotmat_world'] = end_rotmat.tolist()

robotID = env.load_robot(ROBOT_URDF, start_pose.p, robotStartOrn3)
env.set_friction(robotID)

pose, orie = get_robot_ee_pose(robotID, 5)
theta_x, theta_y, theta_z = p.getEulerFromQuaternion(orie)
ornshowAxes(pose, orie)

# move back
# activate contact checking
controlJoints = gripper_main_control_joint_name + mimic_joint_name
gripperJointsInfo = env.setup_gripper(env.robotID, controlJoints)
env.start_checking_contact(env.gripper_actor_ids, env.hand_actor_id)

### main steps
out_info['start_target_part_qpos'] = env.get_target_part_qpos().tolist()

target_link_mat44 = np.eye(4)
movable_joints = get_movable_joints(robotID)
min_limits = [get_joint_limits(robotID, joint)[0] for joint in movable_joints]
max_limits = [get_joint_limits(robotID, joint)[1] for joint in movable_joints]
max_velocities = [get_max_velocity(robotID, joint) for joint in movable_joints] # Range of Jacobian
current_conf = get_joint_positions(robotID, movable_joints)

robotStartOrn = p.getQuaternionFromEuler([0, 0, np.pi/2])
robotStartrot3x3 = R.from_quat(robotStartOrn).as_matrix()
custom_matrix = np.eye(4)
custom_matrix[:3, :3] = robotStartrot3x3
custom_matrix[:3, 3] = robotStartPos2
finaljointPose = p.calculateInverseKinematics(robotID, 5, final_pose.p, robotStartOrn3, lowerLimits=min_limits,
                                         upperLimits=max_limits, jointRanges=max_velocities, restPoses=current_conf)

target_link_start_pose = get_link_pose(objectID, objectLinkid) # 得到世界坐标系下物体Link的位姿
env.object_start_pose = target_link_start_pose

success = True
try:
    # env.open_gripper(env.robotID, env.joints, gripper_main_control_joint_name, mimic_joint_name, 0, 0)
    env.open_gripper2()
    env.wait_n_steps(300)

    # approach
    joint_indices = [0,1,2,3,4,5,6,7,8,9,10,11]
    # numJoints = p.getNumJoints(robotID)
    # joint_indices = [j for j in range(5) if p.getJointInfo(robotID, j)[2] == p.JOINT_REVOLUTE]
    # move_to_target_positions_array(robotID, env, joint_indices, finaljointPose, slow_down_rate = 0.01)
    control_joints_to_target(env, robotID, finaljointPose, env.numJoints, close_gripper = True)
    # print("move to start pose end")
    # move to the final pose
    rgb_final_pose, depth, _, _ = update_camera_image_to_base(relative_offset_pose, cam, cwT)

    rgb_final_pose = cv2.circle(rgb_final_pose, (y, x), radius=2, color=(255, 0, 3), thickness=5)
    Image.fromarray((rgb_final_pose).astype(np.uint8)).save(os.path.join(out_dir, 'viz_target_pose.png'))

    env.wait_n_steps(500)

    ##### 计算位置变化 ####
    target_link_pose = get_link_pose(objectID, objectLinkid) # 得到世界坐标系下物体Link的位姿
    mov_dir = np.array(target_link_pose[0][:2], dtype=np.float32) - \
            np.array([0,0], dtype=np.float32)
    mov_dir = np.linalg.norm(mov_dir, ord=2)
    # print("mov_dir", mov_dir)
    if mov_dir > 0.13:
        success = False
        print("move start contact: ", mov_dir)
        raise ContactError
    env.close_gripper2()
    # env.close_gripper(env.robotID, env.joints, gripper_main_control_joint_name, mimic_joint_name, 0.7, -0.6, controlMode=0)
  
    startjointPose = p.calculateInverseKinematics(robotID, 5, start_pose.p, robotStartOrn3, lowerLimits=min_limits,
                                         upperLimits=max_limits, jointRanges=max_velocities, restPoses=current_conf)

#   activate contact checking
    # print("move end")
    env.end_checking_contact()
    move_to_target_positions_array(env, robotID, joint_indices, startjointPose, slow_down_rate = 0.01)
    # control_joints_to_target(env, robotID, startjointPose, env.numJoints)
    # env.wait_n_steps(500)
    # print("move finish")
    
except ContactError:
    success = False

target_link_end_pose = get_link_pose(objectID, objectLinkid) # 得到世界坐标系下物体Link的位姿
flog.write('touch_position_world_xyz_start: %s\n' % str(target_link_start_pose))
flog.write('touch_position_world_xyz_end: %s\n' % str(target_link_end_pose))
out_info['touch_position_world_xyz_start'] = np.array(target_link_start_pose[0]).tolist()
out_info['touch_position_world_xyz_end'] = np.array(target_link_end_pose[0]).tolist()

if success:
    out_info['result'] = 'VALID'
    out_info['final_target_part_qpos'] = env.get_target_part_qpos().tolist()
else:
    out_info['result'] = 'CONTACT_ERROR'

# # save results
with open(os.path.join(out_dir, 'result.json'), 'w') as fout:
    json.dump(out_info, fout)

# #close the file
flog.close()

if args.no_gui:
    # close env
    disconnect()
else:
    if success:
        print('[Successful Interaction] Done. Ctrl-C to quit.')
        ### wait forever
        # env.wait_n_steps(500)
        disconnect()
    else:
        print('[Unsuccessful Interaction] invalid gripper-object contact.')
        # close env
        disconnect()

