"""
    For panda (two-finger) gripper: pushing, pushing-left, pushing-up, pulling, pulling-left, pulling-up
        50% all parts closed, 50% middle (for each part, 50% prob. closed, 50% prob. middle)
        Simulate until static before starting
    
    RECOLLECT
        Pick src_data_dir shape, part states and image
        Given interaction <X, Y>, dirs1, dirs2
"""

import os
import sys
import shutil
import numpy as np
from PIL import Image
from utils import control_joints_to_target, get_robot_ee_pose, save_h5, length_to_plane, ContactError
import cv2
import json
from argparse import ArgumentParser
from camera import ornshowAxes, Camera, CameraIntrinsic, update_camera_image_to_base, point_cloud_flter
from env_kinova import Env
from sapien.core import Pose
from env_kinova import Env
from collections import namedtuple
import pyvista as pv
import pcl
import pcl.pcl_visualization
from subprocess import call
import pybullet as p
from scipy.spatial.transform import Rotation as R
sys.path.append('../')
from pybullet_planning import get_joint_limits, get_max_velocity, get_max_force, get_link_pose
from pybullet_planning import get_movable_joints, set_joint_positions, get_joint_positions, disconnect

parser = ArgumentParser()
parser.add_argument('src_data_dir', type=str)
parser.add_argument('record_name', type=str)
parser.add_argument('tar_data_dir', type=str)
parser.add_argument('--random_seed', type=int, default=None)
parser.add_argument('--no_gui', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--x', type=int)
parser.add_argument('--y', type=int)
parser.add_argument('--dir1', type=str)
parser.add_argument('--dir2', type=str)
args = parser.parse_args()
print("record_name", args.record_name)

category, trial_id = args.record_name.rsplit('_', 1)
HERE = os.path.dirname(__file__)
ROBOT_URDF = os.path.join(HERE, 'data', 'kinova_j2s7s300', 'urdf', 'j2s7s300.urdf')
OBJECT_URDF = os.path.join(HERE, 'datasets', 'grasp', '%s', 'model.urdf') % category

gripper_main_control_joint_name = ["j2s7s300_joint_finger_1",
                    "j2s7s300_joint_finger_2",
                    "j2s7s300_joint_finger_3",
                    ]

mimic_joint_name = ["j2s7s300_joint_finger_tip_1",
                    "j2s7s300_joint_finger_tip_2",
                    "j2s7s300_joint_finger_tip_3",
                    ]
jointInfo = namedtuple("jointInfo",
                       ["id","name","type","lowerLimit","upperLimit","maxForce","maxVelocity"])

robotStartPos0 = [0, 0, 0.2]
robotStartPos1 = [0.1, 0, 0.4]
robotStartPos2 = [0.1, 0, 0.6]

if args.no_gui:
    out_dir = os.path.join(args.tar_data_dir, '%s_%s' % (category, trial_id))
else: # ../data/gt_data-train_10cats_train_data-pushing/
    out_dir = os.path.join(args.tar_data_dir, '%s_%d' % (category, (int(trial_id)+1)))
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
os.mkdir(out_dir)
flog = open(os.path.join(out_dir, 'log.txt'), 'w')
out_info = dict()

# set random seed
if args.random_seed is not None:
    np.random.seed(args.random_seed)
    out_info['random_seed'] = args.random_seed

# load old-data result.json
with open(os.path.join(args.src_data_dir, args.record_name, 'result.json'), 'r') as fin:
    replay_data = json.load(fin)

camera_config = "setup.json"
with open(camera_config, "r") as j:
    config = json.load(j)

camera_intrinsic = CameraIntrinsic.from_dict(config["intrinsic"])  # 相机内参数据

# setup camera
cam_theta = replay_data['camera_metadata']['theta']
cam_phi = replay_data['camera_metadata']['phi']

dist = 0.5
pose = np.array([dist*np.cos(cam_phi)*np.cos(cam_theta), \
        dist*np.cos(cam_phi)*np.sin(cam_theta), \
        dist*np.sin(cam_phi)])
relative_offset_pose = pose
cam = Camera(camera_intrinsic, dist=0.5, phi=cam_phi, theta=cam_theta, fixed_position=False)

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

### use the GT vision
rgb, depth, pc, cwT = update_camera_image_to_base(relative_offset_pose, cam)
out_info['camera_metadata'] = cam.get_metadata_json()
cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts = point_cloud_flter(pc, depth)

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

gt_nor = cam.get_normal_map(relative_offset_pose, cam)[0]
Image.fromarray(((gt_nor+1)/2*255).astype(np.uint8)).save(os.path.join(out_dir, 'gt_nor.png'))

object_link_ids = env.movable_link_ids

gt_movable_link_mask = cam.get_grasp_regien_mask(cam_XYZA_filter_id1, cam_XYZA_filter_id2, depth.shape[0], depth.shape[1]) # gt_movable_link_mask 表示为：像素图中可抓取link对应其link_id
Image.fromarray((gt_movable_link_mask>0).astype(np.uint8)*255).save(os.path.join(out_dir, 'interaction_mask.png')) # 将gt_movable_link_mask转为二值图进行保存

x, y = args.x, args.y
# direction_cam = normalpoint[idx_][:3]
out_info['pixel_locs'] = [int(x), int(y)] # 采样到的像素位置
direction_cam = gt_nor[x, y, :3]

direction_cam /= np.linalg.norm(direction_cam)
out_info['direction_camera'] = direction_cam.tolist()
flog.write('Direction Camera: %f %f %f\n' % (direction_cam[0], direction_cam[1], direction_cam[2]))

# get pixel 3D position (cam/world)
position_world_xyz1 = cam_XYZA[x, y, :3]
position_world = position_world_xyz1[:3]

# use dir1, dir2
action_direction_cam = np.array([float(elem) for elem in args.dir1.split(',')[1:]], dtype=np.float32)
action_forward_direction_cam = np.array([float(elem) for elem in args.dir2.split(',')[1:]], dtype=np.float32)

action_direction_world = action_forward_direction_cam
out_info['gripper_direction_camera'] = action_direction_cam.tolist() # position p
out_info['gripper_direction_world'] = action_direction_world.tolist()

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

# 计算朝向坐标
forward = np.array(action_direction_world, dtype=np.float32)
up = np.array(action_direction_cam, dtype=np.float32)
left = np.cross(forward, up)
left /= np.linalg.norm(left)

up = np.cross(left, forward)
up /= np.linalg.norm(up)
fg = np.vstack([forward, up, left]).T
out_info['gripper_forward_direction_camera'] = up.tolist()

# gripper坐标变换
basegrippermatT = fg@basegrippermatZTX
robotStartOrn3 = R.from_matrix(basegrippermatT).as_quat()
ornshowAxes(robotStartPos2, robotStartOrn3)

rotmat = np.eye(4).astype(np.float32) # 旋转矩阵
rotmat[:3, :3] = basegrippermatT

h = length_to_plane(position_world, action_direction_world, plane_height=0.05)
if h > 0.05:
    d_gsp = 0.17
else:
    d_gsp = 0.19 - h
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

success = True
try:
    env.open_gripper(env.robotID, env.joints, gripper_main_control_joint_name, mimic_joint_name, 0, 0)
    env.wait_n_steps(1000)

    # approach
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

    env.close_gripper(env.robotID, env.joints, gripper_main_control_joint_name, mimic_joint_name, 1, 0.7)
  
    startjointPose = p.calculateInverseKinematics(robotID, 5, start_pose.p, robotStartOrn3, lowerLimits=min_limits,
                                         upperLimits=max_limits, jointRanges=max_velocities, restPoses=current_conf)


#   activate contact checking
    # print("move end")
    env.end_checking_contact()
    control_joints_to_target(env, robotID, startjointPose, env.numJoints, close_gripper = True)
    env.wait_n_steps(500)
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
        env.wait_n_steps(500)
        disconnect()
    else:
        print('[Unsuccessful Interaction] invalid gripper-object contact.')
        # close env
        disconnect()
