import os

import numpy as np
import pybullet as p
import cv2
import json
from tqdm import tqdm
from env import ClutteredPushGrasp
from utilities import YCBModels, GoogleModels
from camera import ornshowAxes, Camera, CameraIntrinsic, ground_points_seg, update_camera_image, point_cloud_flter, camera_setup, rebuild_pointcloud_format, update_camera_image_to_base, piontcloud_preprocess
from utils import get_ikcal_config, control_joints_to_target
import pyvista as pv
import torch
import torch.nn.functional as F
from PIL import Image
from open3D_visualizer import Open3D_visualizer
from scipy.spatial.transform import Rotation as R
from utils import length_to_plane, get_model_module
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from env_robotiq import Env
from sapien.core import Pose
cmap = plt.cm.get_cmap("jet")

# test parameters
parser = ArgumentParser()
parser.add_argument('--exp_name', type=str, help='name of the training run')
parser.add_argument('--model_epoch', type=int, help='epoch')
parser.add_argument('--model_version', type=str, help='model version')
parser.add_argument('--result_suffix', type=str, default='nothing')
parser.add_argument('--device', type=str, default='cuda:0', help='cpu or cuda:x for using cuda on GPU number x')
parser.add_argument('--overwrite', action='store_true', default=False, help='overwrite if result_dir exists [default: False]')
eval_conf = parser.parse_args()

HERE = os.path.dirname(__file__)
ROBOT_URDF = os.path.join(HERE, '../', 'data', 'robotiq_85', 'urdf', 'ur5_robotiq_85.urdf')
OBJECT_URDF = os.path.join(HERE, 'datasets', 'grasp', 'yellow_cup', 'model.urdf')
robotStartPos2 = [0.1, 0, 0.6]

def heuristic_demo():
    ycb_models = YCBModels(
        os.path.join('./data/ycb', '**', 'textured-decmp.obj'),
    )
    camera = Camera((0, -0.5, 1.5), 0.1, 5, (320, 320), 40)

    env = ClutteredPushGrasp(ycb_models, camera, vis=True, num_objs=5, gripper_type='85')
    p.resetDebugVisualizerCamera(2.0, -270., -60., (0., 0., 0.))
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)  # Shadows on/off

    (rgb, depth, seg) = env.reset()
    step_cnt = 0
    while True:

        h_, w_ = np.unravel_index(depth.argmin(), depth.shape)
        x, y, z = camera.rgbd_2_world(w_, h_, depth[h_, w_])

        p.addUserDebugLine([x, y, 0], [x, y, z], [0, 1, 0])
        p.addUserDebugLine([x, y, z], [x, y, z+0.05], [1, 0, 0])

        (rgb, depth, seg), reward, done, info = env.step((x, y, z), 1, 'grasp')

        print('Step %d, grasp at %.2f,%.2f,%.2f, reward %f, done %s, info %s' %
              (step_cnt, x, y, z, reward, done, info))
        step_cnt += 1
        # time.sleep(3)

def sim_demo():

    device = torch.device(eval_conf.device)
    print(f'Using device: {device}')
    model_def = get_model_module(eval_conf.model_version)
    train_conf = torch.load(os.path.join('../', 'logs', eval_conf.exp_name, 'conf.pth'))

    HERE = os.path.dirname(__file__)
    ycb_models = GoogleModels(
        # os.path.join(HERE, '../datasets/grasp_multi', '**', 'model.urdf') # multi objects load
        #  os.path.join(HERE, '../../../../../', 'media/zhou/软件/博士/具身/data_small', 'YumYum_D3_Liquid', 'model.sdf'), # Lutein
        os.path.join(HERE, '../datasets/grasp', 'plastic_apple', 'model.urdf'),
    )
    env = ClutteredPushGrasp(ycb_models, vis=True, num_objs=5, gripper_type='85') # 0.0, -0.5, 0.8
    end_pose = p.getLinkState(env.robotID, 6)

    dist = 1
    camera_config = "../setup.json"
    theta, phi, _, config = camera_setup(camera_config, dist)
    pose = end_pose
    relative_offset_pose = pose
    print("pose: ", pose)
    camera_intrinsic = CameraIntrinsic.from_dict(config["intrinsic"])  # 相机内参数据
    # setup camera
    camera = Camera(camera_intrinsic, dist=0.5, phi=phi, theta=theta, fixed_position=False)
    rgb, depth, pc, cwT = update_camera_image(relative_offset_pose, camera)

    cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts = point_cloud_flter(pc, depth)
    # ''' show
    pv.plot(
        cam_XYZA_pts,
        scalars=cam_XYZA_pts[:, 2],
        render_points_as_spheres=True,
        point_size=5,
        show_scalar_bar=False,
    )
    # '''
    cam_XYZA_filter_pts, inliers = ground_points_seg(cam_XYZA_pts)
    # ''' show
    pv.plot(
        cam_XYZA_filter_pts,
        scalars=cam_XYZA_filter_pts[:, 2],
        render_points_as_spheres=True,
        point_size=5,
        show_scalar_bar=False,
    )
    # '''
    cam_XYZA_filter_id1, cam_XYZA_filter_id2 = rebuild_pointcloud_format(inliers, cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts)
    # 将计算出的三维点信息组织成一个矩阵格式。
    cam_XYZA = camera.compute_XYZA_matrix(cam_XYZA_filter_id1, cam_XYZA_filter_id2, cam_XYZA_filter_pts, depth.shape[0], depth.shape[1])
    gt_movable_link_mask = camera.get_grasp_regien_mask(cam_XYZA_filter_id1, cam_XYZA_filter_id2, depth.shape[0], depth.shape[1]) # gt_movable_link_mask 表示为：像素图中可抓取link对应其link_id

    # x, y = 270, 270
    idx_ = np.random.randint(cam_XYZA_filter_pts.shape[0])
    x, y = cam_XYZA_filter_id1[idx_], cam_XYZA_filter_id2[idx_]
    # get pixel 3D position (cam/world)
    position_world_xyz1 = cam_XYZA[x, y, :3]
    position_world = position_world_xyz1[:3]

    pc, pccolors = piontcloud_preprocess(x, y, cam_XYZA, rgb, train_conf, gt_movable_link_mask, device)
    # create models
    network = model_def.Network(train_conf.feat_dim)

    # load pretrained model
    print('Loading ckpt from ', os.path.join('logs', eval_conf.exp_name, 'ckpts'), eval_conf.model_epoch)
    data_to_restore = torch.load(os.path.join('../', 'logs', eval_conf.exp_name, 'ckpts', '%d-network.pth' % eval_conf.model_epoch))
    network.load_state_dict(data_to_restore, strict=False)
    print('DONE\n')

    # send to device
    network.to(device)
    # set models to evaluation mode
    network.eval()

    # push through unet
    feats = network.pointnet2(pc.repeat(1, 1, 2))[0].permute(1, 0)    # N x F = 10000 x 128
    # robotID = env.load_robot(ROBOT_URDF, start_pose.p, robotStartOrn3)
    
    grasp_succ = 1
    # sample a random direction to query
    while grasp_succ:
        gripper_direction_camera = torch.randn(1, 3).to(device)
        gripper_direction_camera = F.normalize(gripper_direction_camera, dim=1)
        gripper_forward_direction_camera = torch.randn(1, 3).to(device)
        gripper_forward_direction_camera = F.normalize(gripper_forward_direction_camera, dim=1)

        up = gripper_direction_camera
        forward = gripper_forward_direction_camera

        h = length_to_plane(position_world, gripper_forward_direction_camera[0,:].cpu(), plane_height=0.05)
        if h > 0.05:
            d_gsp = 0.135
        else:
            d_gsp = 0.155 - h
        # final_dist = 0.13 # ur5 grasp
        final_dist = d_gsp
        depth = torch.full((train_conf.num_point_per_shape, 1),final_dist).float().to(device)
        # plot_figure(up[0].cpu().numpy(), forward[0].cpu().numpy(), position_world, robotStartPos2, env)

        dirs2 = up.repeat(train_conf.num_point_per_shape, 1)
        dirs1 = forward.repeat(train_conf.num_point_per_shape, 1)

        # infer for all pixels
        with torch.no_grad():
            input_queries = torch.cat([dirs1, dirs2, depth], dim=1)
            net = network.critic(feats, input_queries)
            result = torch.sigmoid(net).cpu().numpy()
            print("max(result) : ", np.max(result))
            if np.max(result) > 0.95:
                # plot_figure(up[0].cpu().numpy(), forward[0].cpu().numpy(), position_world, robotStartPos2, env)

                grasp_succ = 0
                resultcolors = cmap(result)[:, :3]
                pccolors = pccolors * (1 - np.expand_dims(result, axis=-1)) + resultcolors * np.expand_dims(result, axis=-1)
                o3dvis = Open3D_visualizer(pc[0].cpu().numpy())
                o3dvis.add_colors_map(pccolors)

                # start grasp
                # compute final pose
                # 初始化 gripper坐标系，默认gripper正方向朝向-z轴

                forward = forward.cpu().numpy()
                up = up.cpu().numpy()
                relative_forward = forward
                left = np.cross(relative_forward, up)
                left /= np.linalg.norm(left)

                up = np.cross(left, relative_forward)
                up /= np.linalg.norm(up)
                fg = np.vstack([relative_forward, up, left]).T

                rotmat = np.eye(4).astype(np.float32) # 旋转矩阵
                rotmat[:3, :3] = fg

                # min_limits, max_limits, max_velocities, current_conf = get_ikcal_config(env.robotID)
                robotStartOrn3 = R.from_matrix(fg).as_quat()

                # start pose
                endeffort_pose = p.getLinkState(env.robotID, 7)
                endeffort_pos = endeffort_pose[0]
                
                start_rotmat = np.array(rotmat, dtype=np.float32)
                start_rotmat[:3, 3] = endeffort_pos
                start_pose = Pose().from_transformation_matrix(start_rotmat) 
                # ornshowAxes(start_rotmat[:3, 3], robotStartOrn3)
                # startjointPose = p.calculateInverseKinematics(env.robotID, 6, start_pose.p, robotStartOrn3, lowerLimits=min_limits,
                #                                         upperLimits=max_limits, jointRanges=max_velocities, restPoses=current_conf)
                # env.move_joints_to_target(env.robotID, startjointPose, 6)
                res, _ = env.move_ee((start_pose.p[0], start_pose.p[1], start_pose.p[2], robotStartOrn3))
                
                # final pose
                final_rotmat = np.array(rotmat, dtype=np.float32)
                final_rotmat[:3, 3] = position_world - forward * final_dist
                final_pose = Pose().from_transformation_matrix(final_rotmat) 
                
                # show coordinate
                ornshowAxes(final_rotmat[:3, 3], robotStartOrn3)
                res, _ = env.move_ee((final_pose.p[0], final_pose.p[1], final_pose.p[2], robotStartOrn3))
                env.close_gripper(check_contact=True)

                # start pose1
                start_rotmat = np.array(rotmat, dtype=np.float32)
                start_rotmat[:3, 3] = position_world - forward * 0.38
                start_pose = Pose().from_transformation_matrix(start_rotmat) 
                res, _ = env.move_ee((start_pose.p[0], start_pose.p[1], start_pose.p[2], robotStartOrn3), custom_velocity=1)
                
                # start pose2
                start_rotmat2 = np.array(rotmat, dtype=np.float32)
                start_rotmat2[:3, 3] = [0.5, 0, 1]
                start_pose2 = Pose().from_transformation_matrix(start_rotmat2) 
                res, _ = env.move_ee((start_pose2.p[0], start_pose2.p[1], start_pose2.p[2], robotStartOrn3), custom_velocity=1)
                

                # finaljointPose = p.calculateInverseKinematics(env.robotID, 5, final_pose.p, robotStartOrn3, lowerLimits=min_limits,
                #                                         upperLimits=max_limits, jointRanges=max_velocities, restPoses=current_conf)
                # env.move_ee((finaljointPose, y, z, orn))
                # env.move_joints_to_target(env.robotID, finaljointPose, 6)

    p.resetDebugVisualizerCamera(2.0, -270., -60., (0., 0., 0.))
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)  # Shadows on/off
    p.addUserDebugLine([0, -0.5, 0], [0, -0.5, 1.1], [0, 1, 0])


def user_control_demo():
    ycb_models = YCBModels(
        os.path.join('./data/ycb', '**', 'textured-decmp.obj'),
    )
    camera = Camera((0, -0.5, 1.5), 0.1, 5, (320, 320), 40)

    env = ClutteredPushGrasp(ycb_models, camera, vis=True, num_objs=5, gripper_type='85')
    p.resetDebugVisualizerCamera(2.0, -270., -60., (0., 0., 0.))
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)  # Shadows on/off
    p.addUserDebugLine([0, -0.5, 0], [0, -0.5, 1.1], [0, 1, 0])

    env.reset()
    while True:
        env.step(None, None, None, True)

        # key control
        keys = p.getKeyboardEvents()
        # key "Z" is down and hold
        if (122 in keys) and (keys[122] == 3):
            print('Grasping...')
            if env.close_gripper(check_contact=True):
                print('Grasped!')
        # key R
        if 114 in keys:
            env.open_gripper()
        # time.sleep(1 / 120.)


def plot_figure(up, forward, position_world, robotStartPos2, env):
    # cam to world
    # up = mat33 @ up
    # forward = mat33 @ forward

    # 初始化 gripper坐标系，默认gripper正方向朝向-z轴
    robotStartOrn = p.getQuaternionFromEuler([0, 0, 0])
    # gripper坐标系绕y轴旋转-pi/2, 使其正方向朝向+x轴
    robotStartOrn1 = p.getQuaternionFromEuler([0, -np.pi/2, 0])
    robotStartrot3x3 = R.from_quat(robotStartOrn).as_matrix()
    robotStart2rot3x3 = R.from_quat(robotStartOrn1).as_matrix()
    # gripper坐标变换
    basegrippermatZTX = robotStartrot3x3@robotStart2rot3x3

    relative_forward = -forward
    # 计算朝向坐标
    relative_forward = np.array(relative_forward, dtype=np.float32)
    up = np.array(up, dtype=np.float32)
    left = np.cross(relative_forward, up)
    left /= np.linalg.norm(left)

    up = np.cross(left, relative_forward)
    up /= np.linalg.norm(up)
    fg = np.vstack([relative_forward, up, left]).T

    # gripper坐标变换
    basegrippermatT = fg@basegrippermatZTX
    robotStartOrn3 = R.from_matrix(basegrippermatT).as_quat()
    ornshowAxes(position_world, robotStartOrn3)

    rotmat = np.eye(4).astype(np.float32) # 旋转矩阵
    rotmat[:3, :3] = basegrippermatT
    start_rotmat = np.array(rotmat, dtype=np.float32)
    # start_rotmat[:3, 3] = position_world - action_direction_world * 0.2 # 以齐次坐标形式添加 平移向量  ur5 grasp
    start_rotmat[:3, 3] = position_world - forward * 0.17 # 以齐次坐标形式添加 平移向量
    start_pose = Pose().from_transformation_matrix(start_rotmat) # 变换矩阵转位置和旋转（四元数）
    robotID = env.load_robot(ROBOT_URDF, start_pose.p, robotStartOrn3)

    return basegrippermatT
    # rgb_final_pose, depth, pc, cwT = update_camera_image(relative_offset_pose, camera)

    # rgb_final_pose = cv2.circle(rgb_final_pose, (y, x), radius=2, color=(255, 0, 3), thickness=5)
    # Image.fromarray((rgb_final_pose).astype(np.uint8)).save(os.path.join(result_dir, 'viz_target_pose.png'))


if __name__ == '__main__':
    # user_control_demo()
    # heuristic_demo()
    sim_demo()
