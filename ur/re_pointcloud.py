import os
import sys
import shutil
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import pyrealsense2 as rs
from argparse import ArgumentParser
import json
import pybullet as p
from sapien.core import Pose
from scipy.spatial.transform import Rotation as R
import pyvista as pv
import matplotlib.pyplot as plt
from realsense import RealSenseCamera
external_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) 
sys.path.append(external_dir) 
from camera import Camera, CameraInfo, create_point_cloud_from_depth_image
from camera import rebuild_pointcloud_format, piontcloud_preprocess, CameraIntrinsic, CameraExtrinsic, point_cloud_flter, ground_points_seg, camera_setup
from utils import length_to_plane, get_model_module, save_h5, create_orthogonal_vectors, ContactError
from open3D_visualizer import Open3D_visualizer

cmap = plt.cm.get_cmap("jet")

parser = ArgumentParser()
parser.add_argument('--exp_name', type=str, help='name of the training run')
parser.add_argument('--model_epoch', type=int, help='epoch')
parser.add_argument('--model_version', type=str, help='model version')
parser.add_argument('--result_suffix', type=str, default='nothing')
parser.add_argument('--device', type=str, default='cuda:0', help='cpu or cuda:x for using cuda on GPU number x')
parser.add_argument('--overwrite', action='store_true', default=False, help='overwrite if result_dir exists [default: False]')
eval_conf = parser.parse_args()

def plot_figure(up, forward, position_world, cwT):
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
    # ornshowAxes(robotStartPos2, robotStartOrn3)

    rotmat = np.eye(4).astype(np.float32) # 旋转矩阵
    rotmat[:3, :3] = basegrippermatT
    start_rotmat = np.array(rotmat, dtype=np.float32)
    # start_rotmat[:3, 3] = position_world - action_direction_world * 0.2 # 以齐次坐标形式添加 平移向量  ur5 grasp
    start_rotmat[:3, 3] = position_world - forward * 0.17 # 以齐次坐标形式添加 平移向量
    start_pose = Pose().from_transformation_matrix(start_rotmat) # 变换矩阵转位置和旋转（四元数）
    # robotID = env.load_robot(ROBOT_URDF, start_pose.p, robotStartOrn3)

    # rgb_final_pose, depth, _, _ = update_camera_image_to_base(relative_offset_pose, cam, cwT)

    # rgb_final_pose = cv2.circle(rgb_final_pose, (y, x), radius=2, color=(255, 0, 3), thickness=5)
    # Image.fromarray((rgb_final_pose).astype(np.uint8)).save(os.path.join(result_dir, 'viz_target_pose.png'))


pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 15)
pipeline.start(config)

profile = pipeline.get_active_profile()
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()

pc = rs.pointcloud()
colorizer = rs.colorizer()

align_to = rs.stream.color
align = rs.align(align_to)

# load train config
train_conf = torch.load(os.path.join('../', 'logs', eval_conf.exp_name, 'conf.pth'))

# set up device
device = torch.device(eval_conf.device)
print(f'Using device: {device}')

# load model
model_def = get_model_module(eval_conf.model_version)
# network = model_def.Network(train_conf.feat_dim)

result_dir = os.path.join('logs', eval_conf.exp_name, f'visu_critic_heatmap-model_epoch_{eval_conf.model_epoch}-{eval_conf.result_suffix}')

# setup camera
dist = 1
camera_config = "setup.json"
# with open(camera_config, "r") as j:
#     config = json.load(j)
theta, phi, _, config = camera_setup(camera_config, dist)
camera_intrinsic = CameraIntrinsic.from_dict(config["intrinsic"])  # 相机内参数据
camera_extrinsic = CameraExtrinsic.from_dict(config["extrinsic"])  # 相机内参数据
cam = Camera(camera_intrinsic, dist=0.5, phi=phi, theta=theta, fixed_position=False)
extrinsic = camera_extrinsic.extrinsic
try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        aligned_depth_frame = aligned_frames.get_depth_frame()
        aligned_color_frame = aligned_frames.get_color_frame()

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        depth_colormap = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())
        color_image = np.asanyarray(aligned_color_frame.get_data())

        images = np.hstack((cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB), depth_colormap))
        cv2.imshow("win", images)

        pts = pc.calculate(aligned_depth_frame)
        pc.map_to(aligned_color_frame)

        if not aligned_depth_frame or not aligned_color_frame:
            continue

        key = cv2.waitKey(1)

        if key == ord("d"):
            min_distance = 1e-6
            v = pts.get_vertices()
            vertices = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
            vertices_1 = np.ones((vertices.shape[0], 1))
            vertices_4 = np.hstack((vertices, vertices_1))

            h, w, _ = color_image.shape
            pointcloud = vertices.reshape(h, w, -1)
            pointcloud_cTw = extrinsic @ vertices_4.T
            pointcloud_cTw = pointcloud_cTw.T[:, :3]
            pointcloud_cTw = pointcloud_cTw.reshape(h, w, -1)

            # cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts = point_cloud_flter(pointcloud, depth_image)
            cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts = point_cloud_flter(pointcloud_cTw, depth_image)
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

            cam_XYZA_filter_id1, cam_XYZA_filter_id2 = rebuild_pointcloud_format(inliers, cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts)
            # 将计算出的三维点信息组织成一个矩阵格式。
            cam_XYZA = cam.compute_XYZA_matrix(cam_XYZA_filter_id1, cam_XYZA_filter_id2, cam_XYZA_filter_pts, depth_image.shape[0], depth_image.shape[1])
            gt_movable_link_mask = cam.get_grasp_regien_mask(cam_XYZA_filter_id1, cam_XYZA_filter_id2, depth_image.shape[0], depth_image.shape[1]) # gt_movable_link_mask 表示为：像素图中可抓取link对应其link_id

            # x, y = 270, 270
            idx_ = np.random.randint(cam_XYZA_filter_pts.shape[0])
            x, y = cam_XYZA_filter_id1[idx_], cam_XYZA_filter_id2[idx_]
            # get pixel 3D position (cam/world)
            position_world_xyz1 = cam_XYZA[x, y, :3]
            position_world = position_world_xyz1[:3]

            pc, pccolors = piontcloud_preprocess(x, y, cam_XYZA, color_image, train_conf, gt_movable_link_mask, device, h=480, w=640)
            
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
                # sample a random direction to query
                gripper_direction_camera = torch.randn(1, 3).to(device)
                gripper_direction_camera = F.normalize(gripper_direction_camera, dim=1)
                gripper_forward_direction_camera = torch.randn(1, 3).to(device)
                gripper_forward_direction_camera = F.normalize(gripper_forward_direction_camera, dim=1)

                up = gripper_direction_camera
                forward = gripper_forward_direction_camera
                # left = torch.cross(forward, up)
                # left = F.normalize(left, dim=1)

                # up = torch.cross(left, forward)
                # up = F.normalize(up, dim=1)

                h = length_to_plane(position_world, gripper_forward_direction_camera[0,:].cpu(), plane_height=0.05)
                if h > 0.05:
                    d_gsp = 0.13
                else:
                    d_gsp = 0.15 - h
                # final_dist = 0.13 # ur5 grasp
                final_dist = d_gsp
                depth = torch.full((train_conf.num_point_per_shape, 1),final_dist).float().to(device)
                # plot_figure(up[0].cpu().numpy(), forward[0].cpu().numpy(), position_world, cwT) # draw all pose

                dirs2 = up.repeat(train_conf.num_point_per_shape, 1)
                dirs1 = forward.repeat(train_conf.num_point_per_shape, 1)

                # infer for all pixels
                with torch.no_grad():
                    input_queries = torch.cat([dirs1, dirs2, depth], dim=1)
                    net = network.critic(feats, input_queries)
                    result = torch.sigmoid(net).cpu().numpy()
                    print("max(result) : ", np.max(result))
                    if np.max(result) > 0.80:
                        # plot_figure(up[0].cpu().numpy(), forward[0].cpu().numpy(), position_world, cwT)

                        grasp_succ = 0
                        # result *= pc_movable

                        fn = os.path.join(result_dir, 'pred')
                        resultcolors = cmap(result)[:, :3]
                        pccolors = pccolors * (1 - np.expand_dims(result, axis=-1)) + resultcolors * np.expand_dims(result, axis=-1)
                        o3dvis = Open3D_visualizer(pc[0].cpu().numpy())
                        o3dvis.add_colors_map(pc[0].cpu().numpy())
                        # utils.export_pts_color_pts(fn,  pc[0].cpu().numpy(), pccolors)
                        # utils.export_pts_color_obj(fn,  pc[0].cpu().numpy(), pccolors)
                        # utils.render_pts_label_png(fn,  pc[0].cpu().numpy(), result)

        if key == ord("q"):
            break
finally:
    pipeline.stop()

