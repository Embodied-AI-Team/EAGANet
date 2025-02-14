import os
import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
import pyvista as pv
import sys
external_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) 
sys.path.append(external_dir) 
from camera import get_rs_pc, camera_setup, CameraIntrinsic, point_cloud_flter, ground_points_seg

def depth_to_pointcloud(depth_image, intrinsic):
    # Create Open3D Image from depth map
    o3d_depth = o3d.geometry.Image(depth_image)

    # Get intrinsic parameters
    fx, fy, cx, cy = intrinsic.fx, intrinsic.fy, intrinsic.ppx, intrinsic.ppy

    # Create Open3D PinholeCameraIntrinsic object
    o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(width=depth_image.shape[1], height=depth_image.shape[0], fx=fx,
                                                      fy=fy, cx=cx, cy=cy)

    # Create Open3D PointCloud object from depth image and intrinsic parameters
    pcd = o3d.geometry.PointCloud.create_from_depth_image(o3d_depth, o3d_intrinsic)

    return pcd


def save_pointcloud(pcd, file_name):
    o3d.io.write_point_cloud(file_name, pcd)


# 初始化 RealSense 相机
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
profile = pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)

cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()

if not os.path.exists('data'):
    os.makedirs('data')

subfolders = ['images', 'depths', 'point_clouds']
for folder in subfolders:
    folder_path = os.path.join('data', folder)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

counter = 0

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        if not aligned_depth_frame:
            continue

        depth_frame = frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not aligned_depth_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        # color_image1 = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR) 


        # 利用中值核进行滤波
        depth_frame = rs.decimation_filter(1).process(depth_frame)
        # 从深度表示转换为视差表示，反之亦然
        depth_frame = rs.disparity_transform(True).process(depth_frame)
        # 空间滤镜通过使用alpha和delta设置计算帧来平滑图像。
        depth_frame = rs.spatial_filter().process(depth_frame)
        # 时间滤镜通过使用alpha和delta设置计算多个帧来平滑图像。
        depth_frame = rs.temporal_filter().process(depth_frame)
        # 从视差表示转换为深度表示
        depth_frame = rs.disparity_transform(False).process(depth_frame)
        # depth_frame = rs.hole_filling_filter().process(depth_frame)

        # 将深度图转化为RGB准备显示
        depth_color_frame = rs.colorizer().colorize(depth_frame)
        depth_color_image = np.asanyarray(depth_color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        

        color_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
        depth_intrinsics = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
        camera_config = "./setup.json"
        _, _, _, config = camera_setup(camera_config)

        camera_intrinsic = CameraIntrinsic.from_dict(config["intrinsic"])  # 相机内参数据
        # pc = depth_to_pointcloud(depth_image, depth_intrinsics)

        cv2.imshow('RealSense', color_image)

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.008), cv2.COLORMAP_JET)
        cv2.imshow('depth_color', depth_colormap)

        # 检查是否按下了 's' 键，如果按下了，就保存当前帧的 RGB、深度图和点云
        key = cv2.waitKey(1)
        if key == ord('s'):
            pc = get_rs_pc(color_image, depth_image, camera_intrinsic, 640, 480)

            # 保存 RGB 图像
            rgb_file_path = os.path.join('images', 'add_{:04d}.jpg'.format(counter))
            cv2.imwrite(rgb_file_path, color_image)
            print('color saved', rgb_file_path)

            # 保存深度图像
            depth_file_path = os.path.join('depths', 'depth_{:04d}.png'.format(counter))
            cv2.imwrite(depth_file_path, depth_image)
            print('depth saved', depth_file_path)

            # 将点云保存为 pcd 文件
            #pcd_file_path = os.path.join('point_clouds', 'point_cloud_{:04d}.pcd'.format(counter))
            #save_pointcloud(pc, pcd_file_path)
            #print('pc saved', pcd_file_path)
            cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts = point_cloud_flter(pc, depth_image)
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

            # 更新计数器
            counter += 1

        # 检查是否按下了 ESC 键，如果按下了，就退出循环
        if key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()