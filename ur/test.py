import cv2
from realsense import RealSenseCamera
import sys
import pyrealsense2 as rs
import open3d as o3d

def main(pt, conf, nc, input):
    camera = RealSenseCamera()
    camera.start_camera()

    while True:
        '''相机读取RGB+D流'''
        color_img, depth_img, _, point_cloud, depth_frame = camera.read_align_frame()
        # depth_img = (depth_img / 255).astype(np.uint8)

        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)


        '''获取相机内参'''
        intrinsics = rs.video_stream_profile(depth_frame.profile).get_intrinsics()

        pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(intrinsics.width, intrinsics.height,
                                                                     intrinsics.fx, intrinsics.fy,
                                                                     intrinsics.ppx, intrinsics.ppy)
     
        cv2.imshow("annotated_frame", color_img)


if __name__ == '__main__':
    pt = True
    conf = 0.25
    # nc = 80
    # input = "remote"
    nc = 4
    # input = "Explosive Base"
    input = "Bolt"
    # input = "Rivet Nut"
    # input = "Nut"
    # input = "Gear"
    # input = "Base Plate"
    main(pt, conf, nc, input)
