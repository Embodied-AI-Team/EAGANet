"""
    an RGB-D camera
"""
import pybullet as p
import glob
from collections import namedtuple
from attrdict import AttrDict
import functools
import torch
import cv2
from scipy import ndimage
import numpy as np
from PIL import Image
from normal_map import startConvert
from scipy.spatial.transform import Rotation as R
import open3d as o3d
from utils import create_orthogonal_vectors
from pointnet2_ops.pointnet2_utils import furthest_point_sample
import json
import pcl
import pcl.pcl_visualization

class CameraIntrinsic(object):
    """Intrinsic parameters of a pinhole camera model.

    Attributes:
        width (int): The width in pixels of the camera.
        height(int): The height in pixels of the camera.
        K: The intrinsic camera matrix.
    """

    def __init__(self, width, height, fx, fy, cx, cy):
        self.width = width
        self.height = height
        self.K = np.array(
            [[fx, 0.0, cx],
             [0.0, fy, cy],
             [0.0, 0.0, 1.0]]
        )

    @property
    def fx(self):
        return self.K[0, 0]

    @property
    def fy(self):
        return self.K[1, 1]

    @property
    def cx(self):
        return self.K[0, 2]

    @property
    def cy(self):
        return self.K[1, 2]

    def to_dict(self):
        """Serialize intrinsic parameters to a dict object."""
        data = {
            "width": self.width,
            "height": self.height,
            "K": self.K.flatten().tolist(),
        }
        return data

    @classmethod
    def from_dict(cls, data):
        """Deserialize intrinisic parameters from a dict object."""
        intrinsic = cls(
            width=data["width"],
            height=data["height"],
            fx=data["K"][0],
            fy=data["K"][4],
            cx=data["K"][2],
            cy=data["K"][5],
        )
        return intrinsic

class CameraExtrinsic(object):
    """Intrinsic parameters of a pinhole camera model.

    Attributes:
        width (int): The width in pixels of the camera.
        height(int): The height in pixels of the camera.
        K: The intrinsic camera matrix.
    """

    def __init__(self, value):
        self.extrinsic = np.array(
            [[value[0][0], value[0][1], value[0][2], value[0][3]],
             [value[1][0], value[1][1], value[1][2], value[1][3]],
             [value[2][0], value[2][1], value[2][2], value[2][3]],
             [value[3][0], value[3][1], value[3][2], value[3][3]]]
        )

    @classmethod
    def from_dict(cls, data):
        """Deserialize intrinisic parameters from a dict object."""
        extrinsic = cls(
            value = data["value"]
        )
        return extrinsic
        
class Camera:
    def __init__(self, intrinsic, near=0.01, far=20.0, size=448, fov=35, dist=5.0, phi=np.pi/5, theta=np.pi, fixed_position=True):
        self.intrinsic = intrinsic
        self.width, self.height = size, size
        self.near, self.far = near, far
        self.fov = fov
        self.scale = 1000
        aspect = self.width / self.height

        self.proj_matrix = _build_projection_matrix(intrinsic, near, far)
        self.gl_proj_matrix = self.proj_matrix.flatten(order="F")

        if fixed_position:
            phi=np.pi/5
            theta=np.pi,
        pos = np.array([dist*np.cos(phi)*np.cos(theta), \
                dist*np.cos(phi)*np.sin(theta), \
                dist*np.sin(phi)])
        forward = -pos / np.linalg.norm(pos)
        left = np.cross([0, 0, 1], forward)
        left = left / np.linalg.norm(left)
        up = np.cross(forward, left)
        #视图矩阵：计算世界坐标系中的物体在摄像机坐标系下的坐标
        # print("pose", pos)
        self.view_matrix = p.computeViewMatrix(pos,
                                               forward,
                                               up)
        #投影矩阵：计算世界坐标系中的物体在相机二维平面上的坐标
        self.projection_matrix = p.computeProjectionMatrixFOV(self.fov, aspect, self.near, self.far)
        _view_matrix = np.array(self.view_matrix).reshape((4, 4), order='F')
        _projection_matrix = np.array(self.projection_matrix).reshape((4, 4), order='F')
        # print("self.projection_matrix ", self.projection_matrix)
        # print("self._view_matrix ", _view_matrix)
        #@ ：相乘运算，inv：计算逆矩阵
        self.tran_pix_world = np.linalg.inv(_projection_matrix @ _view_matrix)
        
        self.cx = intrinsic.K[0, 2]
        self.cy = intrinsic.K[1, 2]
        self.fx = intrinsic.K[0, 0]
        self.fy = intrinsic.K[1, 1]
        mat44 = np.eye(4)
        mat44[:3, :3] = np.vstack([forward, left, up]).T
        mat44[:3, 3] = pos      # mat44 is cam2world
        self.mat44 = mat44
        # log parameters
        self.near = near
        self.far = far
        self.dist = dist
        self.theta = theta
        self.phi = phi
        self.pos = pos
        self.forward = forward
        self.left = left
        self.up = up
        self._view_matrix = mat44
        # self.gl_view_matrix = _view_matrix.flatten(order="F")

    def render(self, extrinsic):
        """Render synthetic RGB and depth images.

        Args:
            extrinsic: Extrinsic parameters, T_cam_ref.
        """
        # Construct OpenGL compatible view and projection matrices.
        gl_view_matrix = extrinsic.copy() if extrinsic is not None else np.eye(4)
        gl_view_matrix[2, :] *= -1  # flip the Z axis
        self.gl_view_matrix = gl_view_matrix.flatten(order="F")
        result = p.getCameraImage(
            width=self.intrinsic.width,
            height=self.intrinsic.height,
            viewMatrix=self.gl_view_matrix,
            projectionMatrix=self.gl_proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )

        rgb, z_buffer = np.ascontiguousarray(result[2][:, :, :3]), result[3]
        depth = (
                1.0 * self.far * self.near / (self.far - (self.far - self.near) * z_buffer)
        )

        return Frame(rgb, depth, self.intrinsic, extrinsic)

    def rgbd_2_world(self, w, h, d):
        x = (2 * w - self.width) / self.width
        y = -(2 * h - self.height) / self.height
        z = 2 * d - 1
        pix_pos = np.array((x, y, z, 1))
        position = self.tran_pix_world @ pix_pos
        position /= position[3]

        return position[:3]

    def shot(self):
        # Get depth values using the OpenGL renderer
        _w, _h, rgb, depth, seg = p.getCameraImage(self.width, self.height,
                                                   self.gl_view_matrix, self.gl_proj_matrix
                                                   )
        return rgb, depth, seg
    '''
    批量处理深度图像数据, 将多个像素的RGBD信息转换成世界坐标系下的三维位置信息
    '''
    def rgbd_2_world_batch(self, depth):
        x = (2 * np.arange(0, self.width) - self.width) / self.width
        x = np.repeat(x[None, :], self.height, axis=0)
        y = -(2 * np.arange(0, self.height) - self.height) / self.height
        y = np.repeat(y[:, None], self.width, axis=1)
        z = 2 * depth - 1

        pix_pos = np.array([x.flatten(), y.flatten(), z.flatten(), np.ones_like(z.flatten())]).T
        position = self.tran_pix_world @ pix_pos.T
        position = position.T
        # print(position)

        position[:, :] /= position[:, 3:4]

        return position[:, :3].reshape(*x.shape, -1)
    
    def compute_camera_XYZA(self, depth):
        camera_matrix = self.tran_pix_world[:3, :3]
        y, x = np.where(depth < 5) # 输出所有为True的元素的索引
        z = self.near * self.far / (self.far + depth * (self.near - self.far)) # 深度图的像素值（通常是一个归一化的值，表示从相机到物体的距离与近裁剪面和远裁剪面之间距离的比例）转换为实际的Z坐标。
        permutation = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
        points = (permutation @ np.dot(np.linalg.inv(camera_matrix), \
            np.stack([x, y, np.ones_like(x)] * z[y, x], 0))).T # np.ones_like(x)为 写成齐次坐标形式，*z[y, x]为了将深度值映射到一个特定的范围内
        return y, x, points

    def create_point_cloud_from_depth_image(self, depth, organized=True):
        """ Generate point cloud using depth image only.

            Input:
                depth: [numpy.ndarray, (H,W), numpy.float32]
                    depth image
                camera: [CameraInfo]
                    camera intrinsics
                organized: bool
                    whether to keep the cloud in image shape (H,W,3)

            Output:
                cloud: [numpy.ndarray, (H,W,3)/(H*W,3), numpy.float32]
                    generated cloud, (H,W,3) for organized=True, (H*W,3) for organized=False
        """
        assert(depth.shape[0] == self.height and depth.shape[1] == self.width)
        # depImg = self.far * self.near / (self.far - (self.far - self.near) * depth)
        depImg = np.asanyarray(depth).astype(np.float32) * 1000
        depth = (depImg.astype(np.uint16))
        print(type(depImg[0, 0]))
        
        camera_matrix = self.tran_pix_world
        # xmap = np.arange(self.width)
        # ymap = np.arange(self.height)
        ymap, xmap = np.where(depth < 2000)
        # points_z[ymap, xmap]

        # xmap, ymap = np.meshgrid(xmap, ymap)
        points_z = depth[ymap, xmap] / self.scale
        points_x = (xmap - self.cx) * points_z / self.fx
        points_y = (ymap - self.cy) * points_z / self.fy
        cloud = np.stack([points_x, points_y, points_z], axis=-1)
        if not organized:
            cloud = cloud.reshape([-1, 3])
        # permutation = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
        # points = (permutation @ np.dot(np.linalg.inv(camera_matrix), \
        #     np.stack([points_x, points_y, np.ones_like(points_x)] * points_z[points_y, points_x], 0))).T 
        cloud_transformed = self.transform_point_cloud(cloud, self.mat44, format='4x4')
        # cloud_transformed = self.transform_point_cloud(cloud, camera_matrix, format='4x4')
        print("pc min and max: ", np.min(cloud_transformed[:, 2]), np.max(cloud_transformed[:, 2]))
        return ymap, xmap, cloud

    def transform_point_cloud(self, cloud, transform, format='3x3'):
        """ Transform points to new coordinates with transformation matrix.

            Input:
                cloud: [np.ndarray, (N,3), np.float32]
                    points in original coordinates
                transform: [np.ndarray, (3,3)/(3,4)/(4,4), np.float32]
                    transformation matrix, could be rotation only or rotation+translation
                format: [string, '3x3'/'3x4'/'4x4']
                    the shape of transformation matrix
                    '3x3' --> rotation matrix
                    '3x4'/'4x4' --> rotation matrix + translation matrix

            Output:
                cloud_transformed: [np.ndarray, (N,3), np.float32]
                    points in new coordinates
        """
        if not (format == '3x3' or format == '4x4' or format == '3x4'):
            raise ValueError('Unknown transformation format, only support \'3x3\' or \'4x4\' or \'3x4\'.')
        if format == '3x3':
            cloud_transformed = np.dot(transform, cloud.T).T
        elif format == '4x4' or format == '3x4':
            ones = np.ones(cloud.shape[0])[:, np.newaxis]
            cloud_ = np.concatenate([cloud, ones], axis=1)
            cloud_transformed = np.dot(transform, cloud_.T).T
            cloud_transformed = cloud_transformed[:, :3]
        return cloud_transformed

    @staticmethod
    def compute_XYZA_matrix(id1, id2, pts, size1, size2): # 将点 pts 放置在（size1, size2）矩阵的位置 (id1, id2) 上
        out = np.zeros((size1, size2, 4), dtype=np.float32)
        out[id1, id2, :3] = pts
        out[id1, id2, 3] = 1 # 将 (id1, id2) 位置上的第四个维度（A）设置为1
        return out

    def get_normal_map(self, relative_offset, cam, cwT=None):
        rgb, depth, _, _ = update_camera_image_to_base(relative_offset, cam, cwT)
        image_array = rgb[:, :, :3]
        normal_map = startConvert(image_array)
        return normal_map

    def get_grasp_regien_mask(self, id1, id2, sz1, sz2):
        link_mask = np.zeros((sz1, sz2)).astype(np.uint8)
        for i in range(id1.shape[0]): # 返回索引值和元素
            link_mask[id1[i]][id2[i]] = 1
        return link_mask

    def get_observation(self):
        _w, _h, rgba, depth, seg = p.getCameraImage(self.width, self.height,
                                                   self.gl_view_matrix, self.gl_proj_matrix)
        # rgba = (rgba * 255).clip(0, 255).astype(np.float32) / 255
        # white = np.ones((rgba.shape[0], rgba.shape[1], 3), dtype=np.float32)
        # mask = np.tile(rgba[:, :, 3:4], [1, 1, 3])
        # rgb = rgba[:, :, :3] * mask + white * (1 - mask)
        depImg = np.asanyarray(depth).astype(np.float32) * 1000
        depth = (depImg.astype(np.uint16))
        return rgba, depth

    def depth_image(self, depth):
        return np.asarray(depth)

        # return camera parameters
    def get_metadata_json(self):
        return {
            'dist': self.dist,
            'theta': self.theta,
            'phi': self.phi,
            'near': self.near,
            'far': self.far,
            'width': self.width,
            'height': self.height,
            'fov': self.fov,
            'camera_matrix': np.array(self.gl_view_matrix).tolist(),
            'projection_matrix': np.array(self.gl_proj_matrix).tolist(),
            'model_matrix': self.tran_pix_world.tolist(),
            'mat44': self.mat44.tolist(),
        }

class Frame(object):
    def __init__(self, rgb, depth, intrinsic, extrinsic=None, depth_scale=1.0, depth_trunc=2.0):
        if depth_scale is not None:
            self.rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color=o3d.geometry.Image(rgb),
                depth=o3d.geometry.Image(depth),
                depth_scale=depth_scale,
                depth_trunc=depth_trunc,
                convert_rgb_to_intensity=False
            )
        else:
            self.rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color=o3d.geometry.Image(rgb),
                depth=o3d.geometry.Image(depth),
                convert_rgb_to_intensity=False
            )

        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=intrinsic.width,
            height=intrinsic.height,
            fx=intrinsic.fx,
            fy=intrinsic.fy,
            cx=intrinsic.cx,
            cy=intrinsic.cy,
        )

        self.extrinsic = extrinsic if extrinsic is not None \
            else np.eye(4)

    def color_image(self):
        return np.asarray(self.rgbd.color)

    def depth_image(self):
        return np.asarray(self.rgbd.depth)

    def point_cloud(self, w=448, h=448):
        pc = o3d.geometry.PointCloud.create_from_rgbd_image(
            image=self.rgbd,
            intrinsic=self.intrinsic,
            extrinsic=self.extrinsic
        )
        pc = np.asarray(pc.points).reshape(w, h, -1)

        return pc

def point_cloud_flter(pc, depth):
        row, col = np.where(depth < 2000)
        pc = pc[row, col]

        return row, col, pc

def _build_projection_matrix(intrinsic, near, far):
    perspective = np.array(
        [
            [intrinsic.fx, 0.0, -intrinsic.cx, 0.0],
            [0.0, intrinsic.fy, -intrinsic.cy, 0.0],
            [0.0, 0.0, near + far, near * far],
            [0.0, 0.0, -1.0, 0.0],
        ]
    )
    ortho = _gl_ortho(0.0, intrinsic.width, intrinsic.height, 0.0, near, far)
    return np.matmul(ortho, perspective)

'''
坐标可视化
'''
class DebugAxes(object):
    """
    可视化基于base的坐标系, 红色x轴, 绿色y轴, 蓝色z轴
    常用于检查当前关节pose或者测量关键点的距离
    用法:
    goalPosition1 = DebugAxes()
    goalPosition1.update([0,0.19,0.15
                         ],[0,0,0,1])
    """
    def __init__(self):
        self.uids = [-1, -1, -1]

    def update(self, pos, orn):
        """
        Arguments:
        - pos: len=3, position in world frame
        - orn: len=4, quaternion (x, y, z, w), world frame
        """
        pos = np.asarray(pos).reshape(3)

        rot3x3 = R.from_quat(orn).as_matrix()
        axis_x, axis_y, axis_z = rot3x3.T
        self.uids[0] = p.addUserDebugLine(pos, pos + axis_x * 0.15, [1, 0, 0], replaceItemUniqueId=self.uids[0])
        self.uids[1] = p.addUserDebugLine(pos, pos + axis_y * 0.15, [0, 1, 0], replaceItemUniqueId=self.uids[1])
        self.uids[2] = p.addUserDebugLine(pos, pos + axis_z * 0.15, [0, 0, 1], replaceItemUniqueId=self.uids[2])

def showAxes(pos, axis_x, axis_y, axis_z):
    p.addUserDebugLine(pos, pos + axis_x * 0.5, [1, 0, 0]) # red
    p.addUserDebugLine(pos, pos + axis_y * 0.5, [0, 1, 0]) # green
    p.addUserDebugLine(pos, pos + axis_z * 0.5, [0, 0, 1]) # blue

def ornshowAxes(pos, orn):
    rot3x3 = R.from_quat(orn).as_matrix()
    axis_x, axis_y, axis_z = rot3x3.T
    # print("axis_x, axis_y, axis_z ", axis_x, axis_y, axis_z)
    p.addUserDebugLine(pos, pos + axis_x * 0.5, [1, 0, 0], lineWidth=2) # red
    p.addUserDebugLine(pos, pos + axis_y * 0.5, [0, 1, 0], lineWidth=2) # green
    p.addUserDebugLine(pos, pos + axis_z * 0.5, [0, 0, 1], lineWidth=2) # blue

def _gl_ortho(left, right, bottom, top, near, far):
    ortho = np.diag(
        [2.0 / (right - left), 2.0 / (top - bottom), -2.0 / (far - near), 1.0]
    )
    ortho[0, 3] = -(right + left) / (right - left)
    ortho[1, 3] = -(top + bottom) / (top - bottom)
    ortho[2, 3] = -(far + near) / (far - near)
    return ortho

'''
绑定相机位置并获取更新图像
'''
def update_camera_image(end_state, camera):
    end_pos = end_state[0]
    end_orn = end_state[1]
    wcT = _bind_camera_to_end(end_pos, end_orn)
    cwT = np.linalg.inv(wcT)

    frame = camera.render(cwT)
    assert isinstance(frame, Frame)

    rgb = frame.color_image()  # 这里以显示rgb图像为例, frame还包含了深度图, 也可以转化为点云
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])  # flip the rgb channel

    rgbd = frame.depth_image()
    import matplotlib
    matplotlib.use('TkAgg')  # 大小写无所谓 tkaGg ,TkAgg 都行
    import matplotlib.pyplot as plt

    # plt.figure(num=1)
    # plt.imshow(rgb)
    # plt.show()

    # plt.figure(num=2)
    # plt.imshow(rgbd)
    # plt.show()

    pc = frame.point_cloud()

    return rgb, rgbd, pc, cwT

def get_rs_pc(rgb, rgbd, intrinsic, w, h):
    frame = Frame(rgb, rgbd, intrinsic, depth_scale=None)
    pc = frame.point_cloud(w, h)
    
    return pc

def update_camera_image_to_base(relative_offset, camera, cwT=None):

    # end_pos = end_state[0]
    # end_orn = end_state[1]
    end_pos = [0,0,0]
    end_orn = R.from_euler('XYZ', [0, 0, 0])
    end_orn = end_orn.as_quat()
    if cwT is None:
        wcT = _bind_camera_to_base(end_pos, end_orn, relative_offset)
        cwT = np.linalg.inv(wcT)

    frame = camera.render(cwT)
    assert isinstance(frame, Frame)

    rgb = frame.color_image()  # 这里以显示rgb图像为例, frame还包含了深度图, 也可以转化为点云
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])  # flip the rgb channel

    rgbd = frame.depth_image()

    # import matplotlib
    # matplotlib.use('TkAgg')  # 大小写无所谓 tkaGg ,TkAgg 都行
    # import matplotlib.pyplot as plt

    # plt.figure(num=1)
    # plt.imshow(rgb)
    # plt.show()

    # plt.figure(num=2)
    # plt.imshow(rgbd)
    # plt.show()

    pc = frame.point_cloud()


    return rgb, rgbd, pc, cwT


def _bind_camera_to_base(end_pos, end_orn_or, relative_offset):
    """设置相机坐标系与末端坐标系的相对位置

    Arguments:
    - end_pos: len=3, end effector position
    - end_orn: len=4, end effector orientation, quaternion (x, y, z, w)

    Returns:
    - wcT: shape=(4, 4), transform matrix, represents camera pose in world frame
    """
    camera_link = DebugAxes()

    end_orn = R.from_quat(end_orn_or).as_matrix()
    end_x_axis, end_y_axis, end_z_axis = end_orn.T

    wcT = np.eye(4)  # w: world, c: camera, ^w_c T
    # wcT[:3, 0] = -end_y_axis  # camera x axis
    # wcT[:3, 1] = -end_z_axis  # camera y axis
    # wcT[:3, 2] = end_x_axis  # camera z axis

    '''
    dist = 1
    theta = np.random.random() * np.pi*2
    phi = (np.random.random()+1) * np.pi/6
    pos = np.array([dist*np.cos(phi)*np.cos(theta), \
            dist*np.cos(phi)*np.sin(theta), \
            dist*np.sin(phi)])
    '''
    wcT[:3, 3] = end_orn.dot(relative_offset) + end_pos  # eye position
    # fg = R.from_euler('XYZ', [-np.pi/2, 0, 0]).as_matrix()

    forward = -relative_offset / np.linalg.norm(relative_offset)
    left = np.cross([0, 0, 1], forward)
    left = left / np.linalg.norm(left)
    up = np.cross(forward, left)
    fg = np.vstack([left, -up, forward]).T

    # 初始化 gripper坐标系，默认gripper正方向朝向-z轴
    robotStartOrn = p.getQuaternionFromEuler([0, 0, 0])
    # gripper坐标系绕y轴旋转-pi/2, 使其正方向朝向+x轴
    robotStartOrn1 = p.getQuaternionFromEuler([0, -np.pi/2, 0])
    robotStartrot3x3 = R.from_quat(robotStartOrn).as_matrix()
    robotStart2rot3x3 = R.from_quat(robotStartOrn1).as_matrix()
    # gripper坐标变换
    basegrippermatZTX = robotStartrot3x3@robotStart2rot3x3

    # 以 -relative_offset 为x轴建立正交坐标系
    forward, up, left = create_orthogonal_vectors(relative_offset)
    fg = np.vstack([forward, up, left]).T

    # gripper坐标变换
    basegrippermatT = fg@basegrippermatZTX
    wcT[:3, :3] = basegrippermatT

    # camera_link.update(wcT[:3,3],R.from_matrix(wcT[:3, :3]).as_quat())

    return wcT

def _bind_camera_to_end(end_pos, end_orn_or):
    """设置相机坐标系与末端坐标系的相对位置

    Arguments:
    - end_pos: len=3, end effector position
    - end_orn: len=4, end effector orientation, quaternion (x, y, z, w)

    Returns:
    - wcT: shape=(4, 4), transform matrix, represents camera pose in world frame
    """
    relative_offset = [0.1, -0.2, 0.2]  # 相机原点相对于末端执行器局部坐标系的偏移量[x,z,y]
    end_orn = R.from_quat(end_orn_or).as_matrix()
    end_x_axis, end_y_axis, end_z_axis = end_orn.T


    wcT = np.eye(4)  # w: world, c: camera, ^w_c T
    wcT[:3, 0] = end_x_axis  # camera x axis
    wcT[:3, 1] = end_y_axis  # camera y axis
    wcT[:3, 2] = end_z_axis  # camera z axis
    wcT[:3, 3] = end_orn.dot(relative_offset) + end_pos  # eye position
    fg = R.from_euler('XYZ', [-np.pi/2, 0, 0]).as_matrix()

    camera_link = DebugAxes()

    wcT[:3, :3] = np.matmul(wcT[:3, :3], fg)
    camera_link.update(wcT[:3,3], end_orn_or)

    return wcT

def get_target_part_pose(objectID, tablaID):
    cid = p.createConstraint(objectID, -1, tablaID, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 1])
    cInfo = p.getConstraintInfo(cid)
    print("info: ", cInfo[7])
    print("info: ", cInfo[9])

    # info = p.getLinkState(self.objectID, self.objLinkID)
    pose = cInfo[7]
    orie = cInfo[9]

    return pose, orie

def camera_setup(camera_config, dist=0):
    with open(camera_config, "r") as j:
        config = json.load(j)

    theta = np.random.random() * np.pi*2
    phi = (np.random.random()+1) * np.pi/4
    pose = np.array([dist*np.cos(phi)*np.cos(theta), \
            dist*np.cos(phi)*np.sin(theta), \
            dist*np.sin(phi)])
    
    return theta, phi, pose, config

def ground_points_seg(cam_XYZA_pts):
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

    return cam_XYZA_filter_pts, inliers


def rebuild_pointcloud_format(inliers, cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts):

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

    return cam_XYZA_filter_id1, cam_XYZA_filter_id2

def piontcloud_preprocess(x, y, cam_XYZA, rgb, train_conf, gt_movable_link_mask, device, h=448, w=448):
    pt = cam_XYZA[x, y, :3]
    ptid = np.array([x, y], dtype=np.int32)
    mask = (cam_XYZA[:, :, 3] > 0.5)
    mask[x, y] = False
    pc = cam_XYZA[mask, :3]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    grid_xy = np.stack([grid_y, grid_x]).astype(np.int32)    # 2 x 448 x 448
    pcids = grid_xy[:, mask].T
    pc_movable = (gt_movable_link_mask > 0)[mask]
    idx = np.arange(pc.shape[0])
    np.random.shuffle(idx)
    while len(idx) < 30000:
        idx = np.concatenate([idx, idx])
    idx = idx[:30000-1]
    pc = pc[idx, :]
    pc_movable = pc_movable[idx]
    pcids = pcids[idx, :]
    pc = np.vstack([pt, pc])
    pcids = np.vstack([ptid, pcids])
    pc_movable = np.append(True, pc_movable)
    pc[:, 0] -= 5
    pc = torch.from_numpy(pc).unsqueeze(0).to(device)

    input_pcid = furthest_point_sample(pc, train_conf.num_point_per_shape).long().reshape(-1)
    pc = pc[:, input_pcid, :3]  # 1 x N x 3 = [1, 10000, 3]
    pc_movable = pc_movable[input_pcid.cpu().numpy()]     # N
    pcids = pcids[input_pcid.cpu().numpy()]
    pccolors = rgb[pcids[:, 0], pcids[:, 1]]/255

    return pc, pccolors

""" Tools for data processing.
    Author: chenxi-wang
"""
class CameraInfo():
    """ Camera intrisics for point cloud creation. """
    def __init__(self, width, height, fx, fy, cx, cy, scale):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.scale = scale

def create_point_cloud_from_depth_image(depth, camera, organized=True):
    """ Generate point cloud using depth image only.

        Input:
            depth: [numpy.ndarray, (H,W), numpy.float32]
                depth image
            camera: [CameraInfo]
                camera intrinsics
            organized: bool
                whether to keep the cloud in image shape (H,W,3)

        Output:
            cloud: [numpy.ndarray, (H,W,3)/(H*W,3), numpy.float32]
                generated cloud, (H,W,3) for organized=True, (H*W,3) for organized=False
    """
    assert(depth.shape[0] == camera.height and depth.shape[1] == camera.width)
    xmap = np.arange(camera.width)
    ymap = np.arange(camera.height)
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depth / camera.scale
    points_x = (xmap - camera.cx) * points_z / camera.fx
    points_y = (ymap - camera.cy) * points_z / camera.fy
    cloud = np.stack([points_x, points_y, points_z], axis=-1)
    if not organized:
        cloud = cloud.reshape([-1, 3])
    return cloud

def transform_point_cloud(cloud, transform, format='4x4'):
    """ Transform points to new coordinates with transformation matrix.

        Input:
            cloud: [np.ndarray, (N,3), np.float32]
                points in original coordinates
            transform: [np.ndarray, (3,3)/(3,4)/(4,4), np.float32]
                transformation matrix, could be rotation only or rotation+translation
            format: [string, '3x3'/'3x4'/'4x4']
                the shape of transformation matrix
                '3x3' --> rotation matrix
                '3x4'/'4x4' --> rotation matrix + translation matrix

        Output:
            cloud_transformed: [np.ndarray, (N,3), np.float32]
                points in new coordinates
    """
    if not (format == '3x3' or format == '4x4' or format == '3x4'):
        raise ValueError('Unknown transformation format, only support \'3x3\' or \'4x4\' or \'3x4\'.')
    if format == '3x3':
        cloud_transformed = np.dot(transform, cloud.T).T
    elif format == '4x4' or format == '3x4':
        ones = np.ones(cloud.shape[0])[:, np.newaxis]
        cloud_ = np.concatenate([cloud, ones], axis=1)
        cloud_transformed = np.dot(transform, cloud_.T).T
        cloud_transformed = cloud_transformed[:, :3]
    return cloud_transformed

def compute_point_dists(A, B):
    """ Compute pair-wise point distances in two matrices.

        Input:
            A: [np.ndarray, (N,3), np.float32]
                point cloud A
            B: [np.ndarray, (M,3), np.float32]
                point cloud B

        Output:
            dists: [np.ndarray, (N,M), np.float32]
                distance matrix
    """
    A = A[:, np.newaxis, :]
    B = B[np.newaxis, :, :]
    dists = np.linalg.norm(A-B, axis=-1)
    return dists

def remove_invisible_grasp_points(cloud, grasp_points, pose, th=0.01):
    """ Remove invisible part of object model according to scene point cloud.

        Input:
            cloud: [np.ndarray, (N,3), np.float32]
                scene point cloud
            grasp_points: [np.ndarray, (M,3), np.float32]
                grasp point label in object coordinates
            pose: [np.ndarray, (4,4), np.float32]
                transformation matrix from object coordinates to world coordinates
            th: [float]
                if the minimum distance between a grasp point and the scene points is greater than outlier, the point will be removed

        Output:
            visible_mask: [np.ndarray, (M,), np.bool]
                mask to show the visible part of grasp points
    """
    grasp_points_trans = transform_point_cloud(grasp_points, pose)
    dists = compute_point_dists(grasp_points_trans, cloud)
    min_dists = dists.min(axis=1)
    visible_mask = (min_dists < th)
    return visible_mask

def get_workspace_mask(cloud, seg, trans=None, organized=True, outlier=0):
    """ Keep points in workspace as input.

        Input:
            cloud: [np.ndarray, (H,W,3), np.float32]
                scene point cloud
            seg: [np.ndarray, (H,W,), np.uint8]
                segmantation label of scene points
            trans: [np.ndarray, (4,4), np.float32]
                transformation matrix for scene points, default: None.
            organized: [bool]
                whether to keep the cloud in image shape (H,W,3)
            outlier: [float]
                if the distance between a point and workspace is greater than outlier, the point will be removed
                
        Output:
            workspace_mask: [np.ndarray, (H,W)/(H*W,), np.bool]
                mask to indicate whether scene points are in workspace
    """
    if organized:
        h, w, _ = cloud.shape
        cloud = cloud.reshape([h*w, 3])
        seg = seg.reshape(h*w)
    if trans is not None:
        cloud = transform_point_cloud(cloud, trans)
    foreground = cloud[seg>0]
    xmin, ymin, zmin = foreground.min(axis=0)
    xmax, ymax, zmax = foreground.max(axis=0)
    mask_x = ((cloud[:,0] > xmin-outlier) & (cloud[:,0] < xmax+outlier))
    mask_y = ((cloud[:,1] > ymin-outlier) & (cloud[:,1] < ymax+outlier))
    mask_z = ((cloud[:,2] > zmin-outlier) & (cloud[:,2] < zmax+outlier))
    workspace_mask = (mask_x & mask_y & mask_z)
    if organized:
        workspace_mask = workspace_mask.reshape([h, w])

    return workspace_mask