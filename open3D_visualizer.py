from __future__ import annotations

import numpy as np

import pyvista as pv
from pyvista import examples

'''
# kitti_file = '/home/zhou/autolab/Where2ActBaseline/code/logs/model_3d_critic_pushing/visu_critic_heatmap-40147-model_epoch_269-nothing/pred.feats'
kitti_file = 'logs/exp-model_3d_critic-train_3d_critic_kinova/visu_critic_heatmap-model_epoch_350-nothing/pred.feats'
# kitti_file = '/home/zhou/autolab/Where2ActBaseline/code/logs/model_3d_critic_pushing/visu_critic_heatmap-model_epoch_269-nothing/pred.feats'
# kitti_file = '/home/zhou/autolab/Where2ActBaseline/code/logs/model_3d_critic_pushing/visu_critic_heatmap-model_epoch_269-nothing/pred.pts'
# kitti_file = '/home/zhou/autolab/Where2ActBaseline/code/logs/model_3d_critic_pushing/visu_critic_heatmap-model_epoch_269-nothing/pred.feats'
# kitti_file = '/home/zhou/autolab/Where2ActBaseline/code/logs/model_3d_critic_grasp/visu_critic_heatmap_custom-40147-model_epoch_76-nothing/pred.feats'
# kitti_file = '/home/zhou/autolab/Where2ActBaseline/code/logs/model_3d_critic_pushing/visu_critic_heatmap_custom-model_epoch_269-nothing/pred.feats'
# pointcloud = np.fromfile(file=kitti_file, dtype=np.float32, count=-1).reshape([-1, 3])
# pointcloud = np.loadtxt(kitti_file).reshape([-1, 3])
pointcloud = np.loadtxt(kitti_file).reshape([-1, 4])
pc_mask = pointcloud[:, 3]>0
pc_mask_matrix = np.repeat(pc_mask, 3, axis=0)
pc_mask_matrix = pc_mask_matrix.reshape(-1, 3)
pred_pc = pointcloud[:, 0:3]
pred_pc_new = pointcloud[:, :3][pc_mask_matrix]
pred_pc_new = pred_pc_new.reshape(-1, 3)
'''

import numpy as np
import open3d
import mmcv
import matplotlib as mpl

class Open3D_visualizer():

    def __init__(self, points) -> None:
        # self.vis = open3d.visualization.Visualizer()
        self.points = self.points2o3d(points)
        # self.gt_boxes = self.box2o3d(gt_bboxes, 'red') if gt_bboxes is not None else None
        # self.pred_boxes = self.box2o3d(pred_bboxes, 'green') if pred_bboxes is not None else None

    def points2o3d(self, points):
        """
        points: np.array, shape(N, 3)
        """
        self.pointcloud = open3d.geometry.PointCloud()
        self.pointcloud.points = open3d.utility.Vector3dVector(points)
        # self.pointcloud.colors = open3d.utility.Vector3dVector(
        #     [[0, 255, 255] for _ in range(len(points))])
        return self.pointcloud

    def box2o3d(self, bboxes, color):
        """
        bboxes: np.array, shape(N, 7)
        color: 'red' or 'green'
        """

        bbox_lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7],
                      [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
        if color == 'red':
            colors = [[1, 0, 0] for _ in range(len(bbox_lines))]  # red
        elif color == 'green':
            colors = [[0, 1, 0] for _ in range(len(bbox_lines))]  # green
        else:
            print("请输入 green 或者 red。green 表示预测框，red 表示真值框。")

        all_bboxes = open3d.geometry.LineSet()
        for bbox in bboxes:  
            corners_3d = self.compute_box_3d(bbox[0:3], bbox[3:6], bbox[6])
            o3d_bbox = open3d.geometry.LineSet()
            o3d_bbox.lines = open3d.utility.Vector2iVector(bbox_lines)
            o3d_bbox.colors = open3d.utility.Vector3dVector(colors)
            o3d_bbox.points = open3d.utility.Vector3dVector(corners_3d)
            all_bboxes += o3d_bbox

        return all_bboxes

    def compute_box_3d(self, center, size, heading_angle):
        """
        计算 box 的 8 个顶点坐标
        """
        h = size[2]
        w = size[0]
        l = size[1]
        heading_angle = -heading_angle - np.pi / 2

        center[2] = center[2] + h / 2
        R = self.rotz(1 * heading_angle)
        l = l / 2
        w = w / 2
        h = h / 2
        x_corners = [-l, l, l, -l, -l, l, l, -l]
        y_corners = [w, w, -w, -w, w, w, -w, -w]
        z_corners = [h, h, h, h, -h, -h, -h, -h]
        corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
        corners_3d[0, :] = center[0] + corners_3d[0, :]
        corners_3d[1, :] = center[1] + corners_3d[1, :]
        corners_3d[2, :] = center[2] + corners_3d[2, :]
        return np.transpose(corners_3d)

    def rotz(self, t):
        """Rotation about the z-axis."""
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def draw_geometries(self, result, batch):
        if batch:
            for i in range(len(result)):
                open3d.visualization.draw_geometries([result[i]], "result", 800, 600, 50, 50, False, False, True)
        else:
            open3d.visualization.draw_geometries(result, "result", 800, 600, 50, 50, False, False, True)

    def add_colors_map(self, colors_map):
        origin_pcd = open3d.geometry.PointCloud(self.points) 

        self.pointcloud.colors = open3d.utility.Vector3dVector(colors_map)
        self.draw_geometries([origin_pcd, self.pointcloud], True)

    def add_colors_heghtmap(self):
        origin_pcd = open3d.geometry.PointCloud(self.points) 
        points = np.asarray(self.points.points)
        point_number = points.shape[0]  # 获取点数量
        colors = np.zeros((point_number, 3))  # 颜色

        # matlablib内置的色带
        # colors_map = mpl.cm.Oranges_r  # cool
        colors_map = mpl.cm.get_cmap("jet")
        colors_map = colors_map(list(range(256)))
        z = points[:, 2]
        zMin = z.min()
        zMax = z.max()
        rangeZ = zMax - zMin

        for i in range(point_number):
            tz = z[i]
            index = int((tz - zMin) / rangeZ * 255)
            colors[i, 0] = colors_map[index, 0]
            colors[i, 1] = colors_map[index, 1]
            colors[i, 2] = colors_map[index, 2]

        self.pointcloud.colors = open3d.utility.Vector3dVector(colors)
        self.draw_geometries([origin_pcd, self.pointcloud], True)

    def show(self):
        # 创建窗口
        self.vis.create_window(window_name="Open3D_visualizer")
        opt = self.vis.get_render_option()
        opt.point_size = 1
        opt.background_color = np.asarray([0, 0, 0])
        # 添加点云、真值框、预测框
        self.vis.add_geometry(self.points)
        # if self.gt_boxes is not None:
        #     self.vis.add_geometry(self.gt_boxes)
        # if self.pred_boxes is not None:
        #     self.vis.add_geometry(self.pred_boxes)

        self.vis.get_view_control().rotate(180.0, 0.0)
        self.vis.run()


def load_pointcloud(pts_filename):
    """
    读取点云文件
    返回 np.array, shape(N, 3)
    """
    # 加载点云
    mmcv.check_file_exist(pts_filename)
    if pts_filename.endswith('.npy'):
        points = np.load(pts_filename)
    else:
        points = np.fromfile(pts_filename, dtype=np.float32)
    # 转换点云格式
    points = points.reshape(-1, 6)[:, [0, 1, 2]]
    return points

'''
if __name__ == '__main__':
    index = 4
    # pts_filename = f'/path/to/your/point/cloud/file.bin'
    # gt_filename = f'/path/to/your/gt/file.pkl'
    # pred_filename = f'/path/to/your/pred/file.pkl'

    # points = load_pointcloud(pts_filename)
    # #  使用 mmcv.load 读取真值和预测框的 pkl，获取对应的 bboxes。bboxes 格式为 np.array，shape 为 (N, 3)
    gt_bboxes = ...
    pred_bboxes = ...
    o3dvis = Open3D_visualizer(pred_pc, gt_bboxes, pred_bboxes)
    # o3dvis.show()
    o3dvis.add_colors_heghtmap()
'''