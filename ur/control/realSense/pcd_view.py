import open3d as o3d

# 加载 PCD 文件
pcd = o3d.io.read_point_cloud("pointcloud_20241217_112731_000.pcd")

# 可视化点云
o3d.visualization.draw_geometries([pcd])

