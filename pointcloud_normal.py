import pcl
import time
import numpy as np
import os
import time
import functools


# 日志耗时装饰器
def log_execution_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        res = func(*args, **kwargs)
        end = time.perf_counter()
        print('【%s】 took %.2f s' % (func.__name__, (end - start)))
        return res

    return wrapper


def readPointCloud(path):
    cloud = pcl.PointCloud()
    if (str(path).endswith(".las")):
        f = File(path, mode='r')
        print('[INFO] points：{}'.format(len(f.points)))

        # 构建点云
        inFile = np.vstack((f.x, f.y, f.z)).transpose()
        cloud.from_array(np.array(inFile, dtype=np.float32))
        return cloud
    else:
        cloud = pcl.load(path)
        return cloud
    return cloud


@log_execution_time
def radiusSearchNormalEstimation(cloud):
    ne = cloud.make_NormalEstimation()
    ne.set_RadiusSearch(0.1)
    normals = ne.compute()
    print(normals.size, type(normals), normals[0], type(normals[0]))
    count = 0
    for i in range(0, normals.size):
        if (str(normals[i][0]) == 'nan'):
            continue
        count = count + 1
    print(count)


@log_execution_time
def kSearchNormalEstimation(cloud):
    ne = cloud.make_NormalEstimation()
    tree = cloud.make_kdtree()
    ne.set_SearchMethod(tree)
    ne.set_KSearch(10)
    normals = ne.compute()
    print(normals.size, type(normals), normals[0])
    count = 0
    for i in range(0, normals.size):
        if (str(normals[i][0]) == 'nan'):
            continue
        count = count + 1
    print(count)
    # 可视化点云和法线
    import pcl.pcl_visualization
    
    vis = pcl.pcl_visualization.PCLVisualizering("3D Viewer")
    vis.SetBackgroundColor(0, 0, 0)
    vis.AddPointCloud(cloud, b'cloud')
    vis.AddPointCloudNormals(cloud, normals, 10, 0.05, b'normals')
    
    while not vis.WasStopped():
        vis.SpinOnce(100)

    return normals


@log_execution_time
def integralImageNormalEstimation(cloud):
    normalEstimation = cloud.make_IntegralImageNormalEstimation()
    normalEstimation.set_NormalEstimation_Method_AVERAGE_3D_GRADIENT()
    normalEstimation.set_MaxDepthChange_Factor(0.02)
    normalEstimation.set_NormalSmoothingSize(10.0)
    normals = normalEstimation.compute()
    print(normals.size, type(normals), normals[0])
    # print("[INFO] normalEstimate 耗时:%.2f秒" % (endTime - startTime))
    count = 0
    for i in range(0, normals.size):
        if (str(normals[i][0]) == 'nan'):
            continue
        count = count + 1
    print(count)

# import pointcloud_normal
# normalpoint = pointcloud_normal.kSearchNormalEstimation(non_ground_points)