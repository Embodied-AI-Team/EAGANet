#!/usr/bin/env python

import rospy
import sensor_msgs.msg
import sensor_msgs.point_cloud2 as pc2
import open3d as o3d
import numpy as np
from pynput import keyboard
import threading
import os
from datetime import datetime

"""
运行后按s键保存点云， Ctrl+c退出进程
"""

# 用于存储点云的变量
points = None
file_counter = 0  # 文件编号，从 001 开始递增

def callback(data):
    global points
    
    # 从 PointCloud2 消息中提取点云数据
    pc_data = pc2.read_points(data, field_names=("x", "y", "z"), skip_nans=True)
    
    # 将点云数据转换为 numpy 数组
    points = np.array(list(pc_data))
    
    # 创建 open3d 点云对象
    pcd = o3d.geometry.PointCloud()
    
    # 设置点云数据
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 在每次收到新的点云数据时，打印日志（可选）
    rospy.loginfo("Received new point cloud data.")

def save_pointcloud():
    global file_counter
    
    if points is not None:
        # 创建 open3d 点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # 使用时间戳生成文件名，格式为 pointcloud_YYYYMMDD_HHMMSS.pcd
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"pointcloud_{timestamp}_{file_counter:03d}.pcd"
        
        # 递增文件编号
        file_counter += 1
        
        # 保存点云
        o3d.io.write_point_cloud(file_name, pcd)
        rospy.loginfo(f"Saved point cloud as: {file_name}")
    else:
        rospy.logwarn("No point cloud data available to save.")

def on_press(key):
    try:
        if key.char == 's':  # 检测是否按下 's' 键
            save_pointcloud()
    except AttributeError:
        pass  # 非字符键忽略

def keyboard_listener():
    # 创建键盘监听器
    with keyboard.Listener(on_press=on_press) as listener:
        # 保持监听器运行
        listener.join()

def listener():
    rospy.init_node('pointcloud_listener', anonymous=True)

    # 订阅 /camera/depth/color/points 话题，回调函数为 callback
    rospy.Subscriber("/camera/depth/color/points", sensor_msgs.msg.PointCloud2, callback)

    # 启动键盘监听线程
    listener_thread = threading.Thread(target=keyboard_listener)
    listener_thread.daemon = True  # 设置为守护线程，主线程退出时它会自动退出
    listener_thread.start()

    # 保持节点运行，直到用户按 Ctrl+C 退出
    rospy.spin()

if __name__ == '__main__':
    listener()
    

