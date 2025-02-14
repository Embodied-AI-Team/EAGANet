#!/usr/bin/env python

import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

def publish_trajectory(positions, duration=1.0):
    # 创建一个 ROS Publisher
    pub = rospy.Publisher('/gripper_controller/command', JointTrajectory, queue_size=10)
    
    # 等待 ROS 初始化
    rospy.sleep(1)

    # 创建 JointTrajectory 消息
    traj = JointTrajectory()
    traj.header.seq = 0
    traj.header.stamp = rospy.Time.now()
    traj.header.frame_id = ''
    traj.joint_names = ['right_index_1_joint', 'right_little_1_joint', 'right_middle_1_joint', 
                        'right_ring_1_joint', 'right_thumb_2_joint', 'right_thumb_1_joint']
    
    # 创建一个点
    point = JointTrajectoryPoint()
    point.positions = positions
    point.velocities = [0, 0, 0, 0, 0, 0]
    point.accelerations = [0, 0, 0, 0, 0, 0]
    point.effort = [0, 0, 0, 0, 0, 0]
    point.time_from_start = rospy.Duration(duration)

    # 将点添加到 JointTrajectory 消息
    traj.points.append(point)

    # 发布消息
    rospy.loginfo("Publishing trajectory: {}".format(positions))
    pub.publish(traj)

    # 等待指定的时间，确保动作完成
    rospy.sleep(duration)


if __name__ == '__main__':
    try:
        # 初始化 ROS 节点
        rospy.init_node('gripper_control_node')

        # 动作 1
        publish_trajectory([0, 0, 0, 0, 0, 0], duration=1)

        # 动作 2
        publish_trajectory([1.2, 0.5, 0.5, 0.5, 0.65, 0], duration=1)

        # 动作 3
        publish_trajectory([1.7, 1.7, 1.7, 1.7, 0.65, 0], duration=1)

    except rospy.ROSInterruptException:
        pass

