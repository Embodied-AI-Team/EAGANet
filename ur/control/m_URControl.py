import time
import copy
import socket
import struct
import numpy as np
import math

from m_RG2Control import RG2
from m_Realsense import Realsense


class UR_Robot:
    def __init__(self, tcp_host_ip="192.168.1.102", tcp_port=30003,
                 is_use_rg2=True, is_use_camera=False):
        # Init varibles

        self.tcp_host_ip = tcp_host_ip
        self.tcp_port = tcp_port

        self.is_use_rg2 = is_use_rg2
        self.is_use_camera = is_use_camera


        # UR5 robot configuration
        # Default joint/tool speed configuration
        self.joint_acc = 1.4  # Safe: 1.4   8
        self.joint_vel = 1.05  # Safe: 1.05  3

        # Joint tolerance for blocking calls
        self.joint_tolerance = 0.01

        # Default tool speed configuration
        self.tool_acc = 0.5  # Safe: 0.5
        self.tool_vel = 0.2  # Safe: 0.2

        # Tool pose tolerance for blocking calls
        self.tool_pose_tolerance = [0.002, 0.002, 0.002, 0.01, 0.01, 0.01]

        # gripper configuration
        if(self.is_use_rg2):
            # Gripper activate
            self.gripper = RG2()
            time.sleep(1.5)

        # realsense configuration
        if(self.is_use_camera):
            self.camera = Realsense()

        # Default robot home joint configuration (the robot is up to air)
        self.home_joint_config = [-(0 / 360.0) * 2 * np.pi, -(90 / 360.0) * 2 * np.pi,
                             (0 / 360.0) * 2 * np.pi, -(90 / 360.0) * 2 * np.pi,
                             -(0 / 360.0) * 2 * np.pi, 0.0]
        # test
        self.testRobot()

    # Test for robot controlmove_and_wait_for_pos
    def testRobot(self):
        try:
            print("Test for robot...")
            # self.move_j([-(0 / 360.0) * 2 * np.pi, -(90 / 360.0) * 2 * np.pi,
            #                  (0 / 360.0) * 2 * np.pi, -(90 / 360.0) * 2 * np.pi,
            #                  -(0 / 360.0) * 2 * np.pi, 0.0])
            
            # self.move_j_p([0.3, 0, 0.3, np.pi/2, 0, 0],0.5,0.5)
            # for i in range(10):
            #     self.move_j_p([0.3, 0, 0.3, np.pi, 0, i*0.1], 0.5, 0.5)
            #     time.sleep(1)
            # self.move_j_p([0.3, 0, 0.3, -np.pi, 0, 0], 0.5, 0.5)
            # self.move_l([0.2, 0.2, 0.3, -np.pi, 0, 0], 0.1, 0.1)
            self.gripper_control("open_rg2.urp")
            # self.gripper_control("close_rg2.urp")
        except:
            print("Test fail! ")
    
    # joint control
    def move_j(self, joint_configuration,k_acc=1,k_vel=1,t=0,r=0):
        """
            关节空间控制
            joint_configuration: value(rad) for each joint
            k_acc: acceleration
            v: velocity
            t: the time(s) to make move, incompatible with avControl
            r: the blend radius is zero meters        
         """
        # Organize data
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM) #AF_INET:use IPV4; SOCK_STREAM:流式套接字
        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        tcp_command = "movej([%f" % joint_configuration[0]  #"movej([]),a=,v=,\n"
        for joint_idx in range(1,6):
            tcp_command = tcp_command + (",%f" % joint_configuration[joint_idx])
        tcp_command = tcp_command + "],a=%f,v=%f,t=%f,r=%f)\n" % (k_acc*self.joint_acc, k_vel*self.joint_vel,t,r)
        self.tcp_socket.send(str.encode(tcp_command))

        # Block until robot reaches target state
        state_data = self.tcp_socket.recv(1500)
        actual_joint_positions = self.parse_tcp_state_data(state_data, 'joint_data')
        while not all([np.abs(actual_joint_positions[j] - joint_configuration[j]) < self.joint_tolerance for j in range(6)]):
            state_data = self.tcp_socket.recv(1500)
            actual_joint_positions = self.parse_tcp_state_data(state_data, 'joint_data')
            time.sleep(0.01)
        self.tcp_socket.close()


    def move_j_p(self, tool_configuration,k_acc=1,k_vel=1,t=0,r=0):
        '''
            工作空间控制
            move_j_p(self, tool_configuration,k_acc=1,k_vel=1,t=0,r=0)
            input:tool_configuration=[x y z r p y]
            其中x y z为三个轴的目标位置坐标(m)
            r p y单位为弧度
        '''
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        print(f"movej_p([{tool_configuration}])")
        # command: movej([joint_configuration],a,v,t,r)\n
        tcp_command = "def process():\n"
        tcp_command +=" array = rpy2rotvec([%f,%f,%f])\n" %(tool_configuration[3],tool_configuration[4],tool_configuration[5])
        tcp_command += "movej(get_inverse_kin(p[%f,%f,%f,array[0],array[1],array[2]]),a=%f,v=%f,t=%f,r=%f)\n" % (tool_configuration[0],
            tool_configuration[1],tool_configuration[2],k_acc * self.joint_acc, k_vel * self.joint_vel,t,r ) # "movej([]),a=,v=,\n"
        tcp_command += "end\n"
        self.tcp_socket.send(str.encode(tcp_command))

        # Block until robot reaches home state
        state_data = self.tcp_socket.recv(1500)
        actual_tool_positions = self.parse_tcp_state_data(state_data, 'cartesian_info')
        while not all([np.abs(actual_tool_positions[j] - tool_configuration[j]) < self.tool_pose_tolerance[j] for j in
                       range(3)]):
            state_data = self.tcp_socket.recv(1500)
            # print(f"tool_position_error{actual_tool_positions - tool_configuration}")
            actual_tool_positions = self.parse_tcp_state_data(state_data, 'cartesian_info')
            time.sleep(0.01)
        time.sleep(1.5)
        self.tcp_socket.close()

    def move_l(self, tool_configuration,k_acc=1,k_vel=1,t=0,r=0):
        """
            Move to position(linear in tool-space)
            input: pose; a; v; t; r
        """
        print(f"movel([{tool_configuration}])")
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        # command: movel([tool_configuration],a,v,t,r)\n
        tcp_command = "def process():\n"
        tcp_command += " array = rpy2rotvec([%f,%f,%f])\n" % (
            tool_configuration[3], tool_configuration[4], tool_configuration[5])
        tcp_command += "movel(p[%f,%f,%f,array[0],array[1],array[2]],a=%f,v=%f,t=%f,r=%f)\n" % (
            tool_configuration[0], tool_configuration[1], tool_configuration[2],
            k_acc * self.joint_acc, k_vel * self.joint_vel,t,r)  # "movel([]),a=,v=,\n"
        tcp_command += "end\n"
        self.tcp_socket.send(str.encode(tcp_command))

        # Block until robot reaches home state
        state_data = self.tcp_socket.recv(1500)
        actual_tool_positions = self.parse_tcp_state_data(state_data, 'cartesian_info')
        while not all([np.abs(actual_tool_positions[j] - tool_configuration[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
            state_data = self.tcp_socket.recv(1500)
            actual_tool_positions = self.parse_tcp_state_data(state_data, 'cartesian_info')
            time.sleep(0.01)
        time.sleep(1.5)
        self.tcp_socket.close()

    # Usually, We don't use move_c
    # move_c is mean that the robot move circle
    # mode 0: Unconstrained mode. Interpolate orientation from current pose to target pose (pose_to)
    #      1: Fixed mode. Keep orientation constant relative to the tangent of the circular arc (starting from current pose)
    def move_c(self,pose_via,tool_configuration,k_acc=1,k_vel=1,r=0,mode=0):

        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        print(f"movec([{pose_via},{tool_configuration}])")
        # command: movec([pose_via,tool_configuration],a,v,t,r)\n
        tcp_command = "def process():\n"
        tcp_command += " via_pose = rpy2rotvec([%f,%f,%f])\n" % (
        pose_via[3],pose_via[4] ,pose_via[5] )
        tcp_command += " tool_pose = rpy2rotvec([%f,%f,%f])\n" % (
        tool_configuration[3], tool_configuration[4], tool_configuration[5])
        tcp_command = f" movec([{pose_via[0]},{pose_via[1]},{pose_via[2]},via_pose[0],via_pose[1],via_pose[2]], \
                [{tool_configuration[0]},{tool_configuration[1]},{tool_configuration[2]},tool_pose[0],tool_pose[1],tool_pose[2]], \
                a={k_acc * self.tool_acc},v={k_vel * self.tool_vel},r={r})\n"
        tcp_command += "end\n"

        self.tcp_socket.send(str.encode(tcp_command))

        # Block until robot reaches home state
        state_data = self.tcp_socket.recv(1500)
        actual_tool_positions = self.parse_tcp_state_data(state_data, 'cartesian_info')
        while not all([np.abs(actual_tool_positions[j] - tool_configuration[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
            state_data = self.tcp_socket.recv(1500)
            actual_tool_positions = self.parse_tcp_state_data(state_data, 'cartesian_info')
            time.sleep(0.01)
        self.tcp_socket.close()
        time.sleep(1.5)

    def go_home(self):
        self.move_j(self.home_joint_config)

    # get robot current state and information
    def get_state(self):
        self.tcp_cket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        state_data = self.tcp_socket.recv(1500)
        self.tcp_socket.close()
        return state_data
    
    # get robot current joint angles and cartesian pose
    def parse_tcp_state_data(self, data, subpasckage):
        dic = {'MessageSize': 'i', 'Time': 'd', 'q target': '6d', 'qd target': '6d', 'qdd target': '6d',
               'I target': '6d',
               'M target': '6d', 'q actual': '6d', 'qd actual': '6d', 'I actual': '6d', 'I control': '6d',
               'Tool vector actual': '6d', 'TCP speed actual': '6d', 'TCP force': '6d', 'Tool vector target': '6d',
               'TCP speed target': '6d', 'Digital input bits': 'd', 'Motor temperatures': '6d', 'Controller Timer': 'd',
               'Test value': 'd', 'Robot Mode': 'd', 'Joint Modes': '6d', 'Safety Mode': 'd', 'empty1': '6d',
               'Tool Accelerometer values': '3d',
               'empty2': '6d', 'Speed scaling': 'd', 'Linear momentum norm': 'd', 'SoftwareOnly': 'd',
               'softwareOnly2': 'd',
               'V main': 'd',
               'V robot': 'd', 'I robot': 'd', 'V actual': '6d', 'Digital outputs': 'd', 'Program state': 'd',
               'Elbow position': 'd', 'Elbow velocity': '3d'}
        ii = range(len(dic))
        for key, i in zip(dic, ii):
            fmtsize = struct.calcsize(dic[key]) # 计算key对应value的size
            data1, data = data[0:fmtsize], data[fmtsize:] # 根据size分割数据
            fmt = "!" + dic[key]
            dic[key] = dic[key], struct.unpack(fmt, data1)

        if subpasckage == 'joint_data':  # get joint data
            q_actual_tuple = dic["q actual"]
            joint_data= np.array(q_actual_tuple[1])
            return joint_data
        elif subpasckage == 'cartesian_info':
            Tool_vector_actual = dic["Tool vector actual"]  # get x y z rx ry rz
            cartesian_info = np.array(Tool_vector_actual[1])
            return cartesian_info
   
    ## 夹爪控制
    def gripper_control(self, command):
        self.gripper.gripper_control(command)

    ## 相机控制
    def camera_control(self):
        color_image, depth_image = self.camera.get_data()
        return color_image, depth_image

    """
        姿态表示方法转换函数
    """
    def rpy2rotating_vector(self,rpy):
        # rpy to R
        R = self.rpy2R(rpy)
        # R to rotating_vector
        return self.R2rotating_vector(R)

    def rpy2R(self,rpy): # [r,p,y] 单位rad
        rot_x = np.array([[1, 0, 0],
                          [0, math.cos(rpy[0]), -math.sin(rpy[0])],
                          [0, math.sin(rpy[0]), math.cos(rpy[0])]])
        rot_y = np.array([[math.cos(rpy[1]), 0, math.sin(rpy[1])],
                          [0, 1, 0],
                          [-math.sin(rpy[1]), 0, math.cos(rpy[1])]])
        rot_z = np.array([[math.cos(rpy[2]), -math.sin(rpy[2]), 0],
                          [math.sin(rpy[2]), math.cos(rpy[2]), 0],
                          [0, 0, 1]])
        R = np.dot(rot_z, np.dot(rot_y, rot_x))
        return R

    def R2rotating_vector(self,R):
        theta = math.acos((R[0, 0] + R[1, 1] + R[2, 2] - 1) / 2)
        print(f"theta:{theta}")
        rx = (R[2, 1] - R[1, 2]) / (2 * math.sin(theta))
        ry = (R[0, 2] - R[2, 0]) / (2 * math.sin(theta))
        rz = (R[1, 0] - R[0, 1]) / (2 * math.sin(theta))
        return np.array([rx, ry, rz]) * theta

    def R2rpy(self,R):
    # assert (isRotationMatrix(R))
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6
        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0
        return np.array([x, y, z])
    

if __name__ == "__main__":
    m_urcontorl = UR_Robot()