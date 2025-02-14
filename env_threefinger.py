import gym
from gym import error,spaces,utils
from gym.utils import seeding
from collections import namedtuple
from attrdict import AttrDict
import os
import pybullet as p
import pybullet_data
import math
import numpy as np
import random
from utils import ContactError, update_contact_points, pose2exp_coordinate, adjoint_matrix, get_joint_positions, get_movable_joints, unit_point, get_com_pose, link_from_name, CLIENT
import sys
sys.path.append('../')
from pybullet_planning import load_model, connect, wait_for_duration,get_link_pose
from pybullet_planning import get_movable_joints, set_joint_positions, plan_joint_motion

class Env(gym.Env):
    metadata = {'render.modes':['human']}

    def __init__(self, object_position_offset=0.0, vis=False):
        self.vis = vis
        self.current_step = 0
        self.object_position_offset = object_position_offset
        # Observation buffer
        self.control_dt=1./240.
        self.prev_observation = tuple()
        self.endeffort_link = "base_link"
        self.eefID = -1
        self.objLinkID = 0
        self.check_contact = False
        self.hand_actor_id = self.eefID
        self.gripper_actor_ids = []
        self.action_direction_world = []
        self.object_start_pose = []
        self.numJoints = 6
        self.step_length = 0
        connect(use_gui=True)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # connect to engine servers
        # self.physicsClient = p.connect(p.GUI if self.vis else p.DIRECT)
        # add search path for loadURDF
        # p.configureDebugVisualizer(lightPosition=[0, 0, 0.1])
        #开启光线渲染
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)  # Shadows on/off
        # define world
        p.setGravity(0, 0, -10) # NOTE
        p.setTimeStep = self.control_dt
        self.planeID = p.loadURDF("plane.urdf")
        self.tablaID = p.loadURDF("../data/object_urdf/objects/table.urdf",
                            [0.0, 0.0, 0],#base position
                            p.getQuaternionFromEuler([0, 0, 0]),#base orientation
                            useFixedBase=True)
        self.robotUrdfPath = "../data/kinova_j2s7s300/urdf/j2s7s300.urdf"
        self.robotStartPos = [0, 0, 0.35]
        self.robotStartOrn = p.getQuaternionFromEuler([0, 0, 1.57])

    def step(self):
        self.current_step += 1
        self.step_simulation()

    def step_simulation(self):
        """
        Hook p.stepSimulation()
        """
        p.stepSimulation()

    def check_move_direction(self):
        objectLinkid = -1
        object_pose = self.object_start_pose[0]
        target_link_pose = get_link_pose(self.objectID, objectLinkid) # 得到世界坐标系下物体Link的位姿
        mov_dir = np.array(target_link_pose[0], dtype=np.float32) - np.array(object_pose, dtype=np.float32)
        move_dist = np.linalg.norm(mov_dir)
        mov_dir /= np.linalg.norm(mov_dir)
        intended_dir = -np.array(self.action_direction_world, dtype=np.float32)
        success = (intended_dir @ mov_dir > 0.3) and (intended_dir @ mov_dir < 1) # 如果点积接近 1，表示两个向量几乎同向；如果接近 -1，表示几乎反向；如果接近 0，表示两个向量几乎垂直。
        # print("move_dist: , success: , ", move_dist, success, intended_dir @ mov_dir)
        # if move_dist > 0.01 and success:
        #     if (self.first_res):
        #         return True
        if(move_dist > 0.03 and success):
            # self.first_res = True
            return True
        else:
            self.first_res = False
            return False
        
    def wait_n_steps(self, n: int, close_gripper = False):
        for i in range(n):
            self.step_simulation()
            if self.check_contact:
                points = update_contact_points()
                contactSuccess = self.check_contact_is_valid(points, n)
                res = self.check_move_direction()
                if res == True:
                    self.terminal_step = True
                    break
                # print("num point:", i)
                if not ContactError and close_gripper:
                    break
                if not contactSuccess:
                    raise ContactError()
        self.step_length = 0

    def check_depth_change(self, cur_depth):
        _, prev_depth, _ = self.prev_observation
        changed_depth = cur_depth - prev_depth
        changed_depth_counter = np.sum(np.abs(changed_depth) > self.DEPTH_CHANGE_THRESHOLD)
        print('changed depth pixel count:', changed_depth_counter)
        return changed_depth_counter > self.DEPTH_CHANGE_COUNTER_THRESHOLD

    def load_object(self, model_path, x=0, y=0):
        self.objectID = load_model(model_path, ([x, y, 0.1], [0, 0, 0, 1]))

        # compute link actor information
        self.movable_link_ids = []
        dummy_id = 1
        self.movable_link_ids.append(dummy_id)
        self.target_object_part_joint_id = dummy_id

        # t = 0
        # while True:
        #     p.stepSimulation()
        #     t += 1
        #     if t == 120:
        #         break
        return self.objectID
    
    def load_robot(self, model_path, pose, orne):
        self.robotID = load_model(model_path, (pose, orne), fixed_base=True)
        self.eefID = link_from_name(self.robotID, self.endeffort_link)
        
        return self.robotID
        
    def set_friction(self, id):
        # Set friction coefficients for gripper fingers.
        for i in range(p.getNumJoints(id)):
            p.changeDynamics(id, i, lateralFriction=50.0, spinningFriction=10.0, rollingFriction=10.0, frictionAnchor=True)

    def set_target_object_part_actor_id(self, actor_id, custom=True):
        self.target_object_part_actor_id = actor_id
            
    # 计算从一个当前末端执行器（end effector, EE）姿态到目标末端执行器姿态所需的“扭转”（twist）
    def calculate_twist(self, time_to_target, target_ee_pose: np.ndarray):
        eefPose_mat44 = np.eye(4)
        pose, orie = self.get_robot_ee_pose()
        object_matrix = np.array(p.getMatrixFromQuaternion(orie)).reshape(3,3)
        eefPose_mat44[:3, :3] = object_matrix
        eefPose_mat44[:3, 3] = pose
        relative_transform = np.linalg.inv(eefPose_mat44) @ target_ee_pose

        unit_twist, theta = pose2exp_coordinate(relative_transform) # 获得单位旋量和轴角
        velocity = theta / time_to_target # 根据目标角度和时间计算角速度（或扭转速度）
        body_twist = unit_twist * velocity # 将单位扭转乘以速度，得到身体扭转（body twist），它表示在单位时间内末端执行器相对于其当前姿态的扭转
        current_ee_pose = eefPose_mat44
        return adjoint_matrix(current_ee_pose) @ body_twist # adjoint_matrix(current_ee_pose)为ADT，ADT @ V 为速度旋量变化量


    def move_to_target_pose(self, target_ee_pose: np.ndarray, num_steps: int, custom=True) -> None:
        """
        Move the robot hand dynamically to a given target pose
        Args:
            target_ee_pose: (4, 4) transformation of robot hand in robot base frame (ee2base)
            num_steps:  how much steps to reach to target pose, 
                        each step correspond to self.scene.get_timestep() seconds
                        in physical simulation
        """
        executed_time = num_steps * self.control_dt

        spatial_twist = self.calculate_twist(executed_time, target_ee_pose) # [6, ]
        for i in range(num_steps):
            if i % 100 == 0:
                spatial_twist = self.calculate_twist((num_steps - i) * self.control_dt, target_ee_pose)
            qvel = self.compute_joint_velocity_from_twist(spatial_twist) # 末端速度旋量转换到每个自由度
            # print("qvel : ", qvel)
            self.setJointPosition(self.robotID, qvel)
            self.step() # 报异常
            end_effector_pose = get_com_pose(self.robotID, self.eefID)
            print("end_effector_pose", end_effector_pose)
        return

    def move_to_target_pose_onestep(self, target_ee_pose: np.ndarray) -> None:
        self.setJointPosition2(self.robotID, target_ee_pose)


    def open_gripper(self, robotID, joints, gripper_main_control_joint_name, mimic_joint_name, Position0, Position1):
        for i in range(len(gripper_main_control_joint_name)):
            joint = joints[gripper_main_control_joint_name[i]]
            p.setJointMotorControl2(robotID,
                                    joint.id,
                                    p.POSITION_CONTROL,
                                    targetPosition=Position0,
                                    force=joint.maxForce,
                                    maxVelocity=joint.maxVelocity)

        for i in range(len(mimic_joint_name)):
            joint = joints[mimic_joint_name[i]]
            p.setJointMotorControl2(robotID,
                                    joint.id,
                                    p.POSITION_CONTROL,
                                    targetPosition=Position1,
                                    force=joint.maxForce,
                                    maxVelocity=joint.maxVelocity)
        self.wait_n_steps(1000)

    def close_gripper(self, robotID, joints, gripper_main_control_joint_name, mimic_joint_name, Position0, Position1):
        for i in range(len(gripper_main_control_joint_name)):
            joint = joints[gripper_main_control_joint_name[i]]
            p.setJointMotorControl2(robotID,
                                    joint.id,
                                    p.POSITION_CONTROL,
                                    targetPosition=Position0,
                                    targetVelocity=0,
                                    force=joint.maxForce,
                                    maxVelocity=joint.maxVelocity)

        for i in range(len(mimic_joint_name)):
            joint = joints[mimic_joint_name[i]]
            p.setJointMotorControl2(robotID,
                                    joint.id,
                                    p.POSITION_CONTROL,
                                    targetPosition=Position1,
                                    targetVelocity=0,
                                    force=joint.maxForce,
                                    maxVelocity=joint.maxVelocity)
        self.wait_n_steps(1000)   

    def get_target_part_pose(self):
        self.cid = p.createConstraint(self.objectID, -1, self.tablaID, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 1])
        cInfo = p.getConstraintInfo(self.cid)
        print("info: ", cInfo[7])
        print("info: ", cInfo[9])

        # info = p.getLinkState(self.objectID, self.objLinkID)
        pose = cInfo[7]
        orie = cInfo[9]

        return pose, orie

    def get_robot_ee_pose(self):
        cInfo = get_com_pose(self.robotID, self.eefID)
        pose = cInfo[0]
        orie = cInfo[1]

        return pose, orie

    def start_checking_contact(self, robot_hand_actor_id, robot_gripper_actor_ids, strict=False):
        self.check_contact = True
        self.check_contact_strict = strict
        self.first_timestep_check_contact = True
        self.robot_hand_actor_id = robot_hand_actor_id
        self.robot_gripper_actor_ids = robot_gripper_actor_ids

    def end_checking_contact(self):
        # self.check_contact = False
        self.first_timestep_check_contact = False
        self.step_length = 0

    def setup_gripper(self, robotID, gripperControlJoints):
        # controlJoints = ["j2s7s300_joint_finger_1", "j2s7s300_joint_finger_tip_1",
        #                 "j2s7s300_joint_finger_2", "j2s7s300_joint_finger_tip_2",
        #                 "j2s7s300_joint_finger_3", "j2s7s300_joint_finger_tip_3"]
        jointTypeList = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
        numJoints = p.getNumJoints(robotID)
        jointInfo = namedtuple("jointInfo",
                            ["id", "name", "type", "lowerLimit", "upperLimit", "maxForce", "maxVelocity",
                            "controllable"])
        self.joints = AttrDict()
        for i in range(numJoints):
            info = p.getJointInfo(robotID, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = jointTypeList[info[2]]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = True if jointName in gripperControlJoints else False
            self.gripperJointsInfo = jointInfo(jointID, jointName, jointType, jointLowerLimit,
                            jointUpperLimit, jointMaxForce, jointMaxVelocity, controllable)
            # if info.type == "REVOLUTE":  # set revolute joint to static
            #     p.setJointMotorControl2(robotID, info.id, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
            self.joints[self.gripperJointsInfo.name] = self.gripperJointsInfo
            if controllable:
                self.gripper_actor_ids.append(self.gripperJointsInfo[0])
        
        return self.joints

    def get_target_part_qpos(self):
        qpos = [1,1]
        return np.array(float(qpos[self.target_object_part_joint_id]))

    def getJointStates(self, robot):
        joint_states = p.getJointStates(robot, range(p.getNumJoints(robot)))
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        joint_torques = [state[3] for state in joint_states]
        return joint_positions, joint_velocities, joint_torques

    def compute_jacobian(self, robot, link, positions=None):
        joints = get_movable_joints(robot)
        if positions is None:
            positions = get_joint_positions(robot, joints)
        assert len(joints) == len(positions)
        velocities = [0.0] * len(positions)
        accelerations = [0.0] * len(positions)
        translate, rotate = p.calculateJacobian(robot, link, unit_point(), positions,
                                                velocities, accelerations, physicsClientId=CLIENT)
        #movable_from_joints(robot, joints)
        return list(zip(*translate)), list(zip(*rotate)) # len(joints) x 3

    def getMotorJointStates(self, robot):
        joint_states = p.getJointStates(robot, range(p.getNumJoints(robot)))
        joint_infos = [p.getJointInfo(robot, i) for i in range(p.getNumJoints(robot))]
        joint_states = [j for j, i in zip(joint_states, joint_infos) if i[3] > -1]
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        joint_torques = [state[3] for state in joint_states]
        return joint_positions, joint_velocities, joint_torques

    def compute_joint_velocity_from_twist(self, twist: np.ndarray) -> np.ndarray:
        """
        This function is a kinematic-level calculation which do not consider dynamics.
        Pay attention to the frame of twist, is it spatial twist or body twist

        Jacobian is provided for your, so no need to compute the velocity kinematics
        ee_jacobian is the geometric Jacobian on account of only the joint of robot arm, not gripper
        Jacobian in SAPIEN is defined as the derivative of spatial twist with respect to joint velocity

        Args:
            twist: (6,) vector to represent the twist

        Returns:
            (7, ) vector for the velocity of arm joints (not include gripper)

        """
        assert twist.size == 6
        # Jacobian define in SAPIEN use twist (v, \omega) which is different from the definition in the slides
        # So we perform the matrix block operation below
        # dense_jacobian = self.robot.compute_spatial_twist_jacobian()  # (num_link * 6, dof()) (96, 12)
        # ee_jacobian = np.zeros([6, self.robot.dof - 6]) # 2 修改为3  (6,6)
        # ee_jacobian[:3, :] = dense_jacobian[self.end_effector_index * 6 - 3: self.end_effector_index * 6, :self.robot.dof - 6] # 2 修改为6   [33:36, :6]
        # ee_jacobian[3:6, :] = dense_jacobian[(self.end_effector_index - 1) * 6: self.end_effector_index * 6 - 3, :self.robot.dof - 6] # 2 修改为6   [30:33, :6]


        mpos, mvel, mtorq = self.getMotorJointStates(self.robotID)
        # mpos = mpos[:6]
        zero_vec = [0.0] * len(mpos)
        result = p.getLinkState(self.robotID,
                                self.eefID,
                                computeLinkVelocity=1,
                                computeForwardKinematics=1)
        link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot, link_vt, link_vr = result
        # jac_t, jac_r = p.calculateJacobian(self.robotID, self.eefID, com_trn, mpos, zero_vec, zero_vec)
        jac_t, jac_r = self.compute_jacobian(self.robotID, self.eefID)
        ee_jacobian = np.zeros([6, 6]) # 列为关节数
        jac_t = np.array(jac_t)[:6, :]
        jac_r = np.array(jac_r)[:6, :]

        ee_jacobian[:, 3:6] = np.array(jac_t)
        ee_jacobian[:, :3] = np.array(jac_r)

        #numerical_small_bool = ee_jacobian < 1e-1
        #ee_jacobian[numerical_small_bool] = 0
        #inverse_jacobian = np.linalg.pinv(ee_jacobian)
        inverse_jacobian = np.linalg.inv(ee_jacobian)
        #inverse_jacobian[np.abs(inverse_jacobian) > 5] = 0
        #print(inverse_jacobian)
        return inverse_jacobian @ twist
    
    def internal_controller(self, qvel: np.ndarray) -> None:
        """Control the robot dynamically to execute the given twist for one time step

        This method will try to execute the joint velocity using the internal dynamics function in SAPIEN.
        尝试使用SAPIEN中的内部动力学函数来执行关节速度。
        Note that this function is only used for one time step, so you may need to call it multiple times in your code
        Also this controller is not perfect, it will still have some small movement even after you have finishing using
        it. Thus try to wait for some steps using self.wait_n_steps(n) like in the hw2.py after you call it multiple
        time to allow it to reach the target position

        Args:
            qvel: (7,) vector to represent the joint velocity

        """
        assert qvel.size == self.numJoints
        pos, vel, torq = self.getJointStates(self.robotID)
        target_qpos = qvel * self.control_dt + pos[:6] # 2 修改为6
        for i, joint in enumerate(self.numJoints):
            joint.set_drive_velocity_target(qvel[i])
            joint.set_drive_target(target_qpos[i])
        passive_force = self.robot.compute_passive_force()
        self.robot.set_qf(passive_force)

    def setJointPosition(self, robot, qvel: np.ndarray, kp=1.0, kv=0.3):
        assert qvel.size == self.numJoints
        pos, vel, torq = self.getJointStates(self.robotID)
        target_qpos = qvel * self.control_dt + pos[:6] # 2 修改为6

        if len(target_qpos) == self.numJoints:
            p.setJointMotorControlArray(robot,
                                        range(self.numJoints),
                                        p.POSITION_CONTROL,
                                        targetPositions=target_qpos,
                                        targetVelocities=qvel,
                                        positionGains=[kp] * self.numJoints,
                                        velocityGains=[kv] * self.numJoints)
        else:
            print("Not setting torque. "
                "Expected torque vector of "
                "length {}, got {}".format(self.numJoints, len(target_qpos)))

    def setJointPosition2(self, robot, position, kp=1.0, kv=0.3):
        zero_vec = [0.0] * self.numJoints
        if len(position) == self.numJoints:
            p.setJointMotorControlArray(robot,
                                        range(self.numJoints),
                                        p.POSITION_CONTROL,
                                        targetPositions=position,
                                        targetVelocities=zero_vec,
                                        positionGains=[kp] * self.numJoints,
                                        velocityGains=[kv] * self.numJoints)
        else:
            print("Not setting torque. "
                "Expected torque vector of "
                "length {}, got {}".format(self.numJoints, self.numJoints))

    def check_contact_is_valid(self, points, n):
        # self.contacts = self.scene.get_contacts()
        self.contacts = points

        # print("all contact ", self.contacts)
        self.finger1_contact = False
        self.finger2_contact = False
        self.step_length += 1

        for c in self.contacts:
            bodyidA = c.bodyUniqueIdA
            bodyidB = c.bodyUniqueIdB
            linkIndex1 = c.linkIndexA
            linkIndex2 = c.linkIndexB
            has_impulse = False

            # if (self.step_length == n):
            #     print("bodyidA, linkIndex1, bodyidA, linkIndex1", bodyidA, linkIndex1, bodyidA, linkIndex1)
            if abs(c.normalForce) > 1e-4:
                has_impulse = True
            if has_impulse and self.first_timestep_check_contact: # self.gripperJointsInfo
                # print("first contact", self.step_length)
                if (bodyidA == 1 and linkIndex1 == -1 and bodyidA == 3 and linkIndex2 in self.gripper_actor_ids):
                    # print("contact ground")
                    return False
            elif has_impulse and not self.first_timestep_check_contact:
                # print("last contact object", self.step_length)
                if (bodyidA == 1 and linkIndex1 == -1 and bodyidB == 3 and linkIndex2 in self.gripper_actor_ids):
                    # print("contact ground")
                    return False
                elif (bodyidA == 2 and linkIndex1 == -1 and bodyidB == 3 and linkIndex2 in self.gripper_actor_ids):
                    # print("gripper contact object: ", self.finger1_contact)
                    continue
                    # self.finger1_contact = True
                elif (bodyidA == 1 and linkIndex1 == -1 and bodyidB == 2 and linkIndex1 == -1 and self.step_length == n):
                    # print("object on ground at laststep")
                    return False
                elif (bodyidA == 1 and linkIndex1 == -1 and bodyidB == 2 and linkIndex1 == -1):
                    # print("object on ground: ", self.finger2_contact)
                    continue
                    # self.finger2_contact = True
        return True