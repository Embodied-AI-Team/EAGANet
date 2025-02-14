import os
import sys
import h5py
import torch
import numpy as np
import importlib
import random
import shutil
from PIL import Image
from collections import namedtuple
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
from colors import colors
colors = np.array(colors, dtype=np.float32)
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import functools
import glob
from collections import namedtuple
from attrdict import AttrDict
from subprocess import call
import pybullet as p
import time
sys.path.append('../')
from pybullet_planning import load_model, connect, wait_for_duration, get_joint_limits, get_max_velocity, get_max_force
from pybullet_planning import  get_movable_joints, set_joint_positions, plan_joint_motion, control_joint, get_joint_positions

CLIENT = 0
BASE_LINK = -1
# Joints

JOINT_TYPES = {
    p.JOINT_REVOLUTE: 'revolute', # 0
    p.JOINT_PRISMATIC: 'prismatic', # 1
    p.JOINT_SPHERICAL: 'spherical', # 2
    p.JOINT_PLANAR: 'planar', # 3
    p.JOINT_FIXED: 'fixed', # 4
    p.JOINT_POINT2POINT: 'point2point', # 5
    p.JOINT_GEAR: 'gear', # 6
}

class ContactError(Exception):
    pass

def force_mkdir(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.mkdir(folder)

def printout(flog, strout):
    print(strout)
    if flog is not None:
        flog.write(strout + '\n')

def optimizer_to_device(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)

def get_model_module(model_version):
    importlib.invalidate_caches()
    return importlib.import_module('models.' + model_version)

def collate_feats(b):
    return list(zip(*b))

def collate_feats_pass(b):
    return b

def collate_feats_with_none(b):
    b = filter (lambda x:x is not None, b)
    return list(zip(*b))

def worker_init_fn(worker_id):
    """ The function is designed for pytorch multi-process dataloader.
        Note that we use the pytorch random generator to generate a base_seed.
        Please try to be consistent.
        References:
            https://pytorch.org/docs/stable/notes/faq.html#dataloader-workers-random-seed
    """
    base_seed = torch.IntTensor(1).random_().item()
    #print(worker_id, base_seed)
    np.random.seed(base_seed + worker_id)

def viz_mask(ids):
    return colors[ids]

def draw_dot(img, xy):
    out = np.array(img, dtype=np.uint8)
    x, y = xy[0], xy[1]
    neighbors = np.array([[0, 0, 0, 1, 1, 1, -1, -1, 1], \
                          [0, 1, -1, 0, 1, -1, 0, 1, -1]], dtype=np.int32)
    for i in range(neighbors.shape[1]):
        nx = x + neighbors[0, i]
        ny = y + neighbors[1, i]
        if nx >= 0 and nx < img.shape[0] and ny >= 0 and ny < img.shape[1]:
            out[nx, ny, 0] = 0
            out[nx, ny, 1] = 0
            out[nx, ny, 2] = 255

    return out

def print_true_false(d):
    d = int(d)
    if d > 0.5:
        return 'True'
    return 'False'

def img_resize(data):
    data = np.array(data, dtype=np.float32)
    mini, maxi = np.min(data), np.max(data)
    data -= mini
    data /= maxi - mini
    data = np.array(Image.fromarray((data*255).astype(np.uint8)).resize((224, 224)), dtype=np.float32) / 255
    data *= maxi - mini
    data += mini
    return data

def export_pts(out, v):
    with open(out, 'w') as fout:
        for i in range(v.shape[0]):
            fout.write('%f %f %f\n' % (v[i, 0], v[i, 1], v[i, 2]))

def export_label(out, l):
    with open(out, 'w') as fout:
        for i in range(l.shape[0]):
            fout.write('%f\n' % (l[i]))

def export_pts_label(out, v, l):
    with open(out, 'w') as fout:
        for i in range(l.shape[0]):
            fout.write('%f %f %f %f\n' % (v[i, 0], v[i, 1], v[i, 2], l[i]))

def render_pts_label_png(out, v, l):
    export_pts(out+'.pts', v)
    export_label(out+'.label', l)
    export_pts_label(out+'.feats', v, l)
    cmd = 'RenderShape %s.pts -f %s.feats %s.png 448 448 -v 1,0,0,-5,0,0,0,0,1' % (out, out, out)
    call(cmd, shell=True)

def export_pts_color_obj(out, v, c):
    with open(out+'.obj', 'w') as fout:
        for i in range(v.shape[0]):
            fout.write('v %f %f %f %f %f %f\n' % (v[i, 0], v[i, 1], v[i, 2], c[i, 0], c[i, 1], c[i, 2]))

def export_pts_color_pts(out, v, c):
    with open(out+'.pts', 'w') as fout:
        for i in range(v.shape[0]):
            fout.write('%f %f %f %f %f %f\n' % (v[i, 0], v[i, 1], v[i, 2], c[i, 0], c[i, 1], c[i, 2]))

def load_checkpoint(models, model_names, dirname, epoch=None, optimizers=None, optimizer_names=None, strict=True):
    if len(models) != len(model_names) or (optimizers is not None and len(optimizers) != len(optimizer_names)):
        raise ValueError('Number of models, model names, or optimizers does not match.')

    for model, model_name in zip(models, model_names):
        filename = f'net_{model_name}.pth'
        if epoch is not None:
            filename = f'{epoch}_' + filename
        model.load_state_dict(torch.load(os.path.join(dirname, filename)), strict=strict)

    start_epoch = 0
    if optimizers is not None:
        filename = os.path.join(dirname, 'checkpt.pth')
        if epoch is not None:
            filename = f'{epoch}_' + filename
        if os.path.exists(filename):
            checkpt = torch.load(filename)
            start_epoch = checkpt['epoch']
            for opt, optimizer_name in zip(optimizers, optimizer_names):
                opt.load_state_dict(checkpt[f'opt_{optimizer_name}'])
            print(f'resuming from checkpoint {filename}')
        else:
            response = input(f'Checkpoint {filename} not found for resuming, refine saved models instead? (y/n) ')
            if response != 'y':
                sys.exit()

    return start_epoch

def get_global_position_from_camera(camera, depth, x, y):
    """
    This function is provided only to show how to convert camera observation to world space coordinates.
    It can be removed if not needed.

    camera: an camera agent
    depth: the depth obsrevation
    x, y: the horizontal, vertical index for a pixel, you would access the images by image[y, x]
    """ 
    cm = camera.get_metadata()
    proj, model = cm['projection_matrix'], cm['model_matrix']
    print('proj:', proj)
    print('model:', model)
    w, h = cm['width'], cm['height']

    # get 0 to 1 coordinate for (x, y) coordinates
    xf, yf = (x + 0.5) / w, 1 - (y + 0.5) / h

    # get 0 to 1 depth value at (x,y)
    zf = depth[int(y), int(x)]

    # get the -1 to 1 (x,y,z) coordinate
    ndc = np.array([xf, yf, zf, 1]) * 2 - 1

    # transform from image space to view space
    v = np.linalg.inv(proj) @ ndc
    v /= v[3]

    # transform from view space to world space
    v = model @ v

    return v

def rot2so3(rotation):
    assert rotation.shape == (3, 3)
    if np.isclose(rotation.trace(), 3):
        return np.zeros(3), 1
    if np.isclose(rotation.trace(), -1):
        raise RuntimeError
    theta = np.arccos((rotation.trace() - 1) / 2)
    omega = 1 / 2 / np.sin(theta) * np.array(
        [rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]]).T
    return omega, theta

def skew(vec):
    return np.array([[0, -vec[2], vec[1]],
                     [vec[2], 0, -vec[0]],
                     [-vec[1], vec[0], 0]])

def adjoint_matrix(pose):
    adjoint = np.zeros([6, 6])
    adjoint[:3, :3] = pose[:3, :3]
    adjoint[3:6, 3:6] = pose[:3, :3]
    adjoint[3:6, 0:3] = skew(pose[:3, 3]) @ pose[:3, :3]
    return adjoint

def pose2exp_coordinate(pose):
    """
    Compute the exponential coordinate corresponding to the given SE(3) matrix
    Note: unit twist is not a unit vector

    Args:
        pose: (4, 4) transformation matrix

    Returns:
        Unit twist: (6, ) vector represent the unit twist
        Theta: scalar represent the quantity of exponential coordinate
    """

    omega, theta = rot2so3(pose[:3, :3])
    ss = skew(omega)
    inv_left_jacobian = np.eye(3, dtype=float) / theta - 0.5 * ss + (
            1.0 / theta - 0.5 / np.tan(theta / 2)) * ss @ ss # np.eye生成主对角元素为1， 其余元素为0的矩阵
    v = inv_left_jacobian @ pose[:3, 3]
    return np.concatenate([omega, v]), theta # [omega, v]组合得到一个六维向量，记作旋量

def viz_mask(ids):
    return colors[ids]

def process_angle_limit(x):
    if np.isneginf(x):
        x = -10
    if np.isinf(x):
        x = 10
    return x

def get_random_number(l, r):
    return np.random.rand() * (r - l) + l

def save_h5(fn, data):
    fout = h5py.File(fn, 'w')
    for d, n, t in data:
        fout.create_dataset(n, data=d, compression='gzip', compression_opts=4, dtype=t)
    fout.close()

JointInfo = namedtuple('JointInfo', ['jointIndex', 'jointName', 'jointType',
                                     'qIndex', 'uIndex', 'flags',
                                     'jointDamping', 'jointFriction', 'jointLowerLimit', 'jointUpperLimit',
                                     'jointMaxForce', 'jointMaxVelocity', 'linkName', 'jointAxis',
                                     'parentFramePos', 'parentFrameOrn', 'parentIndex'])

JointState = namedtuple('JointState', ['jointPosition', 'jointVelocity',
                                       'jointReactionForces', 'appliedJointMotorTorque'])

LinkState = namedtuple('LinkState', ['linkWorldPosition', 'linkWorldOrientation',
                                     'localInertialFramePosition', 'localInertialFrameOrientation',
                                     'worldLinkFramePosition', 'worldLinkFrameOrientation'])

BodyInfo = namedtuple('BodyInfo', ['base_name', 'body_name'])

CollisionInfo = namedtuple('CollisionInfo',
                           '''
                           contactFlag
                           bodyUniqueIdA
                           bodyUniqueIdB
                           linkIndexA
                           linkIndexB
                           positionOnA
                           positionOnB
                           contactNormalOnB
                           contactDistance
                           normalForce
                           lateralFriction1
                           lateralFrictionDir1
                           lateralFriction2
                           lateralFrictionDir2
                           '''.split())

def get_pose(body):
    return p.getBasePositionAndOrientation(body, physicsClientId=CLIENT)

def get_num_joints(body):
    return p.getNumJoints(body, physicsClientId=CLIENT)

def get_joints(body):
    return list(range(get_num_joints(body)))

def get_joint_info(body, joint):
    return JointInfo(*p.getJointInfo(body, joint, physicsClientId=CLIENT))

def get_joint_type(body, joint):
    return get_joint_info(body, joint).jointType

def is_fixed(body, joint):
    return get_joint_type(body, joint) == p.JOINT_FIXED

def is_movable(body, joint):
    return not is_fixed(body, joint)

def prune_fixed_joints(body, joints):
    return [joint for joint in joints if is_movable(body, joint)]

def get_movable_joints(body): # 45 / 87 on pr2
    return prune_fixed_joints(body, get_joints(body))

def get_joint_state(body, joint):
    return JointState(*p.getJointState(body, joint, physicsClientId=CLIENT))

def get_joint_position(body, joint):
    return get_joint_state(body, joint).jointPosition

def get_joint_velocity(body, joint):
    return get_joint_state(body, joint).jointVelocity

def get_joint_reaction_force(body, joint):
    return get_joint_state(body, joint).jointReactionForces

def get_joint_torque(body, joint):
    return get_joint_state(body, joint).appliedJointMotorTorque

def get_joint_positions(body, joints): # joints=None):
    return tuple(get_joint_position(body, joint) for joint in joints)

def get_joint_velocities(body, joints):
    return tuple(get_joint_velocity(body, joint) for joint in joints)

def unit_point():
    return (0., 0., 0.)

def get_link_state(body, link, kinematics=True, velocity=True):
    # TODO: the defaults are set to False?
    # https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/pybullet.c
    return LinkState(*p.getLinkState(body, link,
                                     #computeForwardKinematics=kinematics,
                                     #computeLinkVelocity=velocity,
                                     physicsClientId=CLIENT))

def get_com_pose(body, link): # COM = center of mass
    if link == BASE_LINK:
        return get_pose(body)
    link_state = get_link_state(body, link)
    # urdfLinkFrame = comLinkFrame * localInertialFrame.inverse()
    return link_state.linkWorldPosition, link_state.linkWorldOrientation

def get_body_info(body):
    # TODO: p.syncBodyInfo
    return BodyInfo(*p.getBodyInfo(body, physicsClientId=CLIENT))

def get_base_name(body):
    return get_body_info(body).base_name.decode(encoding='UTF-8')

def get_link_name(body, link):
    if link == BASE_LINK:
        return get_base_name(body)
    return get_joint_info(body, link).linkName.decode('UTF-8')

def get_link_names(body, links):
    return [get_link_name(body, link) for link in links]

def get_link_parent(body, link):
    if link == BASE_LINK:
        return None
    return get_joint_info(body, link).parentIndex

parent_link_from_joint = get_link_parent

def link_from_name(body, name):
    if name == get_base_name(body):
        return BASE_LINK
    for link in get_joints(body):
        if get_link_name(body, link) == name:
            return link
    raise ValueError(body, name)

def rotateMatrixToEulerAngles(RM):
    theta_z = np.arctan2(RM[1, 0], RM[0, 0])
    theta_y = np.arctan2(-1 * RM[2, 0], np.sqrt(RM[2, 1] * RM[2, 1] + RM[2, 2] * RM[2, 2]))
    theta_x = np.arctan2(RM[2, 1], RM[2, 2])
    print(f"Euler angles:\ntheta_x: {theta_x}\ntheta_y: {theta_y}\ntheta_z: {theta_z}")
    return theta_x, theta_y, theta_z

# 旋转矩阵到欧拉角(角度制)
def rotateMatrixToEulerAngles2(RM):
    theta_z = np.arctan2(RM[1, 0], RM[0, 0]) / np.pi * 180
    theta_y = np.arctan2(-1 * RM[2, 0], np.sqrt(RM[2, 1] * RM[2, 1] + RM[2, 2] * RM[2, 2])) / np.pi * 180
    theta_x = np.arctan2(RM[2, 1], RM[2, 2]) / np.pi * 180
    print(f"Euler angles:\ntheta_x: {theta_x}\ntheta_y: {theta_y}\ntheta_z: {theta_z}")
    return theta_x, theta_y, theta_z

def eulerAnglesToRotationMatrix(theta):
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(theta[0]), -np.sin(theta[0])],
                    [0, np.sin(theta[0]), np.cos(theta[0])]
                    ])

    R_y = np.array([[np.cos(theta[1]), 0, np.sin(theta[1])],
                    [0, 1, 0],
                    [-np.sin(theta[1]), 0, np.cos(theta[1])]
                    ])

    R_z = np.array([[np.cos(theta[2]), -np.sin(theta[2]), 0],
                    [np.sin(theta[2]), np.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])

    R = np.dot(R_z, np.dot(R_y, R_x))
    print(f"Rotate matrix:\n{R}")
    return R

def rotation_matrix_to_quaternion(R):
    """将旋转矩阵转换为四元数"""
    w = np.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2]) / 2
    x = (R[2, 1] - R[1, 2]) / (4 * w)
    y = (R[0, 2] - R[2, 0]) / (4 * w)
    z = (R[1, 0] - R[0, 1]) / (4 * w)
    print("R", R)
    return np.array([w, x, y, z])

def create_orthogonal_vectors(v):
    """ 
    Creates three orthonormal vectors from a given direction vector, respecting the right-hand rule. 
    Args: v: The input direction vector (NumPy array). 
    Returns: A tuple containing three orthonormal vectors (u, m, n) as NumPy arrays. 
    Returns None if input is invalid. 
    """ 
    if v.shape != (3,): 
        print("Error: Input vector must be a 3D vector.") 
        return None 
    
    # 规范化方向向量 
    u = v / np.linalg.norm(v) 
    # 选择第二个向量 (这里选择 y 轴，或者随机)
    # w = np.array([0, 1, 0]) 
    w = np.random.randn(3).astype(np.float32)
    w /= np.linalg.norm(w)
    if np.allclose(np.cross(u,w),np.zeros(3)): #如果与Y轴平行，就选X轴 
        w = np.array([1, 0, 0]) 
        
    # 计算第三个向量并规范化 
    v_prime = np.cross(u, w) 
    n = v_prime / np.linalg.norm(v_prime) 
    
    #重新计算第二个向量 
    m = np.cross(n, u) 
    
    return u, m, n  # forward, up, left

def create_orthogonal_vectors2(v):
    """ 
    Creates three orthonormal vectors from a given direction vector, respecting the right-hand rule. 
    Args: v: The input direction vector (NumPy array). 
    Returns: A tuple containing three orthonormal vectors (u, m, n) as NumPy arrays. 
    Returns None if input is invalid. 
    """ 
    if v.shape != (3,): 
        print("Error: Input vector must be a 3D vector.") 
        return None 

    # 规范化方向向量 
    u = v / np.linalg.norm(v) 
    # 选择第二个向量 (这里选择随机，你可以根据需要修改)
    w = np.random.randn(3).astype(np.float32)
 
    while (u @ w) > 0.99 or (u @ w) < 0:
        w = np.random.randn(3).astype(np.float32)

    # 计算第三个向量并规范化 
    v_prime = np.cross(u, w) 
    n = v_prime / np.linalg.norm(v_prime) 
    
    #重新计算第二个向量 
    m = np.cross(n, u) 
    
    return u, m, n

def are_parallel(a, b, tolerance=1e-6):
    cross_product = np.cross(a, b)

    return np.allclose(cross_product, np.zeros(3), atol=tolerance)

def get_robot_ee_pose(robotID, eefID):
    cInfo = get_com_pose(robotID, eefID)
    pose = cInfo[0]
    orie = cInfo[1]

    return pose, orie

def update_scene():
    # TODO: https://github.com/bulletphysics/bullet3/pull/3331
    # Always recomputes (no caching)
    p.performCollisionDetection(physicsClientId=CLIENT)

def get_contact_points(**kwargs):
    return [CollisionInfo(*info) for info in p.getContactPoints(physicsClientId=CLIENT, **kwargs)]

def update_contact_points(**kwargs):
    #step_simulation()
    update_scene()
    return get_contact_points(**kwargs)

def control_joints_to_target(env, robotID, jointPose, numJoints, close_gripper=False):
    for i in range(numJoints):
        forcemaxforce = 500 if get_max_force(robotID, i) == 0 else get_max_force(robotID, i)
        p.setJointMotorControl2(bodyUniqueId=robotID,
                                jointIndex=i,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=jointPose[i],
                                targetVelocity=0.0,
                                force=forcemaxforce,
                                maxVelocity = 1,
                                positionGain=0.03,
                                velocityGain=1)
    env.wait_n_steps(1000, close_gripper)

def move_to_target_positions_array(env, robotId, joint_indices, target_positions, slow_down_rate, tolerance=0.1, update_interval=20, max_iter = 500):
    """
    Moves multiple joints of the robot to target positions using setJointMotorControlArray and position control,
    slowing down the motion, and exits when all joints reach their target.
    Returns True if all target positions reached, otherwise False.
    """
    num_joints = len(joint_indices)
    if len(target_positions) != num_joints:
        raise ValueError("Number of target positions must match number of joints")
    current_positions = [p.getJointState(robotId, joint)[0] for joint in joint_indices]
    step_counter = 0
    i = 0
    while True:
        # p.stepSimulation()
        step_counter += 1
        i+=1
        env.wait_n_steps(1)
        if env.terminal_step == True:
            break
        if step_counter % update_interval == 0:
             # Get current joint positions
            # env.wait_n_steps(1)
            # Check if the robot reached all target positions
            if np.all(np.abs(np.array(current_positions) - np.array(target_positions)) < tolerance):
                print("Reached all target positions!")
                # env.wait_n_steps(1)
                return True  # Arrived at all target positions

            p.setJointMotorControlArray(
                bodyUniqueId=robotId,
                jointIndices=joint_indices,
                controlMode=p.POSITION_CONTROL,
                targetPositions=target_positions,
                targetVelocities=[slow_down_rate] * num_joints,
            )

        current_positions = [p.getJointState(robotId, joint)[0] for joint in joint_indices]
        if i > max_iter:
           print("Did not reach target positions in max iteration")
           env.wait_n_steps(1)
           return False
        time.sleep(1/240)


def length_to_plane(point, vector, plane_height): 
    """ Calculates the length along a vector from a point to the ground (z=0 plane). 
    Args: 
        point: A NumPy array representing the 3D point [x0, y0, z0]. 
        vector: A NumPy array representing the 3D vector [x, y, z]. 
        
    Returns: 
    The length to the ground along the vector. Returns an error if the vector is parallel to the ground or if the inputs are not 3D vectors. 
    """ 
    
    if point.shape != (3,) or vector.shape != (3,): 
        raise ValueError("Inputs must be 3D vectors (NumPy arrays of shape (3,)).") 
    x0, y0, z0 = point 
    x, y, z = vector 
    
    if z == 0: 
        raise ValueError("Vector cannot be parallel to the ground plane.") 
    
    t = (plane_height-z0) / z 
    length = abs(t) * np.linalg.norm(vector) 
    
    return length

def get_ikcal_config(robotID):
    movable_joints = get_movable_joints(robotID)
    min_limits = [get_joint_limits(robotID, joint)[0] for joint in movable_joints]
    max_limits = [get_joint_limits(robotID, joint)[1] for joint in movable_joints]
    max_velocities = [get_max_velocity(robotID, joint) for joint in movable_joints] # Range of Jacobian
    current_conf = get_joint_positions(robotID, movable_joints)

    return min_limits, max_limits, max_velocities, current_conf


def setup_sisbot(p, robotID, gripper_type):
    controlJoints = ["finger_joint"]
    jointTypeList = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
    numJoints = p.getNumJoints(robotID)
    jointInfo = namedtuple("jointInfo",
                           ["id", "name", "type", "lowerLimit", "upperLimit", "maxForce", "maxVelocity",
                            "controllable"])
    joints = AttrDict()
    for i in range(numJoints):
        info = p.getJointInfo(robotID, i)
        jointID = info[0]
        jointName = info[1].decode("utf-8")
        jointType = jointTypeList[info[2]]
        jointLowerLimit = info[8]
        jointUpperLimit = info[9]
        jointMaxForce = info[10]
        jointMaxVelocity = info[11]
        controllable = True if jointName in controlJoints else False
        info = jointInfo(jointID, jointName, jointType, jointLowerLimit,
                         jointUpperLimit, jointMaxForce, jointMaxVelocity, controllable)
        # if info.type == "REVOLUTE":  # set revolute joint to static
        #     p.setJointMotorControl2(robotID, info.id, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
        joints[info.name] = info

    # explicitly deal with mimic joints
    def controlGripper(robotID, parent, children, mul, **kwargs):
        controlMode = kwargs.pop("controlMode")
        if controlMode == p.POSITION_CONTROL:
            pose = kwargs.pop("targetPosition")
            # move parent joint
            p.setJointMotorControl2(robotID, parent.id, controlMode, targetPosition=pose,
                                    force=parent.maxForce, maxVelocity=parent.maxVelocity)
            # move child joints
            for name in children:
                child = children[name]
                childPose = pose * mul[child.name]
                p.setJointMotorControl2(robotID, child.id, controlMode, targetPosition=childPose,
                                        force=child.maxForce, maxVelocity=child.maxVelocity)
        else:
            raise NotImplementedError("controlGripper does not support \"{}\" control mode".format(controlMode))
        # check if there
        if len(kwargs) is not 0:
            raise KeyError("No keys {} in controlGripper".format(", ".join(kwargs.keys())))

    assert gripper_type in ['85', '140']
    mimicParentName = "finger_joint"
    if gripper_type == '85':
        mimicChildren = {"right_outer_knuckle_joint": 1,
                         "left_inner_knuckle_joint": 1,
                         "right_inner_knuckle_joint": 1,
                         "left_inner_finger_joint": -1,
                         "right_inner_finger_joint": -1}
    else:
        mimicChildren = {
            "right_outer_knuckle_joint": -1,
            "left_inner_knuckle_joint": -1,
            "right_inner_knuckle_joint": -1,
            "left_inner_finger_joint": 1,
            "right_inner_finger_joint": 1}
    parent = joints[mimicParentName]
    children = AttrDict((j, joints[j]) for j in joints if j in mimicChildren.keys())
    controlRobotiqC2 = functools.partial(controlGripper, robotID, parent, children, mimicChildren)

    return joints, controlRobotiqC2, controlJoints, mimicParentName