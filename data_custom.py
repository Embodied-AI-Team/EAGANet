"""
    SAPIENVisionDataset
        Joint data loader for six primacts
        for panda (two-finger) gripper: pushing, pushing-left, pushing-up, pulling, pulling-left, pulling-up
"""

import os
import h5py
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import json
from progressbar import ProgressBar
from pyquaternion import Quaternion
from camera import Camera


class PybulletVisionDataset(data.Dataset):

    def __init__(self, primact_types, category_types, data_features, buffer_max_num, \
            abs_thres=0.01, rel_thres=0.5, dp_thres=0.5, img_size=224, \
            no_true_false_equal=False, no_neg_dir_data=False, only_true_data=False):
        self.primact_types = primact_types
        self.category_types = category_types

        self.buffer_max_num = buffer_max_num
        self.img_size = img_size
        self.abs_thres = abs_thres
        self.rel_thres = rel_thres
        self.dp_thres = dp_thres
        self.no_true_false_equal = no_true_false_equal
        self.no_neg_dir_data = no_neg_dir_data
        self.only_true_data = only_true_data

        # data buffer
        self.true_data = []
        self.false_data = []
        # for primact_type in primact_types:
        #     self.true_data[primact_type] = []
        #     self.false_data[primact_type] = []

        # data features
        self.data_features = data_features
        
    def load_data(self, data_list):
        bar = ProgressBar()
        total_num = 0
        print("len(data_list) : ", len(data_list))
        for i in bar(range(len(data_list))):
            cur_dir = data_list[i]
            cur_category, cur_trial_id  = cur_dir.split('/')[-1].rsplit('_', 1)

            # if cur_primact_type not in self.primact_types:
            #     continue

            if cur_category not in self.category_types:
                continue
            try:
                with open(os.path.join(cur_dir, 'result.json'), 'r') as fin: # ../data/gt_data-train_10cats_train_data-pushing/149_Faucet_1_0_pushing_57/result.json
                    total_num = total_num+1
                    result_data = json.load(fin)

                    gripper_direction_camera = np.array(result_data['gripper_direction_camera'], dtype=np.float32)
                    gripper_forward_direction_camera = np.array(result_data['gripper_forward_direction_camera'], dtype=np.float32)
                    grasp_width = np.array(result_data['grasp_width'], dtype=np.float32)

                    ori_pixel_ids = np.array(result_data['pixel_locs'], dtype=np.int32)
                    pixel_ids = np.round(np.array(result_data['pixel_locs'], dtype=np.float32) / 448 * self.img_size).astype(np.int32)
                    
                    success = self.check_success(result_data)

                    # load original data
                    if success:
                        cur_data = (cur_dir, cur_category, cur_trial_id, \
                                ori_pixel_ids, pixel_ids, gripper_direction_camera, gripper_forward_direction_camera, grasp_width, True, True)
                        self.true_data.append(cur_data)
                    else:
                        if not self.only_true_data: # False
                            cur_data = (cur_dir, cur_category, cur_trial_id, \
                                    ori_pixel_ids, pixel_ids, gripper_direction_camera, gripper_forward_direction_camera, grasp_width, True, False)
                            self.false_data.append(cur_data)
                    
                    # load neg-direction false data
                    if not self.no_neg_dir_data: # False
                        cur_data = (cur_dir, cur_category, cur_trial_id, \
                                ori_pixel_ids, pixel_ids, -gripper_direction_camera, gripper_forward_direction_camera, grasp_width, False, False)
                        self.false_data.append(cur_data)
            except FileNotFoundError:
                print("open %d result.json file filed!", cur_trial_id, cur_category)
                continue
        print("Total data lenth: ", total_num)
        print("True data lenth: ", len(self.true_data))
        print("False data lenth: ", len(self.false_data))

        # delete data if buffer full
        if self.buffer_max_num is not None:
            # for primact_type in self.primact_types:
            if len(self.true_data) > self.buffer_max_num:
                self.true_data = self.true_data[-self.buffer_max_num:]
            if len(self.false_data) > self.buffer_max_num:
                self.false_data = self.false_data[-self.buffer_max_num:]

    def check_success(self, result_data):
        if result_data['result'] != 'VALID':
            return False
        
        # abs_motion = abs(result_data['final_target_part_qpos'] - result_data['start_target_part_qpos'])
        # j = result_data['target_object_part_joint_id']
        # tot_motion = result_data['joint_angles_upper'][j] - result_data['joint_angles_lower'][j] + 1e-8
        # success = (abs_motion > self.abs_thres) or (abs_motion / tot_motion > self.rel_thres) # 60 degree
        # if not success:
        #     return False

        # if primact_type == 'pushing':
        #     pass
         
        # elif primact_type == 'pulling':
        # mov_dir = np.array(result_data['touch_position_world_xyz_end'], dtype=np.float32) - \
        #         np.array(result_data['touch_position_world_xyz_start'], dtype=np.float32)
        # mov_dir /= np.linalg.norm(mov_dir)
        # intended_dir = -np.array(result_data['gripper_direction_world'], dtype=np.float32) # gripper_direction_world 世界坐标系下家爪的运动方向
        # success = (intended_dir @ mov_dir > self.dp_thres) # 0.5
        #### 计算位置变化 ####
        mov_dir = np.array(result_data['touch_position_world_xyz_end'], dtype=np.float32) - \
                np.array(result_data['touch_position_world_xyz_start'], dtype=np.float32)
        mov_dir /= np.linalg.norm(mov_dir)
        intended_dir = -np.array(result_data['gripper_direction_world'], dtype=np.float32)
        print("move end distance: ", intended_dir @ mov_dir)
        success = (intended_dir @ mov_dir > 0.01) and (intended_dir @ mov_dir < 1) # 如果点积接近 1，表示两个向量几乎同向；如果接近 -1，表示几乎反向；如果接近 0，表示两个向量几乎垂直。

        # elif primact_type == 'pushing-left' or primact_type == 'pulling-left':
        #     mov_dir = np.array(result_data['touch_position_world_xyz_end'], dtype=np.float32) - \
        #             np.array(result_data['touch_position_world_xyz_start'], dtype=np.float32)
        #     mov_dir /= np.linalg.norm(mov_dir)
        #     intended_dir = np.array(result_data['gripper_forward_direction_world'], dtype=np.float32)
        #     success = (intended_dir @ mov_dir > self.dp_thres)

        # elif primact_type == 'pushing-up' or primact_type == 'pulling-up':
        #     mov_dir = np.array(result_data['touch_position_world_xyz_end'], dtype=np.float32) - \
        #             np.array(result_data['touch_position_world_xyz_start'], dtype=np.float32)
        #     mov_dir /= np.linalg.norm(mov_dir)
        #     up_dir = np.array(result_data['gripper_direction_world'], dtype=np.float32)
        #     forward_dir = np.array(result_data['gripper_forward_direction_world'], dtype=np.float32)
        #     intended_dir = np.cross(up_dir, forward_dir)
        #     success = (intended_dir @ mov_dir > self.dp_thres)

        # else:
        #     raise ValueError('ERROR: primact_type %s not supported in check_success!' % primact_type)
        
        return success
        
    def __str__(self):
        strout = '[SAPIENVisionDataset %d] , abs_thres: %f, rel_thres: %f, dp_thres: %f, img_size: %d\n' % \
                (len(self), self.abs_thres, self.rel_thres, self.dp_thres, self.img_size)

        return strout

    def __len__(self):
        if self.no_true_false_equal:
            max_data = 0
            max_data = max(max_data, len(self.true_data) + len(self.false_data))
            return max_data * len(self.primact_types)
        else:
            max_data = 0
            max_data = max(max_data, len(self.true_data))
            max_data = max(max_data, len(self.false_data))
            return max_data * 2 * len(self.primact_types)

    def __getitem__(self, index):
        primact_id = index % len(self.primact_types)
        # primact_type = self.primact_types[primact_id]
        index = index // len(self.primact_types)

        if self.no_true_false_equal: # False
            if index < len(self.false_data):
                cur_dir, category, trial_id, ori_pixel_ids, pixel_ids, \
                        gripper_direction_camera, gripper_forward_direction_camera, grasp_width, is_original, result = \
                            self.false_data[index]
            else:
                cur_dir, category, trial_id, ori_pixel_ids, pixel_ids, \
                        gripper_direction_camera, gripper_forward_direction_camera, grasp_width, is_original, result = \
                            self.true_data[(index - len(self.false_data)) % len(self.true_data)]
        else:
            if index % 2 == 0:
                cur_dir, category, trial_id, ori_pixel_ids, pixel_ids, \
                        gripper_direction_camera, gripper_forward_direction_camera, grasp_width, is_original, result = \
                            self.true_data[(index//2) % len(self.true_data)]
            else:
                cur_dir, category, trial_id, ori_pixel_ids, pixel_ids, \
                        gripper_direction_camera, gripper_forward_direction_camera, grasp_width, is_original, result = \
                            self.false_data[(index//2) % len(self.false_data)]

        # grids
        grid_x, grid_y = np.meshgrid(np.arange(448), np.arange(448))
        grid_xy = np.stack([grid_y, grid_x]).astype(np.int32)    # 2 x 448 x 448

        data_feats = ()
        out2 = None
        out3 = None
        for feat in self.data_features:
            if feat == 'img':
                with Image.open(os.path.join(cur_dir, 'rgb.png')) as fimg:
                    out = np.array(fimg.resize((self.img_size, self.img_size)), dtype=np.float32) / 255
                out = torch.from_numpy(out).permute(2, 0, 1).unsqueeze(0)
                data_feats = data_feats + (out,)
             
            elif feat == 'pcs':
                '''
                output:
                    pc:
                    out: 点云的三维坐标
                    out2: 点云在图像中的像素坐标
                    out3: 用点云索引重排的interaction mask
                '''
                x, y = ori_pixel_ids[0], ori_pixel_ids[1]
                with h5py.File(os.path.join(cur_dir, 'cam_XYZA.h5'), 'r') as fin: # get有效深度值的像素位置（y, x）和计算出的三维点坐标
                    cam_XYZA_id1 = fin['id1'][:].astype(np.int64)
                    cam_XYZA_id2 = fin['id2'][:].astype(np.int64)
                    cam_XYZA_pts = fin['pc'][:].astype(np.float32)
                out = Camera.compute_XYZA_matrix(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, 448, 448) # out (448, 448, 4)
                with Image.open(os.path.join(cur_dir, 'interaction_mask.png')) as fimg:
                    out3 = (np.array(fimg, dtype=np.float32) > 127)
                pt = out[x, y, :3] # 取像素（x,y）对应的点云
                ptid = np.array([x, y], dtype=np.int32)
                mask = (out[:, :, 3] > 0.5) # 将图像中有点云的位置mask为True, 其他为False  (448, 448)
                mask[x, y] = False
                pc = out[mask, :3] # 重组pc, 将mask拉平，三维压缩为二维，取mask true对应的点云坐标 (sum(mask), 3)
                pcids = grid_xy[:, mask].T # 取mask为true的像素位置  (sum(mask), 2)
                out3 = out3[mask] # 取mask为true对应的像素位置标记为true  (448, 448)
                # Image.fromarray(out3.astype(np.uint8)*255).save(os.path.join(cur_dir, 'interaction_mask2.png'))
                idx = np.arange(pc.shape[0]) # 给pc设置索引  sum(mask)
                np.random.shuffle(idx) # 随机打乱数组的元素顺序
                while len(idx) < 10000:
                    idx = np.concatenate([idx, idx])
                idx = idx[:10000-1] # 取前29999个元素
                pc = pc[idx, :] # 重新索引pc/pcids/out3数组，pc/pcids/out3现在只包含根据随机打乱后的索引选择的子集 (sum(mask), 3)
                pcids = pcids[idx, :]
                out3 = out3[idx] # 重新索引 out3  sum(mask)
                pc = np.vstack([pt, pc]) # (sum(mask)+1, 3)
                pcids = np.vstack([ptid, pcids]) # (sum(mask)+1, 2)
                out3 = np.append(True, out3) # (1, sum(mask)+1)
                # normalize to zero-centered
                pc[:, 0] -= 5
                out = torch.from_numpy(pc).unsqueeze(0)
                out2 = torch.from_numpy(pcids).unsqueeze(0)
                out3 = torch.from_numpy(out3).unsqueeze(0).float()
                data_feats = data_feats + (out,)

            elif feat == 'pc_pxids':
                data_feats = data_feats + (out2,)
             
            elif feat == 'pc_movables':
                data_feats = data_feats + (out3,)
             
            elif feat == 'interaction_mask_small':
                with Image.open(os.path.join(cur_dir, 'interaction_mask.png')) as fimg:
                    out = np.array(fimg.resize((self.img_size, self.img_size)), dtype=np.float32) / 255
                out = (torch.from_numpy(out) > 0.5).float().unsqueeze(0)
                data_feats = data_feats + (out,)
             
            elif feat == 'interaction_mask':
                with Image.open(os.path.join(cur_dir, 'interaction_mask.png')) as fimg:
                    out = np.array(fimg, dtype=np.float32) / 255
                data_feats = data_feats + (out,)
             
            elif feat == 'gripper_img_target':
                if is_original:
                    with Image.open(os.path.join(cur_dir, 'viz_target_pose.png')) as fimg:
                        out = np.array(fimg, dtype=np.float32) / 255
                    out = torch.from_numpy(out).permute(2, 0, 1).unsqueeze(0)
                else:
                    out = torch.ones(1, 3, 448, 448).float()
                data_feats = data_feats + (out,)

            elif feat == 'is_original':
                data_feats = data_feats + (is_original,)
            
            elif feat == 'pixel_id':
                out = torch.from_numpy(pixel_ids).unsqueeze(0)
                data_feats = data_feats + (out,)

            elif feat == 'gripper_direction_camera':
                out = torch.from_numpy(gripper_direction_camera).unsqueeze(0)
                data_feats = data_feats + (out,)

            elif feat == 'gripper_forward_direction_camera':
                out = torch.from_numpy(gripper_forward_direction_camera).unsqueeze(0)
                data_feats = data_feats + (out,)
             
            elif feat == 'grasp_width':
                out = torch.from_numpy(grasp_width).unsqueeze(0)
                data_feats = data_feats + (out,)

            elif feat == 'result':
                out = result
                data_feats = data_feats + (out,)

            elif feat == 'cur_dir':
                data_feats = data_feats + (cur_dir,)

            # elif feat == 'shape_id':
            #     data_feats = data_feats + (shape_id,)

            # elif feat == 'primact_type':
            #     data_feats = data_feats + (primact_type,)
            
            elif feat == 'category':
                data_feats = data_feats + (category,)

            # elif feat == 'cnt_id':
            #     data_feats = data_feats + (cnt_id,)

            elif feat == 'trial_id':
                data_feats = data_feats + (trial_id,)

            else:
                raise ValueError('ERROR: unknown feat type %s!' % feat)

        return data_feats

