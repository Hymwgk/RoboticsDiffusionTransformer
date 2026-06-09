import os
import fnmatch
import json

import h5py
import yaml
import cv2
import numpy as np

from configs.isaaclab_const import ISAACLAB_PROPRIO_INDICES, ISAACLAB_ACTION_INDICES
from data.isaaclab_to_rdt import fill_in_action,fill_in_proprio


class HDF5VLADataset:
    """
    This class is used to sample episodes from the embododiment dataset
    stored in HDF5.
    """
    def __init__(self) -> None:
        # [Modify] The path to the HDF5 dataset directory
        # Each HDF5 file contains one episode
        HDF5_DIR = "data/datasets/rdt_js"
        self.DATASET_NAME = "rdt_js"
        
        self.file_paths = []
        for root, _, files in os.walk(HDF5_DIR):
            for filename in fnmatch.filter(files, '*.hdf5'):
                file_path = os.path.join(root, filename)
                self.file_paths.append(file_path)
                
        # Load the config
        with open('configs/base.yaml', 'r') as file:
            config = yaml.safe_load(file)
        self.CHUNK_SIZE = config['common']['action_chunk_size']
        self.IMG_HISORY_SIZE = config['common']['img_history_size']
        self.STATE_DIM = config['common']['state_dim']
    
        # Get each episode's len
        episode_lens = []
        for file_path in self.file_paths:
            valid, res = self.parse_hdf5_file_state_only(file_path)
            _len = res['state'].shape[0] if valid else 0
            episode_lens.append(_len)
        self.episode_sample_weights = np.array(episode_lens) / np.sum(episode_lens)
    
    def __len__(self):
        return len(self.file_paths)
    
    def get_dataset_name(self):
        return self.DATASET_NAME
    
    def get_item(self, index: int=None, state_only=False):
        """Get a training sample at a random timestep.

        Args:
            index (int, optional): the index of the episode.
                If not provided, a random episode will be selected.
            state_only (bool, optional): Whether to return only the state.
                In this way, the sample will contain a complete trajectory rather
                than a single timestep. Defaults to False.

        Returns:
           sample (dict): a dictionary containing the training sample.
        """
        while True:
            if index is None:
                file_path = np.random.choice(self.file_paths, p=self.episode_sample_weights)
            else:
                file_path = self.file_paths[index]
            valid, sample = self.parse_hdf5_file(file_path) \
                if not state_only else self.parse_hdf5_file_state_only(file_path)
            if valid:
                return sample
            else:
                index = np.random.randint(0, len(self.file_paths))
    
    def parse_hdf5_file(self, file_path):
        """打开每个回合的hdf5
        [Modify] Parse a hdf5 file to generate a training sample at
            a random timestep.

        Args:
            file_path (str): the path to the hdf5 file
        
        Returns:
            valid (bool): whether the episode is valid, which is useful for filtering.
                If False, this episode will be dropped.
            dict: a dictionary containing the training sample,
                {
                    "meta": {
                        "dataset_name": str,    # the name of your dataset.
                        "#steps": int,          # the number of steps in the episode,
                                                # also the total timesteps.
                        "instruction": str      # the language instruction for this episode.
                    },                           
                    "step_id": int,             # the index of the sampled step,
                                                # also the timestep t.
                    "state": ndarray,           # state[t], (1, STATE_DIM).
                    "state_std": ndarray,       # std(state[:]), (STATE_DIM,).
                    "state_mean": ndarray,      # mean(state[:]), (STATE_DIM,).
                    "state_norm": ndarray,      # norm(state[:]), (STATE_DIM,).
                    "actions": ndarray,         # action[t:t+CHUNK_SIZE], (CHUNK_SIZE, STATE_DIM).
                    "state_indicator", ndarray, # indicates the validness of each dim, (STATE_DIM,).
                    "cam_high": ndarray,        # external camera image, (IMG_HISORY_SIZE, H, W, 3)
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                    "cam_high_mask": ndarray,   # indicates the validness of each timestep, (IMG_HISORY_SIZE,) boolean array.
                                                # For the first IMAGE_HISTORY_SIZE-1 timesteps, the mask should be False.
                    "cam_left_wrist": ndarray,  # left wrist camera image, (IMG_HISORY_SIZE, H, W, 3).
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                    "cam_left_wrist_mask": ndarray,
                    "cam_right_wrist": ndarray, # right wrist camera image, (IMG_HISORY_SIZE, H, W, 3).
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                                                # If only one wrist, make it right wrist, plz.
                    "cam_right_wrist_mask": ndarray
                } or None if the episode is invalid.
        """
        # 
        with h5py.File(file_path, 'r') as f:
            
            proprio = f['observations']['proprio'][:]
            num_steps = proprio.shape[0]
            # [Optional] 要求每个回合的长度不小于128步
            if num_steps < 128:
                return False, None
            
            # [Optional] We skip the first few still steps
            EPS = 1e-2
            # Get the idx of the first proprio whose delta exceeds the threshold
            # 这里的目的是为了跳过回合开始时机械臂还没有动的那些步骤
            proprio_delta = np.abs(proprio - proprio[0:1])
            indices = np.where(np.any(proprio_delta > EPS, axis=1))[0]
            if len(indices) > 0:
                first_idx = indices[0]
            else:
                raise ValueError("Found no proprio that exceeds the threshold.")
            
            # 随机采样一个时间步，作为起始
            step_id = np.random.randint(first_idx-1, num_steps)
            

            # 这里直接使用预编码的语言指令embedding文件，避免在训练过程中重复计算语言指令的编码
            dir_path = os.path.dirname(file_path)
            embed_id = np.random.randint(0, 3)
            instruction = os.path.join(dir_path, f"lang_embed_{embed_id}.pt")

            # Assemble the meta
            meta = {
                "dataset_name": self.DATASET_NAME,
                "#steps": num_steps,
                "step_id": step_id,
                "instruction": instruction
            }
            
            
            # Parse the state and action
            state = proprio[step_id:step_id+1]
            state_std = np.std(proprio, axis=0, keepdims=True)
            state_mean = np.mean(proprio, axis=0, keepdims=True)
            state_norm = np.sqrt(np.mean(proprio**2, axis=0, keepdims=True))
            
            actions = f['action'][step_id:step_id+self.CHUNK_SIZE] 

            if actions.shape[0] < self.CHUNK_SIZE:
                # Pad the actions using the last action
                actions = np.concatenate([
                    actions,
                    np.tile(actions[-1:], (self.CHUNK_SIZE-actions.shape[0], 1))
                ], axis=0)
            

            state_indicator = np.zeros(self.STATE_DIM, dtype=np.float32)
            state_indicator[ISAACLAB_PROPRIO_INDICES] = 1.0 # [128,]

            
            # Parse the images
            def parse_img(key):
                imgs = []
                for i in range(max(step_id-self.IMG_HISORY_SIZE+1, 0), step_id+1):
                    img = f['observations']['images'][key][i]
                    # 保证读取出的通道是RGB和推理时候一致
                    img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
                    # if key == "cam_high" and i == step_id:
                    #     from PIL import Image
                    #     from pathlib import Path
                    #     debug_dir = Path("/tmp/rdt_image_debug")
                    #     debug_dir.mkdir(parents=True, exist_ok=True)
                    #     Image.fromarray(img).save(debug_dir / "02_rdt_decoded_as_pil.png")
                    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    imgs.append(img)
                imgs = np.stack(imgs)
                if imgs.shape[0] < self.IMG_HISORY_SIZE:
                    # Pad the images using the first image
                    imgs = np.concatenate([
                        np.tile(imgs[:1], (self.IMG_HISORY_SIZE-imgs.shape[0], 1, 1, 1)),
                        imgs
                    ], axis=0)
                return imgs
            # `cam_high` is the external camera image
            cam_high = parse_img('cam_high')
            # For step_id = first_idx - 1, the valid_len should be one
            valid_len = min(step_id - (first_idx - 1) + 1, self.IMG_HISORY_SIZE)
            cam_high_mask = np.array(
                [False] * (self.IMG_HISORY_SIZE - valid_len) + [True] * valid_len
            )
            cam_left_wrist = parse_img('cam_left_wrist')
            cam_left_wrist_mask = cam_high_mask.copy()
            cam_right_wrist = parse_img('cam_right_wrist')
            cam_right_wrist_mask = cam_high_mask.copy()
            
            # Return the resulting sample
            # For unavailable images, return zero-shape arrays, i.e., (IMG_HISORY_SIZE, 0, 0, 0)
            # E.g., return np.zeros((self.IMG_HISORY_SIZE, 0, 0, 0)) for the key "cam_left_wrist",
            # if the left-wrist camera is unavailable on your robot
            return True, {
                "meta": meta,
                "state": state, # [1,128]
                "state_std": state_std,   # [1,128]TODO debug检查是否在微调过程中使用了 统计量进行归一化？
                "state_mean": state_mean,    # [1,128]
                "state_norm": state_norm,    # [1,128]
                "actions": actions,  # [C,128]
                "state_indicator": state_indicator,   #[128,]
                "cam_high": cam_high,  # [2, H,W,C]
                "cam_high_mask": cam_high_mask, # [2,]
                "cam_left_wrist": cam_left_wrist, # [2, H,W,C]
                "cam_left_wrist_mask": cam_left_wrist_mask, # [2]
                "cam_right_wrist": cam_right_wrist,  # [2, H,W,C]
                "cam_right_wrist_mask": cam_right_wrist_mask # [2,]
            }

    def parse_hdf5_file_state_only(self, file_path):
        with h5py.File(file_path, 'r') as f:
            proprio = f['observations']['proprio'][:].astype(np.float32)
            action_all = f['action'][:].astype(np.float32)

            num_steps = proprio.shape[0]
            if num_steps < 128:
                return False, None

            EPS = 1e-2
            proprio_delta = np.abs(proprio - proprio[0:1])
            indices = np.where(np.any(proprio_delta > EPS, axis=1))[0]
            if len(indices) > 0:
                first_idx = indices[0]
            else:
                raise ValueError("Found no proprio that exceeds the threshold.")

            proprio = proprio[first_idx - 1:]
            action = action_all[first_idx - 1:]



            # proprio = fill_in_proprio(proprio)
            # action = fill_in_action(action)

            return True, {
                "state": proprio,
                "action": action,
            }

if __name__ == "__main__":
    ds = HDF5VLADataset()
    for i in range(len(ds)):
        print(f"Processing episode {i}/{len(ds)}...")
        ds.get_item(i)
