# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play and evaluate a trained policy from robomimic.

This script loads a robomimic policy and plays it in an Isaac Lab environment.

Args:
    task: Name of the environment.
    horizon: If provided, override the step horizon of each rollout.
    num_rollouts: If provided, override the number of rollouts.
    seed: If provided, overeride the default random seed.
    norm_factor_min: If provided, minimum value of the action space normalization factor.
    norm_factor_max: If provided, maximum value of the action space normalization factor.
"""

"""Launch Isaac Sim Simulator first."""

import time
import os
import imageio
from tqdm import tqdm 
import argparse
import torch
import json
from pathlib import Path
import yaml
from collections import deque

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate robomimic policy for Isaac Lab environment.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--task", type=str, default="Set-Mode-Off", help="Name of the task.")

parser.add_argument("--pretrained_path", type=str, 
                    default="/data/checkpoints/daily/rdt-finetune-isaaclab-multitask-3/checkpoint-70000/pytorch_model/mp_rank_00_model_states.pt", 
                    help="rdt本体的权重地址")

parser.add_argument("--horizon", type=int, default=500, help="Step horizon of each rollout.")
parser.add_argument("--num_rollouts", type=int, default=50, help="Number of rollouts.")
parser.add_argument("--seed", type=int, default=1, help="Random seed.")


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

import imageio
import numpy as np
import os


def save_success_video(traj, trial_idx, ckpt_path, fps=20):
    """
    将成功的 trajectory 保存为视频。
    支持多视角（如果有多个相机 key）。
    """
    if not traj or "obs" not in traj or len(traj["obs"]) == 0:
        print("[WARNING] No trajectory data to save.")
        return

    # 1. 自动识别哪些 key 是图像
    # 通常 key 包含 'rgb' 或 'image'，且数据是 3 维的 (C, H, W)
    first_obs = traj["obs"][0]
    image_keys = []
    for k, v in first_obs.items():
        # 你的 rollout 里把 tensor 转成了 numpy，shape 应该是 (3, H, W) 或 (1, H, W)
        if isinstance(v, np.ndarray) and v.ndim == 3 and ("zed_left" in k or "image" in k):
            image_keys.append(k)

    if not image_keys:
        print("[WARNING] No image observations found in trajectory.")
        return
    
    
    model_dir = os.path.dirname(ckpt_path) # .../models
    run_dir = os.path.dirname(model_dir)   # .../run_name
    
    # 可能的路径列表（根据你的保存习惯调整）
    save_dir = os.path.join(run_dir,"videos")
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    print(f"[INFO] Saving videos for trial {trial_idx} to {save_dir}...")

    # 2. 遍历每个相机视角进行保存
    for cam_name in image_keys:
        frames = []
        for i, obs in enumerate(traj["obs"]):
            img_data = obs[cam_name] # Shape: (H, W, C), Range: [0.0, 1.0] float
            img_data = img_data.clip(0, 255).astype(np.uint8)
            #img_data = img_data[:, :, [2, 1, 0]] # 交换 R 和 B 通道
            frames.append(img_data)

        # 3. 写入视频文件
        # 文件名示例: success_trial_0_front_view.mp4
        safe_cam_name = cam_name.replace("/", "_") # 防止 key 里有路径符号
        video_path = os.path.join(save_dir, f"success_trial_{trial_idx}_{safe_cam_name}.mp4")
        
        try:
            imageio.mimwrite(video_path, frames, fps=fps, quality=8)
            print(f"       -> Saved {video_path}")
        except Exception as e:
            print(f"[ERROR] Failed to save video {video_path}: {e}")

DEBUG_WITH_GUI = False
args_cli.headless = not DEBUG_WITH_GUI

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
from isaaclab_tasks.utils import parse_env_cfg


"""Rest everything follows."""

import copy
import gymnasium as gym
import numpy as np
import random
import torch
from scripts.isaaclab_model import create_model

# # 仿真器底层存在bug，需要在重置回合之后，空跑几个step
# init_actions = torch.tensor([[2.7667e-01, -3.4115e-09,  6.0912e-01,  1.7574e-02,  8.9313e-01,
#           3.5028e-02,  4.4808e-01,  1.0000e+00,  4.5574e-01, -2.7267e-02,
#           2.4992e-01, -3.9911e-04,  9.3598e-01,  3.5093e-01,  2.8129e-02,
#           1.0000e+00]],device='cuda')

# scale_init_actions = torch.tensor([[2.7667e-01, -3.4115e-09,  6.0912e-01,  1.7574e-02,  8.9313e-01,
#           3.5028e-02,  4.4808e-01, 1.0, 1.0000e+00,  4.5574e-01, -2.7267e-02,
#           2.4992e-01, -3.9911e-04,  9.3598e-01,  3.5093e-01,  2.8129e-02, 1.0,
#           1.0000e+00]],device='cuda')

@torch.inference_mode()
def rollout(policy, env, text_embed, success_term, horizon, device):
    """Perform a single rollout of the policy in the environment.

    Args:
        policy: The robomimicpolicy to play.
        env: The environment to play in.
        horizon: The step horizon of each rollout.
        device: The device to run the policy on.

    Returns:
        terminated: Whether the rollout terminated.
        traj: The trajectory of the rollout.
    """
    obs_dict, _ = env.reset()
    policy.reset()
    
    # 空跑n个step
    # init_steps = 10
    # if 'Scalable' in args_cli.task:
    #     actions = scale_init_actions
    # else:
    #     actions = init_actions
    # for i in range(init_steps):
    #     obs_dict, _, terminated, truncated, _ = env.step(actions)


    #policy.reset()
    # obs_dict, _ = env.reset()
    traj = dict(actions=[], obs=[], next_obs=[])

    # 时间统计
    start_time = time.perf_counter()

    pbar = tqdm(range(horizon), desc="Rollout Progress", unit="step")
    # 每次推理后真正执行几个 action
    exec_horizon = 4
    global_steps = 0
    for i in pbar:
        step_start = time.perf_counter()   # step 开始计时
        # Prepare observations
        obs = copy.deepcopy(obs_dict["policy"])
        for ob in obs:
            obs[ob] = torch.squeeze(obs[ob],dim=0)
        

        obs_to_store = {}
        for k, v in obs.items():
            if isinstance(v, torch.Tensor):
                obs_to_store[k] = v.cpu().numpy() # 或者 v.cpu()
            else:
                obs_to_store[k] = v
        traj["obs"].append(obs_to_store)
        
        pred_actions = policy.step(obs, text_embed).squeeze(0)
        exec_actions = pred_actions[::4][:exec_horizon]

        # 开环执行少量 action，然后重新观测
        for action in exec_actions:
            if global_steps >= horizon:
                break
            action = action.unsqueeze(0)
            # 存当前 obs
            obs_to_store = {}

            for k, v in obs_dict["policy"].items():
                if isinstance(v, torch.Tensor):
                    obs_to_store[k] = v.detach().cpu().numpy()
                else:
                    obs_to_store[k] = v
            traj["obs"].append(obs_to_store)
            obs_dict, _, terminated, truncated, _ = env.step(action)

            # 存 next_obs
            next_obs_cpu = {}
            for k, v in obs_dict["policy"].items():
                if isinstance(v, torch.Tensor):
                    next_obs_cpu[k] = v.detach().cpu().numpy()
                else:
                    next_obs_cpu[k] = v

            traj["next_obs"].append(next_obs_cpu)
            traj["actions"].append(action.detach().cpu().numpy().tolist())

            global_steps += 1
            pbar.update(1)
            step_time = time.perf_counter() - step_start
            pbar.set_postfix({"step_time (s)": f"{step_time:.4f}"})

            # success 判断
            if bool(success_term.func(env, **success_term.params)[0]):
                elapsed = time.perf_counter() - start_time
                fps = global_steps / elapsed
                stats = dict(elapsed=elapsed, steps=global_steps, fps=fps)
                pbar.close()
                return True, traj, stats

            if terminated or truncated:
                elapsed = time.perf_counter() - start_time
                fps = global_steps / elapsed
                stats = dict(elapsed=elapsed, steps=global_steps, fps=fps)
                pbar.close()
                return False, traj, stats

    elapsed = time.perf_counter() - start_time
    fps = global_steps / elapsed
    stats = dict(elapsed=elapsed, steps=global_steps, fps=fps)
    pbar.close()

    return False, traj, stats




def main():
    """Run a trained policy from robomimic with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric)
    # Set observations to dictionary mode for Robomimic
    env_cfg.observations.policy.concatenate_terms = False
    # Set termination conditions
    env_cfg.terminations.time_out = None
    # Disable recorder
    env_cfg.recorders = None
    # Extract success checking function
    success_term = env_cfg.terminations.success
    env_cfg.terminations.success = None


    # Set seed
    torch.manual_seed(args_cli.seed)
    torch.cuda.manual_seed(args_cli.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(args_cli.seed)
    random.seed(args_cli.seed)

    # 数据集文件夹路径
    HDF5_DIR = "data/datasets/rdt_js"
    instructions_path = Path(HDF5_DIR) / "Instructions.json"
    with open(instructions_path, "r") as f:
        instructions = json.load(f)

    # 加载基本的配置路径
    config_path = 'configs/base.yaml'
    with open(config_path, "r") as fp:
        config = yaml.safe_load(fp)

    # 预训练的语言编码与图像编码器
    pretrained_text_encoder_name_or_path = None # "google/t5-v1_1-xxl"
    pretrained_vision_encoder_name_or_path = "google/siglip-so400m-patch14-384"
    # rdt本体路径
    pretrained_path = args_cli.pretrained_path
    # 创建rdt策略
    print(f"[INFO] Loading policy from {pretrained_path}")
    policy = create_model(
        args=config, 
        dtype=torch.bfloat16,
        pretrained=pretrained_path,
        pretrained_text_encoder_name_or_path=pretrained_text_encoder_name_or_path, # 
        pretrained_vision_encoder_name_or_path=pretrained_vision_encoder_name_or_path
    )
    # 读取任务对应的标准语言预编码
    instruction_type = "instruction"
    try:
        text_embed = torch.load(Path(HDF5_DIR)/"text_embeddings"/f"text_embed_{args_cli.task}_{instruction_type}.pt")
        print(f"Loaded text embedding for {args_cli.task}")
    except FileNotFoundError:
        print(f"Text embedding for {args_cli.task} not found")

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.seed(args_cli.seed)

    video_save_dir = os.path.join(os.path.dirname(pretrained_path), "videos")

    # Run policy
    results = []
    success_count = 0  # 追踪成功次数
    total_steps = 0
    with torch.inference_mode():
        for trial in range(args_cli.num_rollouts):
            print(f"[INFO] Starting trial {trial}")
            is_success, traj, stats = rollout(policy, env, text_embed, success_term, horizon=args_cli.horizon, device="cuda")
            results.append(is_success)
            # 累加步数
            this_trial_steps = stats['steps']
            total_steps += this_trial_steps
            print(f"[INFO] Trial {trial}: {is_success}, Elapsed time: {stats['elapsed']:.3f}s, Steps: {stats['steps']}, FPS: {stats['fps']:.2f}")
            if is_success:
                print(f"[INFO] Trial {trial} succeeded! Generating video...")
                save_success_video(traj, trial, video_save_dir, fps=20)
                success_count += 1
            current_total = trial + 1
            current_success_rate = (success_count / current_total) * 100
            avg_steps = total_steps / current_total  # 计算平均步数
            print(f"[RESULT] Trial {trial}: {'SUCCESS' if is_success else 'FAILED'}")
            print(f"[STATS]  Steps: {this_trial_steps} | Avg Steps: {avg_steps:.1f} | FPS: {stats['fps']:.2f}")
            print(f"[RATE]   Current Success Rate: {current_success_rate:.2f}% ({success_count}/{current_total})")
            
            # ----------------------------

            print(f"[INFO] Elapsed time: {stats['elapsed']:.3f}s, Steps: {stats['steps']}, FPS: {stats['fps']:.2f}")
            del traj 
            del stats

    print(f"\nSuccessful trials: {results.count(True)}, out of {len(results)} trials")
    print(f"Success rate: {results.count(True) / len(results)}")
    print(f"Trial Results: {results}\n")

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()