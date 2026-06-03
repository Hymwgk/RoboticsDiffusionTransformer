#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 isaaclab 输出的 HDF5 数据转换为 RDT 微调 HDF5VLADataset 可读取的 episode 目录结构。

输入isaaclab数据特点：
- input_root/*.hdf5
- 每个 HDF5 文件对应一个任务
- 每个文件内部包含多个 demo:  data/demo_0, data/demo_1, ...

输出RDT格式数据集特点：
- 每个 任务回合 导出为一个独立 episode 文件夹
- 每个 episode 文件夹包含：
    data.hdf5
    expanded_instruction_gpt-4-turbo.json

输出结构示例：
    /data/rdt_js/
      episode_000/
        data.hdf5
        expanded_instruction_gpt-4-turbo.json
      episode_001/
        data.hdf5
        expanded_instruction_gpt-4-turbo.json

每个episode文件夹中的 data.hdf5 内部结构：
    data.hdf5
    ├── observations
    │   ├── qpos
    │   └── images
    │       ├── cam_high
    │       ├── cam_left_wrist
    │       └── cam_right_wrist
    └── action
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
from pathlib import Path

import cv2
import h5py
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

LEFT_GRIPPER_MAX = 0.04
RIGHT_GRIPPER_MAX = 0.04


# 相机字段映射：isaaclab数据集 obs 中的字段 -> 输出 data.hdf5 中的字段
CAMERA_MAPPING= {
    "cam_high": "zed_left",
    "cam_left_wrist": "wrist_cam_left",
    "cam_right_wrist": "wrist_cam_right",
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Isaaclab multi-demo HDF5 files to RDT per-episode HDF5 format."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("/data/isaaclab_js"),
        help="原始 Isaaclab HDF5 所在目录。",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/rdt_js"),
        help="导出的 RDT episode 根目录(数据集目录)。",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="rdt_js",
        help="数据集名称。这里只用于记录和组织，建议与 configs 中注册的数据集名称一致。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="如果 episode 目录已存在，是否覆盖 data.hdf5 和 instruction json。",
    )
    parser.add_argument(
        "--max-demos-per-task",
        type=int,
        default=None,
        help="仅用于快速调试：限制每个任务最多导出多少条 demo。",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="图像保存为 jpg bytes 时的编码质量。",
    )
    return parser.parse_args()


def infer_instruction_payload(task_stem: str, instructions: dict[str, dict[str, str]]) -> dict[str, str]:
    """根据任务文件名task_stem, 自动生成一份语言指令 JSON 内容"""
    canonical_stem = task_stem.removesuffix("-merged")

    if canonical_stem in instructions:
        instruction = instructions[canonical_stem]['instruction']
    else:
        instruction = canonical_stem.replace("-", " ").replace("_", " ").strip()
        instruction = " ".join(instruction.split())
        if instruction:
            instruction = instruction[:1].upper() + instruction[1:] + "."
        else:
            instruction = "Perform the task."

    return {
        "instruction": instruction,
        "simplified_instruction": instruction,
        "expanded_instruction": instruction,
    }


def write_instruction_json(episode_dir: Path, payload: dict[str, str]) -> None:
    """将指令 payload 写成 expanded_instruction_gpt-4-turbo.json。"""
    target_path = episode_dir / "expanded_instruction_gpt-4-turbo.json"
    with target_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)


def iter_source_files(input_root: Path) -> list[Path]:
    hdf5_files: list[Path] = []
    for root, _, files in os.walk(input_root):
        for filename in fnmatch.filter(files, "*.hdf5"):
            hdf5_files.append(Path(root) / filename)
    return sorted(hdf5_files)


def sorted_demo_keys(file_obj: h5py.File) -> list[str]:
    if "data" not in file_obj:
        raise KeyError("输入 HDF5 中没有 'data' group, 无法找到 data/demo_x。")

    return sorted(
        file_obj["data"].keys(),
        key=lambda name: int(name.split("_")[-1]) if name.split("_")[-1].isdigit() else name,
    )


def encode_image_to_jpg_bytes(image: np.ndarray, jpeg_quality: int = 95) -> np.ndarray:
    """
    将 RGB/BGR 图像编码成 jpg bytes。

    注意：
    - 当前 HDF5VLADataset 的 parse_img() 使用 cv2.imdecode(..., cv2.IMREAD_COLOR)
    - 所以这里保存为变长 uint8 bytes。
    """
    image = np.asarray(image)

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    # 如果是灰度图，转成三通道
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # 如果是 RGBA，去掉 alpha
    if image.ndim == 3 and image.shape[-1] == 4:
        image = image[..., :3]

    ok, encoded = cv2.imencode(
        ".jpg",
        image,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
    )
    if not ok:
        raise RuntimeError("图像 jpg 编码失败。")

    return encoded.astype(np.uint8)


def write_encoded_image_sequence(
    image_group: h5py.Group,
    key: str,
    images: np.ndarray,
    jpeg_quality: int,
) -> None:
    """
    写入变长 uint8 图像序列。

    输出形态类似：
        observations/images/cam_high[i] -> 一帧 jpg bytes
    """
    images = np.asarray(images)
    num_steps = images.shape[0]

    vlen_uint8 = h5py.vlen_dtype(np.dtype("uint8"))
    dataset = image_group.create_dataset(key, shape=(num_steps,), dtype=vlen_uint8)

    for i in range(num_steps):
        dataset[i] = encode_image_to_jpg_bytes(images[i], jpeg_quality=jpeg_quality)


def reduce_gripper(gripper: np.ndarray, max_value: float) -> np.ndarray:

    gripper = np.asarray(gripper, dtype=np.float32)

    if gripper.ndim == 1:

        gripper = gripper[:, None]

    elif gripper.ndim == 2 and gripper.shape[-1] == 2:

        gripper = np.mean(np.abs(gripper), axis=-1, keepdims=True)

    return np.clip(gripper / max_value, 0.0, 1.0).astype(np.float32)

def quat_wxyz_to_rot6d(quat_wxyz: np.ndarray) -> np.ndarray:

    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float32)

    quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]

    rotmat = R.from_quat(quat_xyzw).as_matrix()      # (T,3,3)

    rot6d = rotmat[:, :, :2].transpose(0, 2, 1).reshape(-1, 6)

    return rot6d.astype(np.float32)

def build_action_from_raw_action(raw_action: np.ndarray) -> np.ndarray:
    """
    raw_action: (T,16)
      right_pos(3) + right_quat_wxyz(4) + right_gripper(1)
      left_pos(3)  + left_quat_wxyz(4)  + left_gripper(1)
    return: (T,20)
      right_pos(3) + right_rot6d(6) + right_gripper(1)
    left_pos(3)  + left_rot6d(6)  + left_gripper(1)
    """
    raw_action = np.asarray(raw_action, dtype=np.float32)
    assert raw_action.ndim == 2, raw_action.shape
    assert raw_action.shape[-1] == 16, raw_action.shape
    right_pos = raw_action[:, 0:3] 
    right_quat = raw_action[:, 3:7] # wxyz
    # 夹爪动作归一化到 [0,1]
    right_gripper = np.clip(raw_action[:, 7:8], 0.0, 1.0).astype(np.float32) 

    left_pos = raw_action[:, 8:11]
    left_quat = raw_action[:, 11:15] # wxyz
    left_gripper = np.clip(raw_action[:, 15:16], 0.0, 1.0).astype(np.float32)
    # 转换为6d姿态
    right_rot6d = quat_wxyz_to_rot6d(right_quat)
    left_rot6d = quat_wxyz_to_rot6d(left_quat)

    action = np.concatenate(
        [
            right_pos,
            right_rot6d,
            right_gripper,
            left_pos,
            left_rot6d,
            left_gripper,
        ],
        axis=-1,
    ).astype(np.float32)
    assert action.shape[-1] == 20, action.shape
    return action


def build_qpos_from_obs(obs: h5py.Group) -> np.ndarray:
    """
    构造 observations/qpos。

    优先级：
    1. 如果原始 obs 中已有 qpos，直接使用 obs["qpos"]
    2. 否则根据 EEF pose + gripper 拼一个状态向量：
       left:  eef_pos_left_b  + eef_quat_left_b  + gripper_left_pos
       right: eef_pos_right_b + eef_quat_right_b + gripper_right_pos

    注意：
    - 这一步得到的不一定是当前 HDF5VLADataset 示例中的 14 维关节状态。
    - 如果你采用 EEF pose 表示，后续还需要修改 hdf5_vla_dataset.py 里的 fill_in_state()
      以及旋转四元数转 6D 的逻辑。
    """
    if "qpos" in obs:
        return np.asarray(obs["qpos"], dtype=np.float32)

    required_keys = [
        "eef_pos_left_b",
        "eef_quat_left_b",
        "gripper_left_pos",
        "eef_pos_right_b",
        "eef_quat_right_b",
        "gripper_right_pos",
        "joint_pos_left",
        "joint_pos_right",
    ]
    missing = [k for k in required_keys if k not in obs]
    if missing:
        raise KeyError(
            "无法构造 qpos。原始 obs 中既没有 qpos，也缺少这些 EEF 字段："
            + ", ".join(missing)
        )
    # 只要手臂关节不要gripper关节
    right_arm_joints = np.asarray(obs["joint_pos_right"][:, :7], dtype=np.float32)
    left_arm_joints = np.asarray(obs["joint_pos_left"][:, :7], dtype=np.float32)

    left_pos = np.asarray(obs["eef_pos_left_b"], dtype=np.float32)
    right_pos = np.asarray(obs["eef_pos_right_b"], dtype=np.float32)

    # TODO 确定一下isaaclab数据集四元数顺序wxyz
    right_rot6d = quat_wxyz_to_rot6d(obs["eef_quat_right_b"])
    left_rot6d = quat_wxyz_to_rot6d(obs["eef_quat_left_b"])
    # 不一定是0/1  但是归一化到0~1
    right_gripper = reduce_gripper(obs["gripper_right_pos"], RIGHT_GRIPPER_MAX)
    left_gripper = reduce_gripper(obs["gripper_left_pos"], LEFT_GRIPPER_MAX)



    qpos = np.concatenate(
        [
            right_arm_joints,
            right_pos,
            right_rot6d,  # 6D 表示
            right_gripper,
            left_arm_joints,
            left_pos,
            left_rot6d,
            left_gripper,

        ],
        axis=-1,
    ).astype(np.float32)
    assert qpos.shape[-1] == 34, qpos.shape
    return qpos


def copy_optional_obs_fields(obs: h5py.Group, out_obs: h5py.Group) -> None:
    """
    额外复制一些原始 EEF 字段，方便你后续改 parse_hdf5_file() 时直接读取。
    这些字段不是当前官方 HDF5VLADataset 示例必须的，但保留下来更安全。
    """
    optional_keys = [
        "eef_pos_left_b",
        "eef_pos_left_w",
        "eef_pos_right_b",
        "eef_pos_right_w",
        "eef_quat_left_b",
        "eef_quat_left_w",
        "eef_quat_right_b",
        "eef_quat_right_w",
        "gripper_left_pos",
        "gripper_right_pos",
    ]

    for key in optional_keys:
        if key in obs:
            out_obs.create_dataset(key, data=np.asarray(obs[key]))


def convert_one_demo_to_episode(
    *,
    episode: h5py.Group,
    episode_dir: Path,
    instruction_payload: dict[str, str],
    overwrite: bool,
    jpeg_quality: int,
) -> None:
    """
    将单个 demo 写成：
        episode_dir/data.hdf5
        episode_dir/expanded_instruction_gpt-4-turbo.json
    """
    episode_dir.mkdir(parents=True, exist_ok=True)

    data_hdf5_path = episode_dir / "data.hdf5"
    instruction_path = episode_dir / "expanded_instruction_gpt-4-turbo.json"

    if data_hdf5_path.exists() and instruction_path.exists() and not overwrite:
        return

    obs = episode["obs"]
    # 构造 action
    actions = build_action_from_raw_action(episode["actions"])
    # 构造 qpos
    qpos = build_qpos_from_obs(obs)

    num_steps = int(actions.shape[0])
    if qpos.shape[0] != num_steps:
        min_len = min(qpos.shape[0], num_steps)
        qpos = qpos[:min_len]
        actions = actions[:min_len]
        num_steps = min_len
    # 
    with h5py.File(data_hdf5_path, "w") as f_out:
        obs_group = f_out.create_group("observations")
        image_group = obs_group.create_group("images")

        obs_group.create_dataset("qpos", data=qpos.astype(np.float32))
        f_out.create_dataset("action", data=actions.astype(np.float32))

        # 保留 EEF 原始字段，方便后续自定义 Dataset 时使用
        copy_optional_obs_fields(obs, obs_group)

        # 映射三路相机
        # 当前原始字段：
        #   zed_left         -> cam_high
        #   wrist_cam_left   -> cam_left_wrist
        #   wrist_cam_right  -> cam_right_wrist
        camera_mapping = CAMERA_MAPPING

        for out_key, src_key in camera_mapping.items():
            if src_key not in obs:
                raise KeyError(f"原始 obs 中缺少相机字段：{src_key}")

            images = np.asarray(obs[src_key])
            if images.shape[0] != num_steps:
                images = images[:num_steps]

            write_encoded_image_sequence(
                image_group=image_group,
                key=out_key,
                images=images,
                jpeg_quality=jpeg_quality,
            )

    write_instruction_json(episode_dir, instruction_payload)


def convert_all(
    *,
    input_root: Path,
    output_root: Path,
    instructions: dict[str, str],
    dataset_name: str,
    overwrite: bool,
    max_demos_per_task: int | None,
    jpeg_quality: int,
) -> None:
    # 创建输出文件夹
    output_root.mkdir(parents=True, exist_ok=True)
    # 遍历输入文件夹，找到所有 HDF5 文件
    source_files = iter_source_files(input_root)
    if not source_files:
        raise FileNotFoundError(f"No .hdf5 files found under: {input_root}")

    episode_counter = 0
    summary: dict[str, dict[str, int]] = {}
    # 处理所有任务hdf5
    for source_path in source_files:
        task_stem = source_path.stem
        # 根据任务名称自动生成语言指令内容
        instruction_payload = infer_instruction_payload(task_stem, instructions)

        written = 0
        skipped = 0
        # 打开单个任务 HDF5 
        with h5py.File(source_path, "r") as f:
            # 获取该任务hdf5 中所有 demo 的 key，并排序（demo_0, demo_1, ...）
            demo_keys = sorted_demo_keys(f)
            if max_demos_per_task is not None:
                demo_keys = demo_keys[:max_demos_per_task]
            # 循环处理每条 demo，写成独立 episode
            for demo_key in tqdm(demo_keys, desc=f"convert {task_stem}", leave=False):
                episode_dir = output_root / f"episode_{episode_counter:06d}"
                data_hdf5_path = episode_dir / "data.hdf5"

                if data_hdf5_path.exists() and not overwrite:
                    skipped += 1
                    episode_counter += 1
                    continue

                episode = f["data"][demo_key]
                # 处理单条 demo，写成一个 episode 文件夹
                convert_one_demo_to_episode(
                    episode=episode,
                    episode_dir=episode_dir,
                    instruction_payload=instruction_payload,
                    overwrite=overwrite,
                    jpeg_quality=jpeg_quality,
                )

                written += 1
                episode_counter += 1

        summary[source_path.name] = {
            "written": written,
            "skipped": skipped,
        }

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(
        f"done: dataset_name={dataset_name}, "
        f"episodes={episode_counter}, output_root={output_root}"
    )


def main() -> None:
    args = parse_args()
    # 获取语言指令路径
    instructions_path = Path(args.input_root) / "Instructions.json"
    with open(instructions_path, "r") as f:
        instructions = json.load(f)
    # 
    convert_all(
        input_root=args.input_root,
        output_root=args.output_root,
        instructions=instructions, 
        dataset_name=args.dataset_name,
        overwrite=args.overwrite,
        max_demos_per_task=args.max_demos_per_task,
        jpeg_quality=args.jpeg_quality,
    )


if __name__ == "__main__":
    main()