from configs.state_vec import STATE_VEC_IDX_MAPPING

# isaaclab 输出的原始动作的 含义 以及 对应索引
ISAACLAB_RAW_ACTION_SLICE = {
    "right_pos": slice(0, 3),
    "right_quat_wxyz": slice(3, 7),
    "right_gripper": slice(7, 8),

    "left_pos": slice(8, 11),
    "left_quat_wxyz": slice(11, 15),
    "left_gripper": slice(15, 16),
}

# isaaclab 的 6D 旋转表示时候的动作索引
ISAACLAB_ROT6D_ACTION_SLICE = {
    "right_pos": slice(0, 3),
    "right_rot6d": slice(3, 9),
    "right_gripper": slice(9, 10),

    "left_pos": slice(10, 13),
    "left_rot6d": slice(13, 19),
    "left_gripper": slice(19, 20),
}


# 左右夹爪的最大宽度
LEFT_GRIPPER_MAX = 0.04
RIGHT_GRIPPER_MAX = 0.04
# rdt 中的 相机与 isaaclab相机 之间的名称映射关系
CAMERA_MAPPING= {
    "cam_high": "zed_left",
    "cam_left_wrist": "wrist_cam_left",
    "cam_right_wrist": "wrist_cam_right",
}
# isaaclab 数据集的本体感知 key
ISAACLAB_PROPRIO_KEYS = [
    "eef_pos_left_b",
    "eef_quat_left_b",
    "gripper_left_pos",
    "eef_pos_right_b",
    "eef_quat_right_b",
    "gripper_right_pos",
    "joint_pos_left",
    "joint_pos_right",
]


# isaaclab的观测数据对应的state vector索引，将isaaclab的观测数据映射到统一的state vector中对应的位置
ISAACLAB_PROPRIO_INDICES = [
    *[STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(7)],
    STATE_VEC_IDX_MAPPING["right_eef_pos_x"],
    STATE_VEC_IDX_MAPPING["right_eef_pos_y"],
    STATE_VEC_IDX_MAPPING["right_eef_pos_z"],
    *[STATE_VEC_IDX_MAPPING[f"right_eef_angle_{i}"] for i in range(6)],
    STATE_VEC_IDX_MAPPING["right_gripper_open"],

    *[STATE_VEC_IDX_MAPPING[f"left_arm_joint_{i}_pos"] for i in range(7)],
    STATE_VEC_IDX_MAPPING["left_eef_pos_x"],
    STATE_VEC_IDX_MAPPING["left_eef_pos_y"],
    STATE_VEC_IDX_MAPPING["left_eef_pos_z"],
    *[STATE_VEC_IDX_MAPPING[f"left_eef_angle_{i}"] for i in range(6)],
    STATE_VEC_IDX_MAPPING["left_gripper_open"],
]

# isaaclab的动作空间对应的state vector索引，使用其将模型预测的动作映射回isaaclab的动作空间
ISAACLAB_ACTION_INDICES = [
    # 0:3
    STATE_VEC_IDX_MAPPING["right_eef_pos_x"],
    STATE_VEC_IDX_MAPPING["right_eef_pos_y"],
    STATE_VEC_IDX_MAPPING["right_eef_pos_z"],
    # 3:9
    *[STATE_VEC_IDX_MAPPING[f"right_eef_angle_{i}"] for i in range(6)],
    # 9
    STATE_VEC_IDX_MAPPING["right_gripper_open"],
    # 10:13
    STATE_VEC_IDX_MAPPING["left_eef_pos_x"],
    STATE_VEC_IDX_MAPPING["left_eef_pos_y"],
    STATE_VEC_IDX_MAPPING["left_eef_pos_z"],
    # 13:19
    *[STATE_VEC_IDX_MAPPING[f"left_eef_angle_{i}"] for i in range(6)],
    # 19
    STATE_VEC_IDX_MAPPING["left_gripper_open"],
]