from configs.state_vec import STATE_VEC_IDX_MAPPING

# isaaclab 输出的原始动作的 含义 以及 对应索引
ISAACLAB_RAW_ACTION_SLICE = {
    "left_pos": slice(0, 3),
    "left_quat_wxyz": slice(3, 7),
    "left_gripper": slice(7, 8),

    "right_pos": slice(8, 11),
    "right_quat_wxyz": slice(11, 15),
    "right_gripper": slice(15, 16),
}

# isaaclab 的 6D 旋转表示时候的动作索引
ISAACLAB_ROT6D_ACTION_SLICE = {
    "left_pos": slice(0, 3),
    "left_rot6d": slice(3, 9),
    "left_gripper": slice(9, 10),

    "right_pos": slice(10, 13),
    "right_rot6d": slice(13, 19),
    "right_gripper": slice(19, 20),
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
    "joint_pos_right",      # 右臂关节 7  
    "eef_pos_right_b",      # 右手位置 3
    "eef_quat_right_b",     # 右手姿态四元数 4
    "gripper_right_pos",    # 右手夹爪关节 2
    "joint_pos_left",       # 左臂关节 7
    "eef_pos_left_b",       # 左手位置 3
    "eef_quat_left_b",      # 左手姿态四元数 4
    "gripper_left_pos",     # 左手夹爪关节 2
]


# isaaclab的观测数据对应的state vector索引，将isaaclab的观测数据映射到统一的state vector中对应的位置
# rdt的统一动作空间满足 先右后左，而我们设置的Isaaclab双臂动作满足先左后右
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