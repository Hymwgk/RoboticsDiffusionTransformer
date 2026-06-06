# 将RDT模型包装为适合Isaac Lab环境接口的形式，提供模型初始化、指令编码和模型推理的功能。
import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import json
from pathlib import Path
from collections import deque
import torch.nn.functional as F
from configs.state_vec import STATE_VEC_IDX_MAPPING
from models.multimodal_encoder.siglip_encoder import SiglipVisionTower
from models.multimodal_encoder.t5_encoder import T5Embedder
from models.rdt_runner import RDTRunner

from configs.isaaclab_const import LEFT_GRIPPER_MAX, RIGHT_GRIPPER_MAX, ISAACLAB_PROPRIO_INDICES, \
    ISAACLAB_ACTION_INDICES, CAMERA_MAPPING, ISAACLAB_PROPRIO_KEYS,ISAACLAB_ROT6D_ACTION_SLICE






def create_model(args, pretrained, **kwargs):
    model = RoboticDiffusionTransformerModel(args, **kwargs)
    if pretrained is not None:
        model.load_pretrained_weights(pretrained)
    return model



class RoboticDiffusionTransformerModel(object):
    """A wrapper for the RDT model, which handles
            1. Model initialization
            2. Encodings of instructions
            3. Model inference
    """
    def __init__(
        self, args, 
        device='cuda',
        dtype=torch.bfloat16,
        image_size=None,
        control_frequency=30,
        pretrained_text_encoder_name_or_path=None,
        pretrained_vision_encoder_name_or_path=None,
        dataset_name='rdt_js',
    ):
        self.args = args
        self.dtype = dtype
        self.image_size = image_size
        self.device = device
        self.control_frequency = control_frequency
        if pretrained_text_encoder_name_or_path is not None:
            self.text_tokenizer, self.text_model = self.get_text_encoder(pretrained_text_encoder_name_or_path)
        else:
            self.text_tokenizer, self.text_model = None, None
        self.image_processor, self.vision_model = self.get_vision_encoder(pretrained_vision_encoder_name_or_path)
        self.policy = self.get_policy()

        self.obs_window = None

        self.eval_mode()
        self.reset()
    


    def _rot6d_to_wxyz_quat(self, rot6d):
        """
        rot6d: (..., 6)
        return: (..., 4), quat in wxyz format
        """
        # 6D rotation -> rotation matrix
        a1 = rot6d[..., 0:3]
        a2 = rot6d[..., 3:6]

        b1 = torch.nn.functional.normalize(a1, dim=-1)
        b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
        b2 = torch.nn.functional.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)

        # R: (..., 3, 3)
        R = torch.stack([b1, b2, b3], dim=-1)

        # rotation matrix -> quaternion, wxyz
        m00 = R[..., 0, 0]
        m01 = R[..., 0, 1]
        m02 = R[..., 0, 2]
        m10 = R[..., 1, 0]
        m11 = R[..., 1, 1]
        m12 = R[..., 1, 2]
        m20 = R[..., 2, 0]
        m21 = R[..., 2, 1]
        m22 = R[..., 2, 2]

        qw = torch.sqrt(torch.clamp(1.0 + m00 + m11 + m22, min=1e-8)) / 2.0
        qx = (m21 - m12) / (4.0 * qw)
        qy = (m02 - m20) / (4.0 * qw)
        qz = (m10 - m01) / (4.0 * qw)

        quat = torch.stack([qw, qx, qy, qz], dim=-1)
        quat = torch.nn.functional.normalize(quat, dim=-1)

        return quat

    def get_policy(self):
        """Initialize the model."""
        # Initialize model with arguments
        img_cond_len = (self.args["common"]["img_history_size"] 
                        * self.args["common"]["num_cameras"] 
                        * self.vision_model.num_patches)
        
        _model = RDTRunner(
            action_dim=self.args["common"]["state_dim"],
            pred_horizon=self.args["common"]["action_chunk_size"],
            config=self.args["model"],
            lang_token_dim=self.args["model"]["lang_token_dim"],
            img_token_dim=self.args["model"]["img_token_dim"],
            state_token_dim=self.args["model"]["state_token_dim"],
            max_lang_cond_len=self.args["dataset"]["tokenizer_max_length"],
            img_cond_len=img_cond_len,
            img_pos_embed_config=[
                # No initial pos embed in the last grid size
                # since we've already done in ViT
                ("image", (self.args["common"]["img_history_size"], 
                    self.args["common"]["num_cameras"], 
                    -self.vision_model.num_patches)),  
            ],
            lang_pos_embed_config=[
                # Similarly, no initial pos embed for language
                ("lang", -self.args["dataset"]["tokenizer_max_length"]),
            ],
            dtype=self.dtype,
        )

        return _model

    def get_text_encoder(self, pretrained_text_encoder_name_or_path):
        text_embedder = T5Embedder(from_pretrained=pretrained_text_encoder_name_or_path, 
                                   model_max_length=self.args["dataset"]["tokenizer_max_length"], 
                                   device=self.device)
        tokenizer, text_encoder = text_embedder.tokenizer, text_embedder.model
        return tokenizer, text_encoder

    def get_vision_encoder(self, pretrained_vision_encoder_name_or_path):
        vision_encoder = SiglipVisionTower(vision_tower=pretrained_vision_encoder_name_or_path, args=None)
        image_processor = vision_encoder.image_processor
        return image_processor, vision_encoder


    def eval_mode(self):
        self.policy.eval()
        self.vision_model.eval()
        self.policy = self.policy.to(self.device, dtype=self.dtype)
        self.vision_model = self.vision_model.to(self.device, dtype=self.dtype)

    def reset(self):
        """Set model to evaluation mode.
        """
        # 初始化图像历史
        self.img_history_size = self.args["common"]["img_history_size"]
        self.num_cameras = self.args["common"]["num_cameras"]
        # 注意顺序！
        self.camera_keys = ["cam_high", "cam_right_wrist", "cam_left_wrist"]
        self.obs_window = deque(maxlen=self.img_history_size)

    def _extract_current_images(self, obs_dict):
        """从isaaclab的观测中抽取图像"""
        current_images = {}
        for rdt_cam_name, isaac_cam_name in CAMERA_MAPPING.items():
            if isaac_cam_name not in obs_dict:
                current_images[rdt_cam_name] = None
                continue
            # hwc
            img = obs_dict[isaac_cam_name]
            # 转换一下 TODO 检查训练时候的顺序
            img = self._to_uint8_hwc(img)

            current_images[rdt_cam_name] = img

        return current_images

    def _to_uint8_hwc(self, img):
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu()

            while img.ndim > 3:
                img = img.squeeze(0)

            img = img.numpy()

        img = np.asarray(img)

        # CHW -> HWC
        if img.ndim == 3 and img.shape[0] in [1, 3, 4] and img.shape[-1] not in [3, 4]:
            img = np.transpose(img, (1, 2, 0))

        # RGBA -> RGB/BGR 三通道
        if img.ndim == 3 and img.shape[-1] == 4:
            img = img[..., :3]

        # gray -> 3 channels
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=-1)

        if np.issubdtype(img.dtype, np.floating):
            if img.max() <= 1.0:
                img = img * 255.0
            img = np.clip(img, 0, 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)

        return img


    def _build_images_from_obs(self, obs_dict):
        """从历史窗口构建rdt需要的image list"""

        # 第一次 step 时，补齐历史窗口
        if len(self.obs_window) == 0:
            current_images = self._extract_current_images(obs_dict)
            for _ in range(self.img_history_size):
                self.obs_window.append(current_images)
        else:
            current_images = self._extract_current_images(obs_dict)
            self.obs_window.append(current_images)


        images = []

        for frame in self.obs_window:
            for cam_key in self.camera_keys:
                img = frame.get(cam_key, None)

                if img is None:
                    images.append(None)
                else:
                    images.append(Image.fromarray(img))

        return images


    def _reduce_gripper(self, gripper: torch.Tensor, max_value: float) -> torch.Tensor:
        """将obs的2个夹爪关节 映射为 1维度 0/1
        支持输入形状: [2,] 或 [B, 2]
        """
        # 1. 统一转换为 2 维张量处理 [2,] -> [1, 2]
        is_1d = (gripper.ndim == 1)
        if is_1d:
            gripper = gripper.unsqueeze(0) # 注意：是 unsqueeze(0) 变成 [1, 2]，而不是 unsqueeze(-1)

        # 2. 此时无论是单样本还是 Batch，形状都是 [B, 2]，可以安全地做 abs 和 mean
        # [B, 2] -> [B, 1]
        gripper_mean = torch.mean(
            torch.abs(gripper),
            dim=-1,
            keepdim=True,
        )

        # 3. 缩放到 0 ~ 1 之间
        gripper_norm = torch.clamp(
            gripper_mean / max_value,
            min=0.0,
            max=1.0,
        )

        # 4. 如果当初输入的是单向量 [2,]，则把 Batch 维度压扁，返回 [1,] 对应的 0/1 值
        if is_1d:
            return gripper_norm.squeeze(0)
            
        return gripper_norm # 返回 [B, 1]
  

    def _quat_wxyz_to_rot6d(self, quat_wxyz):
        """ 将wxyz顺序的四元数转换为6d表征
        quat_wxyz:
            [4,] 或 [B, 4]

        return:
            [6,] 或 [B, 6]
        """
        # 1. 归一化四元数
        quat_wxyz = F.normalize(quat_wxyz, dim=-1)
        w, x, y, z = quat_wxyz.unbind(-1)
        
        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z
        wx = w * x
        wy = w * y
        wz = w * z

        # 2. 构造旋转矩阵的 9 个元素
        # 使用 dim=-1 可以天然支持单向量 [4,] 和 批处理 [B, 4]
        rot = torch.stack([
            1 - 2*(yy + zz),
            2*(xy - wz),
            2*(xz + wy),

            2*(xy + wz),
            1 - 2*(xx + zz),
            2*(yz - wx),

            2*(xz - wy),
            2*(yz + wx),
            1 - 2*(xx + yy),
        ], dim=-1)
        
        # 3. 恢复成旋转矩阵 [3, 3] 或 [B, 3, 3]
        if quat_wxyz.dim() == 1:
            rotmat = rot.reshape(3, 3)
            # 提取前两列。注意：这里去掉了错误的 transpose(1, 2)
            # rotmat[:, :2] 的形状是 [3, 2]，直接 reshape(6) 展开
            rot6d = rotmat[:, :2].reshape(6)
        else:
            # 如果是 Batch 输入 [B, 4]
            rotmat = rot.reshape(-1, 3, 3)
            # 提取前两列 [B, 3, 2] 并展平成 [B, 6]
            rot6d = rotmat[:, :, :2].reshape(-1, 6)
            
        return rot6d


    def _build_proprio_from_obs(self, obs_dict: dict) -> torch.Tensor:
        """
        输入:
            obs_dict:
                joint_pos_right      [8]
                joint_pos_left       [8]
                eef_pos_right_b      [3]
                eef_pos_left_b       [3]
                eef_quat_right_b     [4]
                eef_quat_left_b      [4]
                gripper_right_pos    [2]
                gripper_left_pos     [2]

        输出:
            proprio [B, 34]
        """
        missing = [k for k in ISAACLAB_PROPRIO_KEYS if k not in obs_dict]
        if missing:
            raise KeyError(
                "无法构造 proprio。原始 obs 中既没有 proprio, 也缺少这些 EEF 字段："
                + ", ".join(missing)
            )
        
        right_arm_joints = obs_dict["joint_pos_right"][..., :7]
        left_arm_joints = obs_dict["joint_pos_left"][..., :7]

        right_pos = obs_dict["eef_pos_right_b"]
        left_pos = obs_dict["eef_pos_left_b"]

        right_rot6d = self._quat_wxyz_to_rot6d(obs_dict["eef_quat_right_b"])
        left_rot6d = self._quat_wxyz_to_rot6d(obs_dict["eef_quat_left_b"])

        right_gripper = self._reduce_gripper(
            obs_dict["gripper_right_pos"],
            RIGHT_GRIPPER_MAX,
        )

        left_gripper = self._reduce_gripper(
            obs_dict["gripper_left_pos"],
            LEFT_GRIPPER_MAX,
        )

        proprio = torch.cat(
            [
                right_arm_joints,     # 7
                right_pos,            # 3
                right_rot6d,          # 6
                right_gripper,        # 1

                left_arm_joints,      # 7
                left_pos,             # 3
                left_rot6d,           # 6
                left_gripper,         # 1
            ],
            dim=-1,
        )
        assert proprio.shape[-1] == 34, proprio.shape
        return proprio




    def load_pretrained_weights(self, pretrained=None):
        if pretrained is None:
            return 
        print(f'Loading weights from {pretrained}')
        filename = os.path.basename(pretrained)
        if filename.endswith('.pt'):
            checkpoint =  torch.load(pretrained)
            self.policy.load_state_dict(checkpoint["module"])
        elif filename.endswith('.safetensors'):
            from safetensors.torch import load_model
            load_model(self.policy, pretrained)
        else:
            raise NotImplementedError(f"Unknown checkpoint format: {pretrained}")

    def _isaac_proprio_to_unformat_state(self, proprio):
        """
        将从isaaclab获取的本体观测信息  转换到 统一动作空间中
        Format the robot joint state into the unified state vector.

        Args:
            proprio (torch.Tensor): The proprio state to be formatted. 
                proprio ([B, N, 34]).

        Returns:
            state (torch.Tensor): The formatted state for RDT ([B, N, 128]). 
        """
        B, N, _ = proprio.shape
        state = torch.zeros(
            (B, N, self.args["model"]["state_token_dim"]), 
            device=proprio.device, dtype=proprio.dtype
        )
        # assemble the unifed state vector
        state[:, :, ISAACLAB_PROPRIO_INDICES] = proprio
        state_elem_mask = torch.zeros(
            (B, self.args["model"]["state_token_dim"]),
            device=proprio.device, dtype=proprio.dtype
        )
        state_elem_mask[:, ISAACLAB_PROPRIO_INDICES] = 1
        return state, state_elem_mask

    def _unformat_action_to_isaac_action(self, unformat_action):
        # 将统一动作空间  转换  为isaaclab动作空间  [B,chunk,N]
        isaac_rot6d_action = unformat_action[:, :, ISAACLAB_ACTION_INDICES]
        # 将动作转换为isaaclab动作，主要是6d动作转换
        right_pos = isaac_rot6d_action[:, :, ISAACLAB_ROT6D_ACTION_SLICE['right_pos']]
        right_rot6d = isaac_rot6d_action[:,:,ISAACLAB_ROT6D_ACTION_SLICE['right_rot6d']]
        right_gripper = isaac_rot6d_action[:, :, ISAACLAB_ROT6D_ACTION_SLICE['right_gripper']]

        left_pos = isaac_rot6d_action[:, :, ISAACLAB_ROT6D_ACTION_SLICE['left_pos']]
        left_rot6d = isaac_rot6d_action[:,:,ISAACLAB_ROT6D_ACTION_SLICE['left_rot6d']]
        left_gripper = isaac_rot6d_action[:, :, ISAACLAB_ROT6D_ACTION_SLICE['left_gripper']]
        # 左右手的动作处理
        right_wxyz_quat = self._rot6d_to_wxyz_quat(right_rot6d)
        left_wxyz_quat = self._rot6d_to_wxyz_quat(left_rot6d)
        # TODO gripper 的动作要转换么？ 检查一下
        # 拼接起来
        isaac_action = torch.cat(
            [
                right_pos,
                right_wxyz_quat,
                right_gripper,
                left_pos,
                left_wxyz_quat,
                left_gripper,
            ],
            dim=-1
        )       
        assert isaac_action.shape[-1] == 20, isaac_action.shape
        # denormalize to action space 不进行缩放 TODO 检查一下在微调的时候是否使用的就是归一化动作？
        return isaac_action

    @torch.no_grad()
    def step(self, obs_dict, text_embeds):
        """
        Args:
            proprio: proprioceptive states
            images: RGB images
            text_embeds: instruction embeddings
        Returns:
            action: predicted action
        """
        device = self.device
        dtype = self.dtype
        # 对图像
        background_color = np.array([
            int(x*255) for x in self.image_processor.image_mean
        ], dtype=np.uint8).reshape(1, 1, 3)
        background_image = np.ones((
            self.image_processor.size["height"], 
            self.image_processor.size["width"], 3), dtype=np.uint8
        ) * background_color
        
        # 获取图像观测列表
        images = self._build_images_from_obs(obs_dict)
        image_tensor_list = []
        # images 本身应该是一个列表
        for image in images:
            if image is None:
                # 缺失的相机观测用背景图补齐
                image = Image.fromarray(background_image)
            # 改变形状尺寸
            if self.image_size is not None:
                image = transforms.Resize(self.image_size)(image)
            # TODO 检查这里的参数含义是否和训练时候一致
            if self.args["dataset"].get("auto_adjust_image_brightness", False):
                pixel_values = list(image.getdata())
                average_brightness = sum(sum(pixel) for pixel in pixel_values) / (len(pixel_values) * 255.0 * 3)
                if average_brightness <= 0.15:
                    image = transforms.ColorJitter(brightness=(1.75,1.75))(image)
            # 图像 pad 成正方形
            if self.args["dataset"].get("image_aspect_ratio", "pad") == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in self.image_processor.image_mean))
            # 用siglip自带的图像预处理器进行图像预处理
            image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            image_tensor_list.append(image)
        # 进行图像堆叠
        image_tensor = torch.stack(image_tensor_list, dim=0).to(device, dtype=dtype)
        # 对图像进行编码
        image_embeds = self.vision_model(image_tensor).detach()
        image_embeds = image_embeds.reshape(-1, self.vision_model.hidden_size).unsqueeze(0)

        # history of actions 
        proprio = self._build_proprio_from_obs(obs_dict)

        proprio = proprio.unsqueeze(0).unsqueeze(0) # (1,1,34)
        #proprio = proprio.to(device).unsqueeze(0)   # (1, 1, 34)
        states, state_elem_mask = self._isaac_proprio_to_unformat_state(proprio)    # (1, 1, 128), (1, 128)
        states, state_elem_mask = states.to(device, dtype=dtype), state_elem_mask.to(device, dtype=dtype)
        states = states[:, -1:, :]  # (1, 1, 128)
        ctrl_freqs = torch.tensor([self.control_frequency]).to(device)
        
        text_embeds = text_embeds.to(device, dtype=dtype)
        
        if text_embeds.ndim == 2:
            # (1,11,4096)
            text_embeds = text_embeds.unsqueeze(0)

        trajectory = self.policy.predict_action(
            lang_tokens=text_embeds,
            lang_attn_mask=torch.ones(
                text_embeds.shape[:2], dtype=torch.bool,
                device=text_embeds.device),
            img_tokens=image_embeds,
            state_tokens=states,
            action_mask=state_elem_mask.unsqueeze(1),  
            ctrl_freqs=ctrl_freqs
        )
        trajectory = self._unformat_action_to_isaac_action(trajectory).to(torch.float32) # (1,T,16)

        return trajectory
