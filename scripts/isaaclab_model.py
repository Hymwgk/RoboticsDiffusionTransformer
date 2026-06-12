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

from data.isaaclab_to_rdt import build_proprio_from_obs,fill_in_proprio

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
 


    def _build_proprio_from_obs(self, obs_dict: dict) -> torch.Tensor:
        """
        输入:
            obs_dict:  # torch
                joint_pos_right      [8]
                joint_pos_left       [8]
                eef_pos_right_b      [3]
                eef_pos_left_b       [3]
                eef_quat_right_b     [4]
                eef_quat_left_b      [4]
                gripper_right_pos    [2]
                gripper_left_pos     [2]

        输出: # numpy
            proprio [1, 34]
        """

        # 1.筛选出 ISAACLAB_PROPRIO_KEYS 中想要的 proprio 2.扩展维度 3.换为nparray 
        proprio = {k:v[None].cpu().numpy() for k,v in obs_dict.items() if k in ISAACLAB_PROPRIO_KEYS}
        # 转换为 34维度本体观测  [1,34]
        proprio = build_proprio_from_obs(proprio)
        # [1,34]
        return proprio

    def load_pretrained_weights(self, pretrained=None):
        if pretrained is None:
            return 
        print(f'Loading weights from {pretrained}')
        filename = os.path.basename(pretrained)
        if filename.endswith('.pt') or filename.endswith('.bin'):
            checkpoint =  torch.load(pretrained)
            if isinstance(checkpoint, dict) and "module" in checkpoint:
                self.policy.load_state_dict(checkpoint["module"])
            else:
                self.policy.load_state_dict(checkpoint)

        elif filename.endswith('.safetensors'):
            from safetensors.torch import load_model
            load_model(self.policy, pretrained)
        else:
            raise NotImplementedError(f"Unknown checkpoint format: {pretrained}")


    def _fill_in_proprio(self,proprio):
        """proprio [1:34] numpy array
        
        """
        # 创建 统一空间向量 
        uni_proprio = fill_in_proprio(proprio)
        uni_proprio_indicator = np.zeros(uni_proprio.shape[:-1] + (self.args["common"]["state_dim"],), dtype=np.float32)
        uni_proprio_indicator[..., ISAACLAB_PROPRIO_INDICES] = 1.0
        return uni_proprio, uni_proprio_indicator


    def _uni_vec_to_action(self, uni_vec):
        # 将统一动作空间  转换  为isaaclab动作空间  [B,chunk,N]
        rot6d_action = uni_vec[:, :, ISAACLAB_ACTION_INDICES]
        # 将动作转换为isaaclab动作，主要是6d动作转换
        right_pos = rot6d_action[:, :, ISAACLAB_ROT6D_ACTION_SLICE['right_pos']]
        right_rot6d = rot6d_action[:,:,ISAACLAB_ROT6D_ACTION_SLICE['right_rot6d']]
        right_gripper = rot6d_action[:, :, ISAACLAB_ROT6D_ACTION_SLICE['right_gripper']]

        left_pos = rot6d_action[:, :, ISAACLAB_ROT6D_ACTION_SLICE['left_pos']]
        left_rot6d = rot6d_action[:,:,ISAACLAB_ROT6D_ACTION_SLICE['left_rot6d']]
        left_gripper = rot6d_action[:, :, ISAACLAB_ROT6D_ACTION_SLICE['left_gripper']]
        # 左右手的动作处理
        right_wxyz_quat = self._rot6d_to_wxyz_quat(right_rot6d)
        left_wxyz_quat = self._rot6d_to_wxyz_quat(left_rot6d)
        # TODO gripper 的动作要转换么？ 检查一下
        # 拼接起来
        action = torch.cat(
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
        assert action.shape[-1] == 16, action.shape
        # denormalize to action space 不进行缩放 TODO 检查一下在微调的时候是否使用的就是归一化动作？
        return action

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

        # [1,34] nparray
        proprio = self._build_proprio_from_obs(obs_dict)
        # [1,128]     [1,128]
        uni_proprio, uni_proprio_indicator = self._fill_in_proprio(proprio)
        # [1,1,128]  
        uni_proprio = torch.as_tensor(uni_proprio,device=device,dtype=dtype,)[None]
        # [1,1,128]
        uni_proprio_indicator = torch.as_tensor(uni_proprio_indicator,device=device,dtype=dtype,)[None]  
        
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
            state_tokens=uni_proprio,
            action_mask=uni_proprio_indicator,  
            ctrl_freqs=ctrl_freqs
        )
        # [1,T,128] -> [1,T,16]
        trajectory = self._uni_vec_to_action(trajectory).to(torch.float32) 

        return trajectory
