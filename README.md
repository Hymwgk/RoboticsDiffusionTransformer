# RDT-1B: a Diffusion Foundation Model for Bimanual Manipulation

### 📝[Paper](https://arxiv.org/pdf/2410.07864) | 🌍[Project Page](https://rdt-robotics.github.io/rdt-robotics/) | 🤗[Model](https://huggingface.co/robotics-diffusion-transformer/rdt-1b) | 🛢️[Data](https://huggingface.co/datasets/robotics-diffusion-transformer/rdt-ft-data) | 🏞️[Poster](./assets/iclr2025_poster.png)

![](./assets/head.png)

RDT-1B is a **1B**-parameter (*largest* to date) imitation learning **Diffusion Transformer** pre-trained on **1M+** (*largest* to date) multi-robot episodes. Given language instruction and RGB images of up to three views, RDT can predict the next $64$ robot actions. RDT is inherently compatible with **almost all kinds of modern mobile manipulators**, from single-arm to dual-arm, joint to EEF, position to velocity, and even with wheeled locomotion.

We have fine-tuned RDT on **6K+** (one of the *largest*) self-collected bimanual episodes and deployed it on the ALOHA **dual-arm** robot. It has achieved state-of-the-art performance in terms of dexterity, zero-shot generalizability, and few-shot learning. You can find Demo videos on our [project page](https://rdt-robotics.github.io/rdt-robotics/).

This repo is an official PyTorch implementation of RDT, containing:

- 🛠️Model [implementation](models/rdt_runner.py) of RDT
- 🤗1M-step [checkpoint](https://huggingface.co/robotics-diffusion-transformer/rdt-1b) of RDT-1B pre-trained on multi-robot data
- 🤗500K-step [checkpoint](https://huggingface.co/robotics-diffusion-transformer/rdt-170m) of RDT-170M (RDT(small) in [ablation](https://arxiv.org/pdf/2410.07864))
- 📈Training and sampling [scripts](train/train.py) (with DeepSpeed)
- 🤖An [example](scripts/agilex_inference.py) of real-robot deployment
- 🕹️Simulation benchmark from [Maniskill](https://github.com/haosulab/ManiSkill) environment

The following guides include the [installation](#installation), [fine-tuning](#fine-tuning-on-your-own-dataset), and [deployment](#deployment-on-real-robots). Please refer to [pre-training](docs/pretrain.md) for a detailed list of pre-training datasets and a pre-training guide.

## 📰 News
- [2025/04/04] [Poster](./assets/iclr2025_poster.png) is uploaded.
- [2024/12/17] 🔥 [Scripts](#simulation-benchmark) for evaluating RDT in Maniskill Simulation Benchmark is released!
- [2024/10/23] 🔥 **RDT-170M** (Smaller) model is released, a more VRAM-friendly solution 🚀💻.

## Installation

1. Clone this repo and install prerequisites:

    ```bash
    # Clone this repo
    git clone git@github.com:thu-ml/RoboticsDiffusionTransformer.git
    cd RoboticsDiffusionTransformer
    
    # Create a Conda environment
    conda create -n rdt python=3.10.0
    conda activate rdt
    
    # Install pytorch
    # Look up https://pytorch.org/get-started/previous-versions/ with your cuda version for a correct command
    pip install torch==2.1.0 torchvision==0.16.0  --index-url https://download.pytorch.org/whl/cu121
    
    # Install other prequisites
    pip install -r requirements.txt    
        
    # Install flash-attn  
    cd ~/
    wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.2.2/flash_attn-2.2.2+cu121torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

    pip install flash_attn-2.2.2+cu121torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl 
    ```

2. Download off-the-shelf multi-modal encoders:

   检查服务器的已有文件结构，确保存在且`/model/huggingface/hub`的文件结构为
   ```
      .
   ├── models--google--siglip-so400m-patch14-384
   │   ├── blobs
   │   ├── refs
   │   └── snapshots
   ├── models--robotics-diffusion-transformer--rdt-1b
   │   ├── blobs
   │   ├── refs
   │   └── snapshots
   ├── t5-v1_1-xxl
   │   ├── README.md
   │   ├── config.json
   │   ├── generation_config.json
   │   ├── pytorch_model.bin
   │   ├── special_tokens_map.json
   │   ├── spiece.model
   │   ├── tf_model.h5
   │   └── tokenizer_config.json
   ├── version.txt
   └── version_diffusers_cache.txt
   ```

   否则，在`/model`中创建huggingface缓存文件夹
   ```
   mkdir -p /model/huggingface/hub
   ```
   从此处下载预训练的编码器模型参数，将下载的编码器参数放在`/model/huggingface/hub`文件夹中

   - `t5-v1_1-xxl`: [link](https://huggingface.co/google/t5-v1_1-xxl/tree/main)🤗
   - `siglip`: [link](https://huggingface.co/google/siglip-so400m-patch14-384)🤗

   创建预训练编码器软链接
   ```
   # 在本项目根文件夹目录下
   cd /path_to/your_project_root 
   mkdir -p google
   # 创建软链接
   ln -s /model/huggingface/hub/t5-v1_1-xxl  google/t5-v1_1-xxl
   ln -s /model/huggingface/hub/siglip  google/siglip-so400m-patch14-384
   ```


3. Fill the missing argument in [this file](configs/base.yaml#L22):
   
   Note that this buffer will only be used during pre-training. See [this doc](docs/pretrain.md) for more details.
   ```
   # ...
   
   dataset:
   # ...
   # ADD YOUR buf_path: the path to the buffer (at least 400GB)
      buf_path: /path/to/buffer
   # ...
   ```

## RDT 数据集结构解释，以hdf5数据集为例


```
rdt_js/                                       # 数据集名称
├── episode_000/                              # 每个回合的单独文件夹
│   ├── data.hdf5                             # 每个回合的数据文件
│   └── expanded_instruction_gpt-4-turbo.json # 任务指令说明
├── episode_001/
│   ├── data.hdf5
│   └── expanded_instruction_gpt-4-turbo.json
```
### 语言指令文件`expanded_instruction_gpt-4-turbo.json`内部结构
```json
{
  "instruction": "...",              // 标准自然语言指令
  "simplified_instruction": "...",   // 简化版自然语言指令
  "expanded_instruction": "..."      // 详细版本
}
```
例如
```json
{
  "instruction": "Pick up the bandage and place it into the first aid kit.",
  "simplified_instruction": "Put the bandage into the kit.",
  "expanded_instruction": "Reach toward the bandage, grasp it securely, lift it from the table, move it over the first aid kit, and place it inside the kit."
}
```
这三个版本的指令，在训练时会被随机采用，各1/3的概率。

### data.hdf5 文件内部结构
```
/
├── observations               
│   ├── qpos                   # 128维度统一空间
│   └── images
│       ├── cam_high           # 顶部相机
│       ├── cam_left_wrist     # 左手腕相机
│       └── cam_right_wrist    # 右手腕相机
└── action                     # 128维度统一空间
```
其中数据格式为
```
observations/qpos: shape = (T, N)
action:            shape = (T, N)
```
长度 T >= 128



RDT数据集的末端位置姿态，的参考坐标系是 base 还是 world ?



## Isaaclab 数据集结构与准备

### 调整Isaaclab数据集
假设Isaaclab数据集文件夹为`/data/isaaclab_js`，确保其文件结构类似于
```
isaaclab_js/
├── Get-place-Bandage-merged.hdf5    # 每个任务使用一个hdf5文件表示
├── Mission-Abort-Estop.hdf5
├── Set-Mode-Off.hdf5
├── Sorting-Bullets.hdf5
└── Instructions.json
```
每个 .hdf5 是一个任务，hdf5内部对应多个 demo（每个回合对应一个demo）：
```
task.hdf5
└── data
    ├── demo_0                           # 回合0 
    │   ├── obs                          # 观测空间
    │   │   ├── eef_pos_left_b           # 左手末端执行器位置，相对于 机器人自身base坐标系 (T,3) dtype=float32
    │   │   ├── eef_pos_left_w           # 左手末端执行器位置，相对于 世界world坐标系 (T,3) dtype=float32
    │   │   ├── eef_pos_right_b          
    │   │   ├── eef_pos_right_w          
    │   │   ├── eef_quat_left_b          # 左手末端执行器姿态(四元数)，相对于 机器人自身base坐标系 (T,4) dtype=float32
    │   │   ├── eef_quat_left_w          # 左手末端执行器姿态(四元数)，相对于 世界world坐标系 (T,4) dtype=float32
    │   │   ├── eef_quat_right_b         
    │   │   ├── eef_quat_right_w         
    │   │   │
    │   │   ├── gripper_left_pos         # 左手夹爪的位置/开合度 (T,2) dtype=float32
    │   │   ├── gripper_right_pos        # 右手夹爪的位置/开合度 (T,2) dtype=float32
    │   │   │
    │   │   ├── joint_pos_left           # 左手绝对关节位置/角度 (T,9) dtype=float32
    │   │   ├── joint_pos_left_rel       # 左手相对关节位置 (T,9) dtype=float32
    │   │   ├── joint_pos_right          # 右手绝对关节位置/角度 (T,9) dtype=float32
    │   │   ├── joint_pos_right_rel      # 右手相对关节位置 (T,9) dtype=float32
    │   │   ├── joint_vel_left_rel       # 左手相对关节速度 (T,9) dtype=float32
    │   │   ├── joint_vel_right_rel      # 右手相对关节速度 (T,9) dtype=float32
    │   │   │
    │   │   ├── gsmini_left_left_tactile_rgb   # 左手左侧视触觉相机的原始 RGB 图像 (T,180,240,3) dtype=float32
    │   │   ├── gsmini_left_left_marker_motion # 左手左侧视触觉相机的 Marker 点特征运动向量 (T,2,99,2) dtype=float32
    │   │   ├── gsmini_right_left_tactile_rgb  # 右手左侧视触觉相机的原始 RGB 图像 (T,180,240,3) dtype=float32
    │   │   ├── gsmini_right_left_marker_motion# 右手左侧视触觉相机的 Marker 点特征运动向量 (T,2,99,2) dtype=float32
    │   │   │
    │   │   ├── zed_left                 # ZED双目相机左眼视角图像 (T,480,640,3) dtype=uint8
    │   │   ├── zed_right                # ZED双目相机右眼视角图像 (T,480,640,3) dtype=uint8
    │   │   ├── wrist_cam_left           # 左手腕部相机视角图像 (T,480,640,3) dtype=uint8
    │   │   ├── wrist_cam_right          # 右手腕部相机视角图像 (T,480,640,3) dtype=uint8
    │   │   └── table_cam                # 台面/全局视角固定相机图像 (T,480,640,3) dtype=uint8
    │   │
    │   └── actions                      # 动作空间：双臂控制目标输出 (T,16) dtype=float32 TODO：action似乎并没有用控制量，检查一下
    │                                    # r_ee_pos(3) + r_quat_wxyz(4) + r_gripper(1) + l_ee_pos(3) + l_quat_wxyz(4) + l_gripper(1)
    │                                    # 夹爪动作 r/l_gripper 范围[-1,1] 开1  闭-1
    ├── demo_1                           # 回合1 (结构与 demo_0 完全一致)
    └── demo_2                           # 回合2 ...
```
`Instruction.json`文件中存放对应的语言指令，为字典格式：
```json
{
    "Get-place-Bandage": {
        "instruction": "Pick up the bandage and place it into the first aid kit."
    },

    "Mission-Abort-Estop": {
        "instruction": "Press the emergency stop button to abort the mission."
    },

    "Set-Mode-Off": {
        "instruction": "Turn the mode switch to the OFF position."
    },

    "Sorting-Bullets": {
        "instruction": "Sort the bullets into the correct locations."
    }
}
```
其中，外层字典的key需要与任务的hdf5的名称相对应


### 转换至RDT转数据集格式
运行转换指令，将isaaclab数据集转换为  rdt 数据集
```bash
cd ./data
python isaaclab_to_rdt.py  --input-root /data/isaaclab_js  --output-root /data/rdt_js
```
检查rdt_js文件夹结构是否如
```bash
rdt_js/
├── episode_000/
│   ├── data.hdf5
│   └── expanded_instruction_gpt-4-turbo.json
├── episode_001/
│   ├── data.hdf5
│   └── expanded_instruction_gpt-4-turbo.json
```



### 对转换后的数据集进行 语言指令预编码


修改`RoboticsDiffusionTransformer/scripts/encode_lang_batch.py`中的`TARGET_DIR = "/data/rdt_js"`
```python
cd  RoboticsDiffusionTransformer/
python -m scripts.encode_lang_batch
```

转换后，将向TARGET_DIR数据集文件夹中，添加预编码的embedding文件，查看预编码后的结构为
```bash
rdt_js/
├── episode_000/
│   ├── data.hdf5
│   ├── expanded_instruction_gpt-4-turbo.json
│   ├── lang_embed_0.pt  # 标准指令预编码embedding  
│   ├── lang_embed_1.pt  # 简化指令预编码embedding 
│   └── lang_embed_2.pt  # 扩展指令预编码embedding 
│
├── episode_001/
│   ├── data.hdf5
│   ├── expanded_instruction_gpt-4-turbo.json
│   ├── lang_embed_0.pt
│   ├── lang_embed_1.pt
│   └── lang_embed_2.pt
```



### 使用自己的数据集进行微调

If your fine-tuning dataset is in the [Open X-Embodiment](https://robotics-transformer-x.github.io/) or the collection of our pre-training datasets (see [this doc](docs/pretrain.md#download-and-prepare-datasets)), you can also fine-tune RDT through the pre-trained pipeline. You need to remove other redundant datasets in the parameters. We refer to [this guide](docs/pretrain.md) (pre-training).

1. 准备自己的数据集，以hdf5格式的数据集为例:
   经过上述转换得到rdt结构的数据集`/data/rdt_js`
   
   创建软链接:
   ```bash
   # 在本项目根目录中设置文件夹
   cd data
   mkdir -p datasets
   # 创建软链接
   ln -s /data/rdt_js   datasets/rdt_js
   ```

2. 计算转换后的rdt数据集统计量
   ```bash
   # Under the root directory of this repo
   # Use -h to see the full usage
   python -m data.compute_dataset_stat_hdf5
   ```


3. 部署数据集加载器:


   i. 对数据集 `rdt_js`进行配置:

      把自己的数据集 `rdt_js` 的控制频率写进 [这个文件里](configs/dataset_control_freq.json). 把数据集名称 `rdt_js` 写到 [这个文件里](configs/finetune_datasets.json) 以及 [这个文件里](configs/finetune_sample_weights.json), 如果只有一个微调用的数据集，采样权重 sampling weight 的数值无所谓不用管. 这两个文件中都有一个占位符 `agilex`; 把他们改为自己的数据集名称`rdt_js`就行.

   ii. 重新部署 `HDF5VLADataset`类:

      在 [这个文件里](data/hdf5_vla_dataset.py)能够找到`HDF5VLADataset`这个类. 在该文件中，提供了论文中加载微调数据集的一个例子 (看[这个链接](https://huggingface.co/datasets/robotics-diffusion-transformer/rdt-ft-data)).

      要想将这个类，改动用于自己的数据集，需要做以下几点改动: (a) 修改 `HDF5_DIR` (自己数据集路径`rdt_js`) 以及数据集名称`DATASET_NAME` (`"rdt_js"`) in L21 and L22; (b) 自行实现两个函数 `parse_hdf5_file()` and `parse_hdf5_file_state_only()`. 仔细看源代码和注释。

      Note 1: 不是非要用HDF5文件来存储自己的数据集，只要保证数据集类是正常设置的就行了。

      Note 2: 在部署期间，需要将自己的机器人的动作设定为“统一动作空间”. 看[这个文件](configs/state_vec.py) (L180-194)有对统一动作空间的每个维度的具体含义解释.
      We have reserved enough slots for each physical quantity. For example, we have reserved ten slots for joint angles. If your robot arm has six degrees of freedom, you only need to fill in the first six. 

      **要点 1:** 如果是单臂机械臂，需要将动作填到“右臂”的部分，而不是“左臂”对应的地方。If your robot is single-arm, please fill its action into the *right-arm* portion of the unified action vector, aligning with our pre-training datasets.

      **要点 2:** 本项目使用的是 [6D representation](https://arxiv.org/pdf/1812.07035) 来表征末端执行器的旋转(EEF rotation). 
      如果自己的机器人动作包含 末端执行器(EEF) 的旋转量（角度或四元数），需要参考 [这个文件](docs/test_6drot.py)进行转换. 其实就是说：多个欧拉角可能对应同一个真实旋转姿态，例如 [0,0,0] 和 [360,0,0]；正负号四元数也可能对应同一个真实旋转姿态，例如 [0,0,0,1] 和 [0,0,0,-1]。这导致对于网络而言，同一个物理姿态可能对应多个数值差异巨大的标签，从而增加学习难度。为了解决这一问题，我们希望采用一种与真实旋转姿态（SO(3)）保持一一对应关系的连续表征方式，从而消除这种参数化带来的歧义性，使网络学习更加稳定。

      **要点 3:** 在预训练期间，训练脚本里没有对动作/物理量（除了夹具宽度）进行归一化。这样做保留了每个物理量的含义，促进了机器人之间的泛化。因此，建议不要标准化任何物理量，而是为它们选择合适的单位。通常，我们使用国际单位制，这可确保大多数值落在 [-1,1] 范围内。作为例外，本项目将夹具宽度执行最小-最大标准化为 [0,1]。

      **要点 4:** 4090 GPU的显存可能无法加载 `t5-v1_1-xxl` 编码器. 建议先去单独计算语言指令的编码(看 [这个文件](scripts/encode_lang_batch.py)有例子) 然后在微调时候加载语言编码. 这样做的话就得在 `HDF5VLADataset` (see L148) 中加载刚预编码好的语言指令的embedding，而不是输入自然语言。




4. 开始微调:
   模型架构和数据处理相关的配置位于[此文件](configs/base.yaml)中。通常情况下，无需修改​​这些配置；否则，加载预训练检查点时会出错。训练相关的配置通过*命令行参数*传递。使用`python main.py -h`查看配置说明。我们在[此文件](finetune.sh)中提供了一个微调脚本示例(`finetune.sh`)。可能需要修改此文件中的一些参数，例如`CUTLASS_PATH`和`WANDB_PROJECT`。


   使用该指令开始微调:

   ```bash
   # 这是多卡
   source finetune.sh
   # 这是单卡
   source finetune_maniskill.sh
   ```

  ```bash
   accelerate launch --num_processes=1  main.py \    # 单卡
    --deepspeed="./configs/zero2.json" \
    --pretrained_model_name_or_path="robotics-diffusion-transformer/rdt-1b" \
    --pretrained_text_encoder_name_or_path=$TEXT_ENCODER_NAME \
    --pretrained_vision_encoder_name_or_path=$VISION_ENCODER_NAME \
    --precomp_lang_embed \
    --output_dir=$OUTPUT_DIR \
    --train_batch_size=1 \       # 批次大小
    --sample_batch_size=1 \      # 训练时进行验证的样本采样大小 
    --gradient_accumulation_steps=24 \  # 对梯度进行累积，等效为train_batch_size*gradient_accumulation_steps的batch size
    --max_train_steps=400000 \   # 训练步数，优先级高于 num_train_epochs
    --checkpointing_period=10000 \
    --sample_period=500 \
    --checkpoints_total_limit=40 \
    --lr_scheduler="constant" \
    --learning_rate=1e-4 \
    --mixed_precision="bf16" \
    --dataloader_num_workers=4 \
    --image_aug \
    --dataset_type="finetune" \
    --state_noise_snr=40 \
    --load_from_hdf5 \
    --report_to=wandb
   ```



   with `finetune.sh` detailed as below:

   ```bash
      deepspeed --hostfile=hostfile.txt main.py \
         --deepspeed="./configs/zero2.json" \   # If you want to use DeepSpeed, which is strongly recommended
         --pretrained_model_name_or_path=<MODEL ID | DIRECTORY OF MODEL WEIGHTS | PATH TO MODEL CHECKPOINT> \
         --pretrained_text_encoder_name_or_path=<MODEL ID | PATH TO MODEL DIRECTORY > \   # e.g., google/t5-v1_1-xxl
         --pretrained_vision_encoder_name_or_path=<MODEL ID | PATH TO MODEL DIRECTORY> \  # e.g., google/siglip-so400m-patch14-384
         --output_dir=<DIRECTORY to SAVE CHECKPOINTS> \ # e.g., checkpoints/rdt-1b-agilex
         --train_batch_size=32 \
         --sample_batch_size=64 \   # batch size for diffusion sampling in validation 
         --max_train_steps=200000 \
         --checkpointing_period=1000 \
         --sample_period=500 \   # sample period for validation
         --checkpoints_total_limit=40 \
         --lr_scheduler="constant" \
         --learning_rate=1e-4 \
         --mixed_precision="bf16" \ # If you want to use mixed precision, bf16 is recommended
         --dataloader_num_workers=8 \
         --image_aug \  # If you want to use image augmentation
         --dataset_type="finetune" \
         --state_noise_snr=40 \  # If you want to add noise to the state
         --load_from_hdf5 \   # If you use HDF5 to store your data
         --report_to=wandb
   ```

   **IMPORTANT**: 如果已经选择使用预先编码的语言embedding来当作语言指令，那么就要在`finetune.sh`中对`--precomp_lang_embed` 进行指定.

   Note 1:如何导入预训练的RDT参数， `pretrained_model_name_or_path` can one of:

      - a string, the *model id* of a pre-trained model hosted inside a model repo on HuggingFace. Please fill with `"robotics-diffusion-transformer/rdt-1b"`, which is the officially-released [RDT-1B model](https://huggingface.co/robotics-diffusion-transformer/rdt-1b)🤗 at HuggingFace. (recommended)
      - a string, the path to a *directory* containing the manually downloaded model weights from HuggingFace, e.g., `"/path/to/rdt-1b"`. You should first manually download the `rdt-1b` directory from this [link](https://huggingface.co/robotics-diffusion-transformer/rdt-1b)🤗.
      - a string, the path to a *directory* containing model weights saved using [`~RDTRunner.save_pretrained`] method. This can be either:
        -  `"checkpoints/rdt-pretrain-1b/checkpoint-<STEP NUMBER>"`: This is the path to the checkpoint saved in the `<STEP NUMBE>` iteration during pre-training. Refer to [this file](docs/pretrain.md) for a tutorial on how to start your own pre-training.
        - `"checkpoints/rdt-pretrain-1b"`: If the pre-training completes normally without any exception, you can specify this path to load the last checkpoint.
      - a string, the path to model checkpoint (`*.pt`) saved by DeepSpeed, e.g., `"checkpoints/rdt-pretrain-1b/checkpoint-<STEP NUMBER>/pytorch_model/mp_rank_00_model_states.pt"` (verified)
      - `None` if you want to randomly initialize the model using configuration at `config_path`.

   Note 2: You can monitor the training process by observing `loss` (through a long window moving average) and `overall_avg_sample_mse` in [Wandb](https://wandb.ai/site) or [TensorBoard](https://www.tensorflow.org/tensorboard). We empirically found that the lower the `overall_avg_sample_mse`, the better the model performs. Usually, fine-tuning is over when this value converges.

   Note 3: 如果训练出现波动，可以通过添加更多 GPU 或设置更大的 `--gradient_accumulation_steps` 来增加批次大小。

   Note 4: 在使用hdf5格式数据集进行微调时，需要指定 `--load_from_hdf5` 参数.

## Deployment on Real-Robots

We have encapsulated the inference of the model into a class named `RoboticDiffusionTransformerModel` (see [this file](scripts/agilex_model.py#L38)). You can call this class's `step()` method for inference. However, you may need to re-implement some parts according to your specific robot. You should at least modify the `_format_joint_to_state()` (L164) and `_unformat_action_to_joint()` (L196) to convert between robot raw actions and unified action vectors that RDT accepts. You may also specify the control frequency of your robot (L49).

**IMPORTANT**: When you feed the images into `step()`, remember the order MUST be `[ext_{t-1}, right_wrist_{t-1}, left_wrist_{t-1}, ext_{t}, right_wrist_{t}, left_wrist_{t}]`.

We provide an example hardware code in [this file](scripts/agilex_inference.py) for deployment on Mobile ALOHA, and the corresponding running script in [this file](inference.sh) (`inference.sh`), which is detailed below;

   ```bash
      python -m scripts.agilex_inference \
         --use_actions_interpolation \
         --pretrained_model_name_or_path=<PATH TO MODEL CHECKPOINT> \  # your finetuned checkpoint: e.g., checkpoints/rdt-finetune-1b/checkpoint-<STEP NUMBER>, checkpoints/rdt-finetune-1b/checkpoint-<STEP NUMBER>/pytorch_model/mp_rank_00_model_states.pt, the same before
         --lang_embeddings_path=<PATH TO YOUR INSTURCTION EMBEDDINGS> \ # e.g. outs/lang_embeddings/your_instr.pt"
         --ctrl_freq=25    # your control frequency
   ```

**IMPORTANT**: If you on-board GPU memory is not enough to encode the language, please refer to [this file](scripts/encode_lang.py) for precomputation and specify the language embedding path in `inference.sh`. Detail instructions are provided below:

   1. Set Required Parameters in `scripts/encode_lang.py`

      ```python
      # ...

      GPU = 0
      MODEL_PATH = "google/t5-v1_1-xxl"
      CONFIG_PATH = "configs/base.yaml"
      SAVE_DIR = "outs/"   # output directory

      # Modify this to your task name and instruction
      TASK_NAME = "handover_pan"
      INSTRUCTION = "Pick up the black marker on the right and put it into the packaging box on the left."

      # Note: if your GPU VRAM is less than 24GB, 
      # it is recommended to enable offloading by specifying an offload directory. 
      OFFLOAD_DIR = None  # Specify your offload directory here, ensuring the directory exists.

      # ...
      ```

   2. Run the script
      ```
      python -m scripts.encode_lang
      ```

Note: If you want to deploy on the Mobile ALOHA robot, don't forget to install the hardware prerequisites (see [this repo](https://github.com/MarkFzp/mobile-aloha)).

## Simulation Benchmark

We comprehensively evaluate RDT against baseline methods using the ManiSkill simulation benchmark. Specifically, we focus on five benchmark tasks: `PegInsertionSide`, `PickCube`, `StackCube`, `PlugCharger`, and `PushCube`. Here's a brief overview of the evaluation setup:

**Evaluation Setup:**

1. **Install ManiSkill:**  
   Within the [RDT environment](#installation), install ManiSkill as follows:
   ```bash
   conda activate rdt
   pip install --upgrade mani_skill
   ```

2. **Configure Vulkan:**  
   Follow the [ManiSkill documentation](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html#vulkan) to properly set up Vulkan。

3. **Obtain Model Weights:**  
   Download the fine-tuned model weights from [this Hugging Face repository](https://huggingface.co/robotics-diffusion-transformer/maniskill-model/tree/main/rdt). Download the precomputed language embeddings from [here](https://huggingface.co/robotics-diffusion-transformer/maniskill-model/tree/main/lang_embeds) to the root directory of this repo.
   
4. **Run Evaluation Scripts:**  
   After completing the setup steps, execute the provided evaluation scripts to assess RDT on the selected tasks.

```
conda activate rdt 
python -m eval_sim.eval_rdt_maniskill \
--pretrained_path PATH_TO_PRETRAINED_MODEL
```

### Implementation Details

#### Data

Utilizing the [official ManiSkill repository](https://github.com/haosulab/ManiSkill), we generated 5,000 trajectories through motion planning. The initial action mode of these trajectories is absolute joint position control and we subsequently converted them into delta end-effector pose control to align with the pre-training action space of OpenVLA and Octo. We strictly adhered to the official codebases of OpenVLA and Octo, modifying only the dataset-loading scripts. Consequently, we finetuned OpenVLA and Octo using the delta end-effector pose data. For RDT and Diffusion-Policy we leverage joint position control data for training which is aligned with our pre-training stage as well.

####  Training
- OpenVLA is fine-tuned from the officially released pre-trained checkpoint with LoRA-rank 32 until converge.
- Octo is fine-tuned from the officially released pre-trained checkpoint for 1M iterations until converge. 
- Diffusion-Policy is trained from scratch for 1000 epochs. We select the checkpoint of 700 epoch which has the lowest validation sample loss of 1e-3.
- RDT is fine-tuned from our released pre-trained checkpoint for 300ks iterations.

#### Results

Each method is evaluated over 250 trials (10 random seeds with 25 trials per seed). The quantitative results, including success rate mean and std value across 10 random seeds are presented below:


||PegInsertionSide|PickCube|StackCube|PlugCharger|PushCube|Mean|
|---|---|---|---|---|---|---|
|RDT|**13.2±0.29%**|**77.2±0.48%**|74.0±0.30%|**1.2±0.07%**|**100±0.00%**|**53.6±0.52%**|
|OpenVLA|0.0±0.00%|8±0.00%|8±0.00%|0.0±0.00%|8±0.00%|4.8±0.00%|
|Octo|0.0±0.00%|0.0±0.00%|0.0±0.00%|0.0±0.00%|0.0±0.00%|0.0±0.00%|
|Diffusion-Policy|0.0±0.00%|40.0±0.00%|**80.0±0.00%**|0.0%±0.00%|88.0±0.00%|30.2±0.00%|

#### Finetune RDT with Maniskill Data

To fine-tune RDT with Maniskill data, first download the Maniskill data from [here](https://huggingface.co/robotics-diffusion-transformer/maniskill-model) and extract it to `data/datasets/rdt-ft-data`. Then copy the code in `data/hdf5_vla_dataset.py` to `data/hdf5_maniskill_dataset.py` and run the following script:

```
bash finetune_maniskill.sh
```

#### Reproducing Baseline Results

Download and extract the fine-tuned model weights from [here](https://huggingface.co/robotics-diffusion-transformer/maniskill-model) to `eval_sim/`.

- OpenVLA: Clone [OpenVLA repo](https://github.com/openvla/openvla) in `./eval_sim/` and install its environment & ManiSkill. Then run the following script:
```
python -m eval_sim.eval_openvla --pretrained_path PATH_TO_PRETRAINED_MODEL
```
- Octo: Clone [Octo repo](https://github.com/octo-models/octo.git) in `./eval_sim/` and install its environment & ManiSkill. The run the following script:
```
python -m eval_sim.eval_octo --pretrained_path PATH_TO_PRETRAINED_MODEL
```
- Diffusion-Policy: Clone our simplified [Diffusion-Policy repo](https://github.com/LBG21/RDT-Eval-Diffusion-Policy) in `./eval_sim/` and run:
```
python -m eval_sim.eval_dp --pretrained_path PATH_TO_PRETRAINED_MODEL
```

### RDT on [RoboTwin Benchmark](https://robotwin-platform.github.io/) 

RoboTwin is a benchmark simulator based on Sapien, comprising 50 common dual-arm tasks. All policies are trained on 50 trajectories collected in a clean environment for each of the 50 tasks, and then deployed and tested for success rates in environments corresponding to either easy (clean) or hard (randomized) configurations for the same 50 tasks. You can refer to ​​[https://robotwin-platform.github.io/doc/usage/RDT.html#1-environment-setup](https://robotwin-platform.github.io/doc/usage/RDT.html#1-environment-setup)​​ to set up the environment required for RoboTwin and RDT. According to the official test results of RoboTwin ([RoboTwin 2.0 Leaderboard](https://robotwin-platform.github.io/leaderboard)), RDT ranks second only to Pi0 (excluding the DP3 policy, which utilizes ground truth point clouds).
<img width="2048" height="1224" alt="image" src="https://github.com/user-attachments/assets/e721ffab-3dde-42f0-b36e-1593aa964a99" />


## FAQ

### 1. How can I fine-tune RDTs with limited VRAM?

- **Use a Smaller Model**: Opt for the [RDT-170M model](https://huggingface.co/robotics-diffusion-transformer/rdt-170m), which requires less VRAM.
  
- **Select a Memory-Efficient ZeRO Stage**: Choose a more memory-efficient ZeRO stage based on your needs:
  - **ZeRO-3 with Offload** > **ZeRO-3** > **ZeRO-2 with Offload** > **ZeRO-2** > **ZeRO-1**
  - By default, we use [ZeRO-2](https://github.com/thu-ml/RoboticsDiffusionTransformer/blob/c68398ed526733faca4eec52cc1a7d15a9f8fea7/finetune.sh#L29) for a balance between speed and memory efficiency. Find more details on ZeRO stages [here](https://huggingface.co/docs/transformers/main/deepspeed#select-a-zero-stage) and [here](https://www.deepspeed.ai/docs/config-json/#zero-optimizations-for-fp16-training).

- **Enable 8-bit Adam Optimization**: Activate 8-bit Adam by setting [`use_8bit_adam=True`](https://github.com/thu-ml/RoboticsDiffusionTransformer/blob/c68398ed526733faca4eec52cc1a7d15a9f8fea7/main.py#L195) for reduced memory usage during training.

- **Apply 4-bit or 8-bit Quantization**: Quantizing model weights can significantly reduce VRAM requirements.

- **Use [XFormers](https://github.com/facebookresearch/xformers)**: This library provides optimized transformers with efficient memory usage.

- **Enable Gradient Checkpointing**: Implement `gradient_checkpointing` manually to save memory during backpropagation. See [here](https://deepspeed.readthedocs.io/en/latest/activation-checkpointing.html) for instructions. Once you have successfully implemented this feature, we welcome you to submit a PR👏.
- **Gradient Accumulation**: Set a larger `--gradient_accumulation_steps=<num_steps>`. This will accumulate the gradients of `<num_steps>` batches for backpropagation. Equivalently, this will increase the batch size by `<num_steps>` times, at the cost of `<num_steps>` times the running time.

### 2. How many steps are recommended for fine-tuning RDT?

Regardless of the batch size you select, it is recommended to train for at least 150K steps to achieve optimal results.

### 3. What to do if t5-xxL is too large to store in GPU memory?

1. Do not load T5-XXL in your GPU memory when training. Pre-compute language embeddings in advance.
2. Set `OFFLOAD_DIR` to enable CPU offloading in `scripts/encode_lang_batch.py` and `scripts/encode_lang.py`.
3. Use smaller versions of t5 like t5-base instead of t5-xxL.

## Citation

If you find our work helpful, please cite us:

```bibtex
@article{liu2024rdt,
  title={RDT-1B: a Diffusion Foundation Model for Bimanual Manipulation},
  author={Liu, Songming and Wu, Lingxuan and Li, Bangguo and Tan, Hengkai and Chen, Huayu and Wang, Zhengyi and Xu, Ke and Su, Hang and Zhu, Jun},
  journal={arXiv preprint arXiv:2410.07864},
  year={2024}
}
```

Thank you!

## License

All the code, model weights, and data are licensed under [MIT license](./LICENSE).
