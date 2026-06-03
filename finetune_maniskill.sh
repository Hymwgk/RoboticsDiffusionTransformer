# 适合单机训练
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_7:1,mlx5_8:1,mlx5_9:1
export NCCL_IB_DISABLE=1
# 设置为本地回环接口，避免NCCL尝试使用其他网络接口进行通信，这在单机训练时是合适的
export NCCL_SOCKET_IFNAME=lo   
export NCCL_DEBUG=INFO
export NCCL_NVLS_ENABLE=0
# 添加公共huggingface缓存路径环境变量
export HF_HOME=/model/huggingface
export HUGGINGFACE_HUB_CACHE=/model/huggingface/hub
# 设定文本编码器和视觉编码器的名称，使用公共文件夹下的预训练模块
# 保持默认，前提是按照教程设置了软链接（见Readme），否则需要修改为实际路径
export TEXT_ENCODER_NAME="google/t5-v1_1-xxl"
export VISION_ENCODER_NAME="google/siglip-so400m-patch14-384"
# 设定为自己账户 wgk 文件夹下
export OUTPUT_DIR="/model/wgk/checkpoints/rdt-finetune-1b-sim"
# 默认不管
export CFLAGS="-I/usr/include"
export LDFLAGS="-L/usr/lib/x86_64-linux-gnu"
export CUTLASS_PATH="./data/cutlass"
# 设置wandb的项目名称和离线模式，确保训练日志能够正确记录到指定项目中，并且避免wandb尝试连接服务器
export WANDB_PROJECT="robotic_diffusion_transformer"
export WANDB_MODE=offline


if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir "$OUTPUT_DIR"
    echo "Folder '$OUTPUT_DIR' created"
else
    echo "Folder '$OUTPUT_DIR' already exists"
fi
# For run in a single node/machine
# accelerate launch main.py \
#     --deepspeed="./configs/zero2.json" \
#     ...

accelerate launch --num_processes=1  main.py \
    --deepspeed="./configs/zero2.json" \
    --pretrained_model_name_or_path="robotics-diffusion-transformer/rdt-1b" \
    --pretrained_text_encoder_name_or_path=$TEXT_ENCODER_NAME \
    --pretrained_vision_encoder_name_or_path=$VISION_ENCODER_NAME \
    --precomp_lang_embed \
    --output_dir=$OUTPUT_DIR \
    --train_batch_size=1 \
    --sample_batch_size=1 \
    --gradient_accumulation_steps=24 \
    --max_train_steps=400000 \
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

