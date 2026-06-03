import os
import json

import torch
import yaml
from tqdm import tqdm

from models.multimodal_encoder.t5_encoder import T5Embedder



GPU = 1
MODEL_PATH = "google/t5-v1_1-xxl"
CONFIG_PATH = "configs/base.yaml"
# rdt格式的数据集路径
TARGET_DIR = "/data/rdt_js"

# Note: if your GPU VRAM is less than 24GB, 
# it is recommended to enable offloading by specifying an offload directory.
OFFLOAD_DIR = None #"/data/t5_offload"  # Specify your offload directory here, ensuring the directory exists.

def as_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        return [x]
    return []

def main():
    with open(CONFIG_PATH, "r") as fp:
        config = yaml.safe_load(fp)
    
    device = torch.device(f"cuda:{GPU}")
    t5_model_kwargs = {
        "low_cpu_mem_usage": True,
        "torch_dtype": torch.float32,
        "device_map": {"shared": device, "encoder": device},
    }

    text_embedder = T5Embedder(
        from_pretrained=MODEL_PATH, 
        model_max_length=config["dataset"]["tokenizer_max_length"], 
        device=device,
        t5_model_kwargs=t5_model_kwargs,
        use_offload_folder=OFFLOAD_DIR
    )
    tokenizer, text_encoder = text_embedder.tokenizer, text_embedder.model
    
    # Get all the task paths
    task_paths = []
    for episode_dir in os.listdir(TARGET_DIR):
        episode_path = os.path.join(TARGET_DIR, episode_dir)
        if not os.path.isdir(episode_path):
            continue

        instr_path = os.path.join(episode_path, "expanded_instruction_gpt-4-turbo.json")
        if os.path.exists(instr_path):
            task_paths.append(episode_path)

    print(f"Found {len(task_paths)} episodes with instruction json.")

    # For each task, encode the instructions
    for task_path in tqdm(task_paths):
        # Load the instructions corresponding to the task from the directory
        with open(os.path.join(task_path, 'expanded_instruction_gpt-4-turbo.json'), 'r') as f_instr:
            instruction_dict = json.load(f_instr)

        


        instructions = (
            as_list(instruction_dict.get("instruction", ""))
            + as_list(instruction_dict.get("simplified_instruction", ""))
            + as_list(instruction_dict.get("expanded_instruction", ""))
        )
    
        # Encode the instructions  对语言指令进行编码
        tokenized_res = tokenizer(
            instructions, return_tensors="pt",
            padding="longest",
            truncation=True
        )
        tokens = tokenized_res["input_ids"].to(device)
        attn_mask = tokenized_res["attention_mask"].to(device)
        
        with torch.no_grad():
            text_embeds = text_encoder(
                input_ids=tokens,
                attention_mask=attn_mask
            )["last_hidden_state"].detach().cpu()
        
        attn_mask = attn_mask.cpu().bool()

        # Save the embeddings for training use
        for i in range(len(instructions)):
            text_embed = text_embeds[i][attn_mask[i]]
            save_path = os.path.join(task_path, f"lang_embed_{i}.pt")
            torch.save(text_embed, save_path)

if __name__ == "__main__":
    main()
