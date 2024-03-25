#!/bin/bash 

#SBATCH -p nvidia
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=2:59:59
#SBATCH --mem=150GB
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err
#SBATCH --job-name=training

# Environment setup
module purge

# Load Conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llava-med

# Set environment variables
export TRANSFORMERS_CACHE="/scratch/ae2195/huggingface_cache"

#python create_filtered_json.py

# torchrun --nnodes=1 --nproc_per_node=1 --master_port=25001 \
#     llava/train/train_mem.py \
#     --model_name_or_path /scratch/ae2195/LLaVA-Med/model \
#     --data_path /scratch/ae2195/LLaVA-Med/data/alignment/filtered_json_file.json \
#     --image_folder /scratch/ae2195/LLaVA-Med/data/images \
#     --vision_tower openai/clip-vit-large-patch14 \
#     --tune_mm_mlp_adapter True \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end \
#     --bf16 True \
#     --output_dir ./checkpoints/llava-med-7b-pretrain \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 2 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 16 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 2400 \
#     --save_total_limit 1 \
#     --learning_rate 2e-3 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --lazy_preprocess True \
#     --report_to none


# Training TinyLlava 
torchrun --nnodes=1 --nproc_per_node=1 --master_port=25001 \
    llava/train/train_mem.py \
    --model_name_or_path "bczhou/tiny-llava-v1-hf" \
    --data_path /scratch/ae2195/LLaVA-Med/data/alignment/filtered_json_file.json \
    --image_folder /scratch/ae2195/LLaVA-Med/data/images \
    --vision_tower openai/clip-vit-large-patch14 \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end \
    --bf16 True \
    --output_dir ./checkpoints/llava-med-7b-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2400 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to none
