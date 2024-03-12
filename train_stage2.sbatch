#!/bin/bash 
#SBATCH -p nvidia
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:a100:3
#SBATCH --time=95:00:00
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
# export TRANSFORMERS_CACHE="/scratch/ae2195/huggingface_cache"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HOME="/scratch/ae2195/huggingface_cache"

# Assume images.json is in the current working directory or provide the full path
#JSON_FILE_PATH="./stage2_train/images.json"


#Stage 2: Medical Visual Instruct Tuning
torchrun --nnodes=1 --nproc_per_node=3 --master_port=25001 \
    llava/train/train_mem.py \
    --model_name_or_path checkpoints/llava-med-7b-pretrain \
    --data_path data/instruct/llava_med_instruct_60k_inline_mention.json \
    --image_folder data/images \
    --vision_tower openai/clip-vit-large-patch14 \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end True \
    --bf16 True \
    --output_dir ./checkpoints/stage2 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to wandb
    #--input_json "${JSON_FILE_PATH}"  # Add the path to the JSON file as an argument


    
# torchrun --nnodes=1 --nproc_per_node=1 --master_port=25001 \
#     llava/train/train_mem.py \
#     --model_name_or_path checkpoints/llava-med-7b-pretrain \
#     --data_path data/instruct/llava_med_instruct_60k_inline_mention.json \
#     --image_folder data/images \
#     --vision_tower openai/clip-vit-large-patch14 \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end True \
#     --bf16 True \
#     --output_dir ./checkpoints/stage2 \
#     --num_train_epochs 3 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 8 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 5000 \
#     --save_total_limit 3 \
#     --learning_rate 2e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --fsdp "full_shard auto_wrap" \
#     --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --lazy_preprocess True \
#     --report_to wandb \