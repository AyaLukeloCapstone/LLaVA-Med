#!/bin/bash 

#SBATCH -p nvidia
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:a100:3
#SBATCH --time=95:59:59
#SBATCH --mem=150GB
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err

#SBATCH --job-name=training

# Environment setup
module purge

# Load Conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llavamed

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK


# Set environment variables

export HF_HOME="/scratch/ltl2113/huggingface_cache"


torchrun --nnodes=1 --nproc_per_node=3 --master_port=25001 \
    llava/train/train_mem.py \
    --model_name_or_path /scratch/ltl2113/LLaVA-Med/checkpoints/llava-med-7b-pretrain-stage-2-02 \
    --data_path /scratch/ltl2113/vqa-rad/VQA-RAD/train.json \
    --image_folder /scratch/ltl2113/vqa-rad/VQA-RAD/train \
    --vision_tower openai/clip-vit-large-patch14 \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end True \
    --bf16 True \
    --output_dir checkpoints/llava-med-7b-finetuned-vqa-rad-3-epochs-02 \
    --num_train_epochs 9 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
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