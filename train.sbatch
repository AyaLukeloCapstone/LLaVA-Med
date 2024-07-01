#!/bin/bash 

#SBATCH -q ebrainccs 
#SBATCH -p nvidia
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:4
#SBATCH --time=35:59:59
#SBATCH --mem=150GB
#SBATCH -C 80g
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err

#SBATCH --job-name=training

# Environment setup
module purge

# Load Conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llava-med

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

nvidia-smi
export CUDA_LAUNCH_BLOCKING=1

# Set environment variables

export HF_HOME="/scratch/ltl2113/huggingface_cache"

# nvidia-smi
torchrun --nnodes=1 --nproc_per_node=3 --master_port=25001 \
    llava/train/train_mem.py \
    --model_name_or_path /scratch/ltl2113/LLaVA-Med/model \
    --data_path /scratch/ltl2113/LLaVA-Med/data/alignment/filtered_json_file.json \
    --image_folder /scratch/ltl2113/LLaVA-Med/data/Allimages \
    --vision_tower openai/clip-vit-large-patch14 \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end True \
    --tune_mm_mlp_adapter True \
    --bf16 True \
    --output_dir /scratch/ltl2113/LLaVA-Med/checkpoints/llava-med-7b-pretrain-SG1_Final \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
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