#!/bin/bash 

#SBATCH -p nvidia
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a100:2
#SBATCH --time=35:59:59
#SBATCH --mem=150GB
#SBATCH -C 80g
#SBATCH -o job.April27th_LLAVA_MED_SG2_with_TUNE_entire_model_ML_ADAPTER.%J.out
#SBATCH -e job.April27th_LLAVA_MED_SG2_with_TUNE_entire_model_ML_ADAPTER.%J.err


#SBATCH --job-name=trainSL

# Environment setup
module purge

# Load Conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llava-med

nvidia-smi 

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Set environment variables
export HF_HOME="/scratch/ltl2113/huggingface_cache"


torchrun --nnodes=1 --nproc_per_node=3 --master_port=25001 \
    llava/train/train_mem.py \
    --model_name_or_path /scratch/ltl2113/LLaVA-Med/checkpoints/llava-med-7b-pretrain-SG1_Final \
    --data_path data/instruct/filtered_json_file.json \
    --image_folder /scratch/ltl2113/LLaVA-Med/data/Allimages \
    --vision_tower openai/clip-vit-large-patch14 \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end True \
    --tune_mm_mlp_adapter True \
    --bf16 True \
    --output_dir /scratch/ltl2113/LLaVA-Med/checkpoints/LLAVA_MED_SG2_with_TUNE_Entire_model_April_27th \
    --num_train_epochs 15 \
    --per_device_train_batch_size 16 \
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
    --run_name April27th_LLAVA_MED_SG2_with_TUNE_entire_model_ML_ADAPTER
