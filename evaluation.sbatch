#!/bin/bash 

#SBATCH -p nvidia
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:a100:3
#SBATCH --time=3:59:59
#SBATCH --mem=150GB
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err

#SBATCH --job-name=Eval

# Environment setup
module purge

# Load Conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llavamed

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Set environment variables
export HF_HOME="/scratch/ltl2113/huggingface_cache"

answers_file="/scratch/ltl2113/LLaVA-Med/data/eval/eval_vqa_rad_finetuned_9_epochs_02.jsonl"

## VQA-RD EVALUATION 
#EVAL A STAGE
python llava/eval/run_med_datasets_eval_batch.py --model-name checkpoints/llava-med-7b-pretrain-finetuned-vqa-rad-9-epochs-02 \
    --question-file /scratch/ltl2113/vqa-rad/VQA-RAD/test.json \
    --image-folder /scratch/ltl2113/vqa-rad/VQA-RAD/test \
    --answers-file $answers_file
    
# #EVAL B stage
python llava/eval/run_eval_modified.py \
    --gt /scratch/ltl2113/vqa-rad/VQA-RAD/test.json \
    --pred $answers_file