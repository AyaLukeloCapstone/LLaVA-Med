#!/bin/bash

#SBATCH -p nvidia
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=3:59:59
#SBATCH --mem=100GB
#SBATCH --job-name=run_infer
#SBATCH -o job.%J.out
#SBTACH -e job.%J.err

# Env setup 
module purge

# Load the Conda module
source ~/.bashrc

 # Activate your Conda environment
conda activate llava-med

# Set the transformers cache path
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# set environment variables 
export HF_HOME = "/scratch/ae2195/huggingface_cache"

#cd /scratch/ae2195/LLaVA-Med/llava/eval

# Generate LLaVA-Med responses
python llava/eval/model_vqa.py \
    --model-name /scratch/ae2195/LLaVA-Med/model \
    --question-file data/eval/llava_med_eval_qa50_qa.jsonl \
    --image-folder data/images/ \
    --answers-file answers/answer-file.jsonl   


# #Evaluate the generated responses.
# python llava/eval/eval_multimodal_chat_gpt_score.py \
#     --question_input_path data/eval/llava_med_eval_qa50_qa.jsonl \
#     --input_path /path/to/answer-file.jsonl \
#     --output_path /path/to/save/gpt4-eval-for-individual-answers.jsonl


# # Summarize the evaluation results
# python summarize_gpt_review.py
