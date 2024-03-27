#!/bin/bash 
 
#SBATCH -p nvidia
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=3:59:59
#SBATCH --mem=150GB
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err

#SBATCH --job-name=Eval

# Environment setup
module purge

# Load Conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tinyllava

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Set environment variables
export HF_HOME="/scratch/ltl2113/huggingface_cache"

## NORMAL EVALUATION PIPELINE
# python3 tinyllava/eval/model_vqa.py \
#     --model-path /scratch/ltl2113/LLaVA-Med/TinyLLaVABench_1/checkpoints/tiny-llava-base-pretrain/checkpoint-72000 \
#     --question-file /scratch/ltl2113/LLaVA-Med/data/eval/llava_med_eval_qa50_qa.jsonl \
#     --image-folder /scratch/ltl2113/LLaVA-Med/data/qa50_images \
#     --answers-file /scratch/ltl2113/LLaVA-Med/data/eval/answers_tiny_eval_qa50_qa_finetuned.jsonl

# python3 llava/eval/eval_multimodal_chat_gpt_score.py \
#     --question_input_path data/eval/llava_med_eval_qa50_qa.jsonl \
#     --input_path /scratch/ltl2113/LLaVA-Med/data/eval/answers_eval_qa50_qa.jsonl \
#     --output_path /scratch/ltl2113/LLaVA-Med/data/eval/gpt4-eval-for-individual-answers.jsonl

# # python3 llava/eval/summarize_gpt_review.py


## VQA-RD EVALUATION 
#EVAL A STAGE: CODE IS RUNNING BUT OUTPUT IS EMPTY
# python tinyllava/eval/run_med_datasets_eval_batch.py \
#     --model-name /scratch/ltl2113/LLaVA-Med/TinyLLaVABench_1/checkpoints/tiny-llava-base-pretrain/checkpoint-72000 \
#     --question-file /scratch/ltl2113/LLaVA-Med/data/Vqa/VQA-RAD/test.json \
#     --image-folder /scratch/ltl2113/LLaVA-Med/data/Vqa/VQA-RAD/test \
#     --answers-file /scratch/ltl2113/LLaVA-Med/data/eval/Tiny_eval_vqa_rd_epoch_stage_2.jsonl


python tinyllava/eval/run_med_datasets_eval_batch.py \
    --model-name bczhou/TinyLLaVA-1.5B \
    --question-file /scratch/ltl2113/LLaVA-Med/data/eval/updated_llava_med_eval_qa50_qa.json \
    --image-folder /scratch/ltl2113/LLaVA-Med/data/qa50_images \
    --answers-file /scratch/ltl2113/LLaVA-Med/data/eval/Tiny_eval_llava_med_stage_2.jsonl

# #EVAL B stage
# python llava/eval/run_eval_modified.py \
#     --gt /scratch/ltl2113/vqa-rad/data/test.json \
#     --pred /scratch/ltl2113/LLaVA-Med/data/eval/eval_vqa_rd_epoch_9.jsonl


# python tinyllava.py