#!/bin/bash 


#SBATCH -p nvidia
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=10:59:59
#SBATCH --mem=150GB
#SBATCH -o job.llava-med-quantized-4bits-VQA_RAD_Eval.%J.out
#SBATCH -e job.llava-med-quantized-4bits-VQA_RAD_Eval.%J.err

#SBATCH --job-name=EvalNORM

# Environment setup
module purge

# Load Conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llava-med

pip show transformers 
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Set environment variables
export HF_HOME="/scratch/ltl2113/huggingface_cache"

# answers_file="/scratch/ltl2113/LLaVA-Med/data/eval/Quantized_AWQ_LLavaMED_eval_pretrained_SLAKE_epoch15.jsonl"

# ## VQA-RD EVALUATION 
# #EVAL A STAGE
# python llava/eval/run_med_datasets_eval_batch.py \
#     --model-name /scratch/ltl2113/LLaVA-Med/llava-med-7b-pretrain-SLAKE_15E_Final-AWQ \
#     --question-file /scratch/ltl2113/LLaVA-Med/data/Vqa/Slake/test.json \
#     --image-folder /scratch/ltl2113/LLaVA-Med/data/Vqa/Slake/imgs \
#     --answers-file $answers_file
    
# # # #EVAL B stage
# python llava/eval/run_eval_modified.py \
#     --gt /scratch/ltl2113/LLaVA-Med/data/Vqa/Slake/test.json \
#     --pred $answers_file


# ## VQA-RD EVALUATION 

# answers_file="/scratch/ltl2113/LLaVA-Med/data/eval/quantized_LLavaMED_eval_pretrained_VQA_epoch15.jsonl"
# # # #EVAL A STAGE
# python llava/eval/run_med_datasets_eval_batch.py \
#     --model-name /scratch/ltl2113/LLaVA-Med/checkpoints/llava-med-7b-pretrain-SLAKE_15E_Final_quantized \
#     --question-file /scratch/ltl2113/LLaVA-Med/data/Vqa/VQA-RAD/test.json \
#     --image-folder /scratch/ltl2113/LLaVA-Med/data/Vqa/VQA-RAD/test \
#     --answers-file $answers_file
    
# # # #EVAL B stage
# python llava/eval/run_eval_modified.py \
#     --gt /scratch/ltl2113/LLaVA-Med/data/Vqa/VQA-RAD/test.json \
#     --pred $answers_file




# python llava/eval/run_med_datasets_eval_batch.py \
#     --model-name /scratch/ltl2113/LLaVA-Med/checkpoints/llava-med-7b-pretrain-quantized-2 \
#     --question-file /scratch/ltl2113/LLaVA-Med/data/Vqa/VQA-RAD/test.json \
#     --image-folder /scratch/ltl2113/LLaVA-Med/data/Vqa/VQA-RAD/test \
#     --answers-file /scratch/ltl2113/LLaVA-Med/data/eval/quantized_LLavaMED_eval_pretrained_VQA_epoch15.jsonl
    
python /scratch/ltl2113/LLaVA-Med/quantize_trial_4.py \
    --model-name /scratch/ltl2113/LLaVA-Med/checkpoints/llava-med-7b-pretrain \
    --question-file /scratch/ltl2113/LLaVA-Med/data/Vqa/VQA-RAD/test.json \
    --image-folder /scratch/ltl2113/LLaVA-Med/data/Vqa/VQA-RAD/test \
    --answers-file /scratch/ltl2113/LLaVA-Med/data/eval/quantized_LLavaMED_eval_pretrained_VQA_epoch15.jsonl
    
