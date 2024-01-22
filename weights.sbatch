#!/bin/bash

#SBATCH -p nvidia
#SBATCH -C 80g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=47:59:59
#SBATCH --mem=100GB
#SBATCH --job-name=7B_wz

# Environment setup
module purge

# Load Conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llava-med

# Install necessary packages
# pip install llama-recipes transformers datasets accelerate sentencepiece protobuf==3.20 py7zr scipy peft bitsandbytes fire torch_tb_profiler ipywidgets

# Set environment variables
export TRANSFORMERS_CACHE="/scratch/ltl2113/huggingface_cache"

# Determine the path to the conversion script
TRANSFORM=$(python -c "import transformers;print('/'.join(transformers.__file__.split('/')[:-1])+'/models/llama/convert_llama_weights_to_hf.py')")

# Set model directories
model_dir='/scratch/ltl2113/LLaVA-Med/data/Llama-2-7b'
model_size='7B'
hf_model_dir='/scratch/ltl2113/LLaVA-Med/data/hf_model'

# Convert LLaMA weights to Hugging Face format
python $TRANSFORM --input_dir $model_dir --model_size $model_size --output_dir $hf_model_dir

# Apply delta to the converted model
python3 -u -m llava.model.apply_delta \
    --base $hf_model_dir \
    --target /scratch/ltl2113/LLaVA-Med/model \
    --delta liuhaotian/LLaVA-7b-delta-v0
