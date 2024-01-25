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

# Set environment variables
export HF_HOME="/scratch/ltl2113/huggingface_cache"

# Start the controller in the background
python -u -m llava.serve.controller --host 0.0.0.0 --port 10000 &

# Wait for a short time to ensure the controller starts
sleep 10

# Launch the model worker
 python3 -u -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path /scratch/ltl2113/LLaVA-Med/model --multi-modal
sleep 10

python3 -u -m llava.serve.test_message --model-name LLaVA-Med-7B --controller http://localhost:10000
sleep 10

python3 -u -m llava.serve.gradio_web_server --controller http://localhost:10000
