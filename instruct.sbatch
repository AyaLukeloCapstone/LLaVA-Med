#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH -p compute
#SBATCH --mem=40GB
#SBATCH --time=96:00:00
#SBATCH --job-name=Data
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err


# Environment setup
module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llavamed

# Set the number of threads for ThreadPoolExecutor
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASKi 


python llava/instruct/instruct_generate.py \
    --input_path data/instruct/llava_med_instruct_fig_captions.json \
    --output_path data/instruct/llava_med_instruct_60k_inline_mentions_gen.json \
    --max_size 60000 \
    --use_inline_mentions True

python llava/instruct/instruct_postprocess.py \
    --input_path data/instruct/llava_med_instruct_60k_inline_mentions_gen.json \
    --output_path data/instruct/llava_med_instruct_60k_inline_mentions_post.json

