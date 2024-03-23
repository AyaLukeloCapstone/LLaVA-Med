#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH -p compute
#SBATCH --mem=40GB
#SBATCH --time=96:00:00
#SBATCH --job-name=Data
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err


# Environment setup
module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llava-med

# Set the number of threads for ThreadPoolExecutor
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK


# # Create directories and download files
# mkdir -p data/alignment
# cd data/alignment
# wget https://hanoverprod.z21.web.core.windows.net/med_llava/alignment/llava_med_alignment_500k.json
# cd ..

# mkdir -p instruct
# cd instruct
# wget https://hanoverprod.z21.web.core.windows.net/med_llava/instruct/llava_med_instruct_10k.json
# wget https://hanoverprod.z21.web.core.windows.net/med_llava/instruct/llava_med_instruct_60k.json
# wget https://hanoverprod.z21.web.core.windows.net/med_llava/instruct/llava_med_instruct_60k_inline_mention.json
# wget https://hanoverprod.z21.web.core.windows.net/med_llava/instruct/llava_med_instruct_fig_captions.json
# cd ..

# mkdir -p eval
# cd eval
# wget https://hanoverprod.z21.web.core.windows.net/med_llava/eval/llava_med_eval_qa50_qa.jsonl
# wget https://hanoverprod.z21.web.core.windows.net/med_llava/eval/llava_med_eval_qa50_fig_captions.json
# wget https://hanoverprod.z21.web.core.windows.net/med_llava/eval/llava_med_qa50_instruct_caption_in_text_cleaned-60k-3epoch.json
# cd ..

# wget https://hanoverprod.z21.web.core.windows.net/med_llava/llava_med_image_urls.jsonl
# cd data
# mkdir -p pmc_articles
# mkdir -p images

# cd ..

# Install Python packages and run Python scripts
pip install tqdm
python llava/data/download_images.py --input_path data/llava_med_image_urls.jsonl --pmc_output_path data/pmc_articles/ --images_output_path data/Allimages