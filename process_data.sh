#!/bin/bash
#SBATCH --job-name=data_spliting
#SBATCH --output=logs/data_splitting.out
#SBATCH --error=logs/data_splitting.err
#SBATCH --partition=eddy
#SBATCH -c 4
#SBATCH --mem=50G
#SBATCH --time=72:00:00



module load python/3.12.5-fasrc01

conda deactivate
source activate NLP 

# Set env variables
export HF_DATASETS_CACHE=/n/eddy_lab/users/asarkar/personal_projects/Qwen_SFT/hf_datasets_cache

# Run the data processing script
python scripts/data_processing.py \
  --dataset_name "allenai/SciRIFF,TIGER-Lab/WebInstructSub" \
  --output_dir /n/eddy_lab/users/asarkar/personal_projects/Qwen2.5-SFT-SciRIFF/datasets/SciRIFF_X_WebInstructSub_Full \
  --cache_dir /n/eddy_lab/users/asarkar/personal_projects/Qwen2.5-SFT-SciRIFF/hf_datasets_cache \