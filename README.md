# Qwen2.5 SFT

This repository provides scripts for **fine-tuning Qwen2.5 models** on custom datasets using Hugging Face `transformers` and `trl`.

---

## Setting Up the Environment
---
To ensure compatibility, create a **Conda environment** and install dependencies:

```bash
conda create --name Qwen2.5_SFT python=3.12 -y
conda activate Qwen2.5_SFT

# Install PyTorch with CUDA support (adjust based on GPU availability)
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

# Install required libraries
pip install trl transformers datasets accelerate

# Set up `accelerate` for multi-GPU training (configure when prompted)
accelerate config
```

## Run the Data Preprocessing Script
---
Before training, datasets must be formatted into a structured conversation format.

```bash
python scripts/data_processing.py \
  --dataset_name "<dataset1>,<dataset2>" \
  --output_dir "<path_to_processed_data>" \
  --cache_dir "<path_to_hf_cache>"
```

## Run the Training Script
---
After preprocessing, run the training script to fine-tune the model.

```bash
export ACCELERATE_CONFIG_FILE=<path_to_accelerate_config>
accelerate launch --config_file=$ACCELERATE_CONFIG_FILE scripts/train.py \
  --data_dir "<path_to_processed_data>" \
  --model_name_or_path "<model>" \
  --num_train_epochs 5 \
  --max_seq_length 4096 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-5 \
  --lr_scheduler_type cosine \
  --output_dir "<path_to_checkpoints>" \
  --save_strategy epoch \
  --logging_steps 5 \
  --packing \
  --gradient_checkpointing \
  --push_to_hub False \
  --fp16 \
  --trust_remote_code True \
  --torch_dtype float16 \
  --eval_strategy epoch \
  --report_to wandb \
  --eval_accumulation_steps 1 \
  --per_device_eval_batch_size 1
```