import os
import argparse
import datasets
from datasets import load_dataset, concatenate_datasets, DatasetDict

# Use the HF cache directory from environment variables (set in your shell script)
CACHE_DIR = os.environ.get('HF_DATASETS_CACHE', './hf_datasets_cache')

# Disable disk space check if needed
datasets.builder.has_sufficient_disk_space = lambda needed_bytes, directory='.': True

def format_chat_example(example: dict) -> dict:
    """
    Convert the input example into a unified conversation format.
    
    Handles datasets with:
      - 'input'/'output' pairs (mapped to 'user' and 'assistant' roles).
      - 'question'/'answer' pairs (mapped to 'user' and 'assistant' roles).
      - Pre-existing 'messages' fields.
    
    Ensures that the returned format always follows:
      [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}].
    
    Raises ValueError if the expected fields are missing.
    """
    if "messages" in example:
        # If the dataset already has a structured messages format, use it as is
        return {"messages": example["messages"]}

    elif "input" in example and "output" in example:
        # Convert 'input' and 'output' fields to user-assistant conversation
        messages = [
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": example["output"]}
        ]
        return {"messages": messages}

    elif "question" in example and "answer" in example:
        # Convert 'question' and 'answer' fields to user-assistant conversation
        messages = [
            {"role": "user", "content": example["question"]},
            {"role": "assistant", "content": example["answer"]}
        ]
        return {"messages": messages}

    else:
        raise ValueError("Example must contain either 'messages', 'input'/'output', or 'question'/'answer'.")

def load_and_process_datasets(dataset_names_str: str, cache_dir: str, seed: int = 100) -> DatasetDict:
    """
    Load and process one or more datasets specified as a comma‐separated string.
    - For 'allenai/SciRIFF', the "4096" configuration is used.

    Each example is then formatted via format_chat_example.
    The training and (if available) evaluation splits are concatenated and reproducibly shuffled.
    
    Returns a DatasetDict with keys "train" and (optionally) "validation".
    """
    dataset_names = [ds.strip() for ds in dataset_names_str.split(",")]
    train_datasets = []
    eval_datasets = []

    for ds in dataset_names:
        if ds == "allenai/SciRIFF":
            ds_loaded = load_dataset("allenai/SciRIFF", "4096", cache_dir=cache_dir) # to add 4096 setting

        else:
            ds_loaded = load_dataset(ds, cache_dir=cache_dir)

        if "train" not in ds_loaded:
            raise ValueError(f"Dataset {ds} does not have a 'train' split.")
        
        # Format each example for chat
        ds_loaded["train"] = ds_loaded["train"].map(format_chat_example, batched=False)
        train_datasets.append(ds_loaded["train"])
        if "validation" in ds_loaded:
            ds_loaded["validation"] = ds_loaded["validation"].map(format_chat_example, batched=False)
            eval_datasets.append(ds_loaded["validation"])

    # Concatenate and reproducibly shuffle training datasets.
    if len(train_datasets) > 1:
        train_dataset = concatenate_datasets(train_datasets).shuffle(seed=seed)
    else:
        train_dataset = train_datasets[0] # No need to shuffle if there's only one dataset

    # Concatenate and reproducibly shuffle evaluation datasets if available.
    if eval_datasets:
        if len(eval_datasets) > 1:
            eval_dataset = concatenate_datasets(eval_datasets).shuffle(seed=seed)
        else:
            eval_dataset = eval_datasets[0] # No need to shuffle if there's only one dataset
    else:
        eval_dataset = None

    ds_dict = {"train": train_dataset}
    if eval_dataset is not None:
        ds_dict["validation"] = eval_dataset
    return DatasetDict(ds_dict)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Comma-separated list of dataset names (e.g., 'allenai/SciRIFF,TIGER-Lab/WebInstructSub')")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory where the preprocessed dataset will be saved")
    parser.add_argument("--cache_dir", type=str, default=CACHE_DIR,
                        help="Directory for dataset caching")
    args = parser.parse_args()

    processed_dataset = load_and_process_datasets(args.dataset_name, args.cache_dir)
    processed_dataset.save_to_disk(args.output_dir)
    print(f"Preprocessed dataset saved to {args.output_dir}")

if __name__ == "__main__":
    main()