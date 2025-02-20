import datasets
from datasets import load_from_disk
from transformers import AutoTokenizer
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
)

# Disable disk space check if needed
datasets.builder.has_sufficient_disk_space = lambda needed_bytes, directory='.': True

def main():
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()

    dataset = load_from_disk(script_args.dataset_name)
    train_dataset = dataset["train"]
    if training_args.eval_strategy != "no" and "validation" in dataset:
        eval_dataset = dataset["validation"]
    else:
        eval_dataset = None

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
        use_fast=True,
    )

    trainer = SFTTrainer(
        model=model_config.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=None,  # Full fine-tuning
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub()

if __name__ == "__main__":
    main()