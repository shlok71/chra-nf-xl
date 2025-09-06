import os
import subprocess
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, TrainerCallback
from datasets import load_dataset

# --- Storage Check ---
def check_storage():
    """Checks disk usage and exits if it exceeds 15GB."""
    try:
        result = subprocess.run(['df', '-h', '.'], capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split('\n')
        if len(lines) > 1:
            # Output of df -h: Filesystem Size Used Avail Use% Mounted on
            # We parse the 'Used' value.
            used_str = lines[1].split()[2]
            if 'G' in used_str:
                used_gb = float(used_str.replace('G', ''))
                if used_gb > 15:
                    print(f"Error: Storage usage ({used_gb}GB) exceeds the 15GB limit.")
                    raise SystemExit("Stopping training due to storage limit.")
    except (subprocess.CalledProcessError, FileNotFoundError, IndexError, ValueError) as e:
        print(f"Could not check storage: {e}. Proceeding with caution.")

class StorageCallback(TrainerCallback):
    """A callback to check storage at each training step."""
    def on_step_begin(self, args, state, control, **kwargs):
        check_storage()

# --- Main Training Script ---
def main():
    # --- Configuration ---
    # NLP: 'wikitext', 'openwebtext', 'c4'
    # Vision: 'imagenet-1k', 'cifar10' (would require a different model and tokenization)
    dataset_name = 'allenai/c4'
    dataset_config = 'en' # For 'c4' dataset
    model_name = 'distilgpt2'
    output_dir = './training_results'
    final_model_dir = './final_model'
    batch_size = 2 # Keep batch size small
    max_train_steps = 1000 # Limit training to 1000 steps for this example

    # --- 1. Load Streaming Dataset ---
    print(f"Loading streaming dataset: {dataset_name} ({dataset_config})")
    train_dataset = load_dataset(dataset_name, dataset_config, split='train', streaming=True)

    # For a streaming dataset, we can't shuffle the whole dataset, but we can shuffle a buffer.
    train_dataset = train_dataset.shuffle(buffer_size=10000, seed=42)

    # --- 2. Initialize Model and Tokenizer ---
    print(f"Loading model and tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True)

    # --- Add PEFT with LoRA ---
    from peft import LoraConfig, get_peft_model
    print("Applying LoRA to the model...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        # This function is specific to NLP datasets like 'c4'.
        # Vision datasets would need a different preprocessing function (e.g., using a Feature Extractor).
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

    tokenized_dataset = train_dataset.map(tokenize_function, batched=True)

    # --- 3. Configure and Run Training ---
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        max_steps=max_train_steps,
        logging_dir=f'{output_dir}/logs',
        logging_steps=100,
        save_strategy="no", # We will save the final model manually.
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    from transformers import DataCollatorForLanguageModeling
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        callbacks=[StorageCallback()],
    )

    print("Starting training...")
    try:
        trainer.train()
    except SystemExit as e:
        print(f"Training was halted: {e}")

    # --- 4. Save Final Model ---
    print(f"Training complete. Saving final model to {final_model_dir}")
    os.makedirs(final_model_dir, exist_ok=True)
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    print("Model saving complete.")

if __name__ == '__main__':
    main()
