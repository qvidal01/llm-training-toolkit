#!/usr/bin/env python3
"""
Unsloth Fine-tuning Script - 2x faster training with 70% less VRAM
Best option for RTX 3090 (24GB)
"""

import argparse
import os
import shutil
import subprocess
import torch
from pathlib import Path
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

# Supported models for Unsloth (4x faster)
UNSLOTH_MODELS = [
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/llama-3-8b-bnb-4bit",
    "unsloth/llama-3.1-8b-bnb-4bit",
    "unsloth/Qwen2.5-7B-bnb-4bit",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct-bnb-4bit",
]

# Synology NAS backup location
SYNOLOGY_BACKUP_PATH = "/mnt/nas/projects/ai-models"


def backup_to_synology(model_name, output_dir):
    """Backup trained model to Synology NAS"""
    synology_path = Path(SYNOLOGY_BACKUP_PATH)
    
    if not synology_path.exists():
        print(f"\n⚠ Synology not mounted at {SYNOLOGY_BACKUP_PATH}, skipping backup")
        return False
    
    # Create model directory on Synology
    model_backup_dir = synology_path / model_name
    model_backup_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📦 Backing up model to Synology: {model_backup_dir}")
    
    # Get the Ollama model blob path
    try:
        result = subprocess.run(
            ['ollama', 'show', model_name, '--modelfile'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            # Extract the FROM path (the GGUF blob)
            for line in result.stdout.split('\n'):
                if line.startswith('FROM /'):
                    blob_path = line.replace('FROM ', '').strip()
                    if os.path.exists(blob_path):
                        dest_gguf = model_backup_dir / f"{model_name}.gguf"
                        print(f"  Copying model blob to {dest_gguf}...")
                        shutil.copy2(blob_path, dest_gguf)
                        
                        # Create Modelfile for easy import
                        modelfile_content = f"""FROM ./{model_name}.gguf

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 4096

SYSTEM You are a helpful AI assistant trained by AIQSO.
"""
                        modelfile_path = model_backup_dir / "Modelfile"
                        with open(modelfile_path, 'w') as f:
                            f.write(modelfile_content)
                        
                        print(f"  ✓ Model backed up to Synology!")
                        print(f"    Location: {model_backup_dir}")
                        print(f"    To import elsewhere: cd {model_backup_dir} && ollama create {model_name} -f Modelfile")
                        return True
    except Exception as e:
        print(f"  ⚠ Could not backup to Synology: {e}")
    
    # Fallback: copy the HuggingFace model files
    print("  Copying training output files...")
    output_path = Path(output_dir)
    if output_path.exists():
        for file in output_path.iterdir():
            if file.suffix in ['.safetensors', '.json', '.gguf', '.bin']:
                dest = model_backup_dir / file.name
                print(f"    {file.name}...")
                shutil.copy2(file, dest)
        print(f"  ✓ Training files backed up to Synology!")
        return True
    
    return False


def main(args):
    print(f"\n{'='*60}")
    print(f"Unsloth Fine-tuning (2x faster, 70% less VRAM)")
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*60}\n")

    # Load model with Unsloth
    print(f"Loading {args.model_name} with Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,  # Auto-detect
        load_in_4bit=True,
    )

    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # Load dataset
    print(f"\nLoading dataset: {args.dataset}")
    if os.path.exists(args.dataset):
        dataset = load_dataset("json", data_files=args.dataset, split="train")
    else:
        dataset = load_dataset(args.dataset, split="train")

    # Format dataset for chat
    def format_prompt(example):
        if "instruction" in example and "output" in example:
            text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
        elif "text" in example:
            text = example["text"]
        elif "messages" in example:
            text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
        else:
            text = str(example)
        return {"text": text}

    dataset = dataset.map(format_prompt)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        warmup_steps=5,
        optim="adamw_8bit",
        seed=42,
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
    )

    # Train
    print("\nStarting Unsloth training...")
    gpu_stats = torch.cuda.get_device_properties(0)
    print(f"GPU: {gpu_stats.name} ({gpu_stats.total_memory / 1e9:.1f}GB)")
    
    trainer.train()

    # Save
    print(f"\nSaving model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save to GGUF for Ollama
    if args.save_gguf:
        print("\nSaving as GGUF for Ollama...")
        model.save_pretrained_gguf(
            args.output_dir,
            tokenizer,
            quantization_method=args.quant_type
        )
        print(f"GGUF saved to: {args.output_dir}")
        print(f"Import to Ollama: ollama create {args.model_name.split('/')[-1]}-ft -f Modelfile")

    # Backup to Synology if requested
    if args.backup_synology:
        # Get model name from output directory
        model_name = Path(args.output_dir).name
        backup_to_synology(model_name, args.output_dir)

    print("\n✓ Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unsloth Fine-tuning")
    parser.add_argument("--model_name", type=str, default="unsloth/llama-3.1-8b-bnb-4bit",
                       help=f"Supported: {', '.join(UNSLOTH_MODELS)}")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="/aidata/projects/cyberque-finetune/outputs/unsloth-model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--save_gguf", action="store_true")
    parser.add_argument("--quant_type", type=str, default="q4_k_m")
    parser.add_argument("--backup_synology", action="store_true", 
                       help="Backup trained model to Synology NAS")
    args = parser.parse_args()
    main(args)
