#!/usr/bin/env python3
"""
LoRA Fine-tuning Script for HP840 (RTX 3090)
Optimized for 24GB VRAM using QLoRA (4-bit quantization)
"""

import argparse
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

def main(args):
    print(f"\n{'='*60}")
    print(f"LoRA Fine-tuning on HP840")
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*60}\n")

    # Check GPU
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # 4-bit quantization config (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model
    print(f"\nLoading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # LoRA configuration
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    print(f"\nLoading dataset: {args.dataset}")
    if os.path.exists(args.dataset):
        dataset = load_dataset("json", data_files=args.dataset, split="train")
    else:
        dataset = load_dataset(args.dataset, split="train")

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
        warmup_ratio=0.03,
        optim="paged_adamw_8bit",
        report_to="tensorboard",
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        dataset_text_field="text" if "text" in dataset.column_names else None,
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save
    print(f"\nSaving model to {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="/aidata/projects/cyberque-finetune/outputs/lora-model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    args = parser.parse_args()
    main(args)
