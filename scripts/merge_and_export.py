#!/usr/bin/env python3
"""
Merge LoRA adapters with base model and export to GGUF for Ollama
"""

import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import subprocess

def main(args):
    print(f"\n{'='*60}")
    print(f"Merging LoRA and Exporting to GGUF")
    print(f"Base Model: {args.base_model}")
    print(f"LoRA Path: {args.lora_path}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*60}\n")

    # Load base model
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    # Load and merge LoRA
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, args.lora_path)
    print("Merging weights...")
    model = model.merge_and_unload()

    # Save merged model
    merged_path = os.path.join(args.output_dir, "merged")
    print(f"Saving merged model to {merged_path}")
    model.save_pretrained(merged_path, safe_serialization=True)
    tokenizer.save_pretrained(merged_path)

    # Convert to GGUF if requested
    if args.convert_gguf:
        print("\nConverting to GGUF format...")
        gguf_path = os.path.join(args.output_dir, f"{args.model_name}.gguf")
        
        # Use llama.cpp convert script
        convert_cmd = [
            "python3", "-m", "llama_cpp.convert",
            "--outfile", gguf_path,
            "--outtype", args.quant_type,
            merged_path
        ]
        
        try:
            subprocess.run(convert_cmd, check=True)
            print(f"GGUF saved to: {gguf_path}")
            
            # Create Ollama modelfile
            if args.create_modelfile:
                modelfile_path = os.path.join(args.output_dir, "Modelfile")
                with open(modelfile_path, "w") as f:
                    f.write(f"FROM {gguf_path}\n")
                    f.write(f"PARAMETER temperature 0.7\n")
                    f.write(f"PARAMETER top_p 0.9\n")
                    f.write(f"SYSTEM You are a helpful assistant.\n")
                print(f"Modelfile saved to: {modelfile_path}")
                print(f"\nTo import to Ollama: ollama create {args.model_name} -f {modelfile_path}")
        except Exception as e:
            print(f"GGUF conversion failed: {e}")
            print("You can manually convert using llama.cpp")

    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA and Export")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--lora_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="/aidata/projects/cyberque-finetune/outputs/merged")
    parser.add_argument("--model_name", type=str, default="custom-model")
    parser.add_argument("--convert_gguf", action="store_true")
    parser.add_argument("--quant_type", type=str, default="q4_k_m", choices=["f16", "q8_0", "q4_k_m", "q4_0"])
    parser.add_argument("--create_modelfile", action="store_true")
    args = parser.parse_args()
    main(args)
