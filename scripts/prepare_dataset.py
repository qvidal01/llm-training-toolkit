#!/usr/bin/env python3
"""
Dataset Preparation Script
Converts various formats to training-ready JSONL
"""

import argparse
import json
import os
from pathlib import Path

def convert_conversations(input_file, output_file):
    """Convert conversation format to instruction/output pairs"""
    with open(input_file, "r") as f:
        data = json.load(f)
    
    examples = []
    for item in data:
        if "conversations" in item:
            convs = item["conversations"]
            for i in range(0, len(convs) - 1, 2):
                if convs[i]["from"] in ["human", "user"]:
                    examples.append({
                        "instruction": convs[i]["value"],
                        "output": convs[i + 1]["value"]
                    })
        elif "messages" in item:
            msgs = item["messages"]
            for i, msg in enumerate(msgs):
                if msg["role"] == "user" and i + 1 < len(msgs):
                    examples.append({
                        "instruction": msg["content"],
                        "output": msgs[i + 1]["content"]
                    })
    
    with open(output_file, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    
    print(f"Converted {len(examples)} examples to {output_file}")
    return examples

def convert_alpaca(input_file, output_file):
    """Convert Alpaca format (instruction, input, output)"""
    with open(input_file, "r") as f:
        data = json.load(f)
    
    examples = []
    for item in data:
        instruction = item.get("instruction", "")
        inp = item.get("input", "")
        output = item.get("output", "")
        
        if inp:
            instruction = f"{instruction}\n\nInput: {inp}"
        
        examples.append({
            "instruction": instruction,
            "output": output
        })
    
    with open(output_file, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    
    print(f"Converted {len(examples)} examples to {output_file}")
    return examples

def convert_text(input_file, output_file, chunk_size=2048):
    """Convert plain text to chunks"""
    with open(input_file, "r") as f:
        text = f.read()
    
    # Split into chunks
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        if len(chunk) > 100:  # Skip very small chunks
            chunks.append({"text": chunk})
    
    with open(output_file, "w") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")
    
    print(f"Created {len(chunks)} chunks in {output_file}")
    return chunks

def main(args):
    print(f"\nDataset Preparation")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Format: {args.format}\n")

    if args.format == "conversations":
        convert_conversations(args.input, args.output)
    elif args.format == "alpaca":
        convert_alpaca(args.input, args.output)
    elif args.format == "text":
        convert_text(args.input, args.output, args.chunk_size)
    else:
        print(f"Unknown format: {args.format}")
        return

    # Show sample
    print("\nSample from output:")
    with open(args.output, "r") as f:
        for i, line in enumerate(f):
            if i >= 2:
                break
            data = json.loads(line)
            print(json.dumps(data, indent=2)[:500])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset for training")
    parser.add_argument("--input", type=str, required=True, help="Input file path")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL path")
    parser.add_argument("--format", type=str, default="alpaca",
                       choices=["alpaca", "conversations", "text"],
                       help="Input format")
    parser.add_argument("--chunk_size", type=int, default=2048,
                       help="Chunk size for text format")
    args = parser.parse_args()
    main(args)
