#!/bin/bash
# Training launcher script for HP840

# Activate virtual environment
source /aidata/projects/cyberque-finetune/venv/bin/activate

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

echo ""
echo "============================================"
echo "HP840 Training Environment"
echo "============================================"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)"
echo "Python: $(python3 --version)"
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python3 -c 'import torch; print(torch.version.cuda)')"
echo "============================================"
echo ""
echo "Available scripts:"
echo "  1. train_lora.py      - Standard LoRA/QLoRA training"
echo "  2. train_unsloth.py   - Unsloth (2x faster, recommended)"
echo "  3. merge_and_export.py - Merge LoRA and export to GGUF"
echo "  4. prepare_dataset.py  - Convert datasets to training format"
echo ""
echo "Example usage:"
echo "  python scripts/train_unsloth.py --model_name unsloth/llama-3.1-8b-bnb-4bit --dataset data/my_data.jsonl"
echo ""

# If arguments provided, run them
if [ $# -gt 0 ]; then
    echo "Running: $@"
    exec "$@"
else
    echo "Starting bash in training environment..."
    exec bash
fi
