# LLM Training Toolkit

A comprehensive toolkit for fine-tuning Large Language Models (LLMs) using LoRA/QLoRA with Unsloth optimization. Designed for local GPU training on systems like RTX 3090/4090.

## Features

- **Interactive Training Wizard** - Step-by-step guided training with `interactive-trainer.py`
- **Unsloth Optimization** - 2x faster training with 70% less VRAM
- **Multiple Training Methods** - Standard LoRA, QLoRA, and Unsloth-optimized training
- **Ollama Integration** - Direct export to GGUF format for Ollama deployment
- **Synology NAS Backup** - Automatic model backup to network storage
- **Dataset Preparation** - Convert various formats to training-ready JSONL

## Quick Start

### 1. Setup Environment

```bash
# Clone the repo
git clone https://github.com/yourusername/llm-training-toolkit.git
cd llm-training-toolkit

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Interactive Trainer

The easiest way to train a model:

```bash
./interactive-trainer.py
```

This will guide you through:
1. Selecting a base model
2. Choosing your dataset
3. Configuring training parameters
4. Starting training
5. Exporting to Ollama

### 3. Command Line Training

For scripted/automated training:

```bash
python scripts/train_unsloth.py \
    --model_name unsloth/llama-3.1-8b-bnb-4bit \
    --dataset data/my_training_data.json \
    --output_dir outputs/my-model \
    --epochs 3 \
    --batch_size 2 \
    --save_gguf
```

## Directory Structure

```
llm-training-toolkit/
├── interactive-trainer.py  # Interactive training wizard
├── train.sh                # Environment setup script
├── scripts/
│   ├── train_unsloth.py    # Unsloth training (recommended)
│   ├── train_lora.py       # Standard LoRA training
│   ├── prepare_dataset.py  # Dataset conversion
│   └── merge_and_export.py # Merge LoRA & export to GGUF
├── data/                   # Training datasets
├── configs/                # Training configurations
└── outputs/                # Trained models
```

## Training Data Format

### Alpaca Format (Recommended)

Create a `.json` or `.jsonl` file with instruction/output pairs:

```json
{"instruction": "What services does your company offer?", "output": "We offer AI automation, cloud infrastructure, and security solutions."}
{"instruction": "How do I contact support?", "output": "You can reach our support team at support@example.com or call 1-800-XXX-XXXX."}
{"instruction": "What are your business hours?", "output": "We're available Monday through Friday, 8 AM to 6 PM Central Time."}
```

### With Input Field

For tasks that require additional context:

```json
{"instruction": "Summarize this text", "input": "Long article text here...", "output": "Brief summary..."}
```

### Converting Other Formats

```bash
# From conversation format (ShareGPT style)
python scripts/prepare_dataset.py --input data/conversations.json --output data/prepared.jsonl --format conversations

# From plain text
python scripts/prepare_dataset.py --input data/knowledge.txt --output data/prepared.jsonl --format text
```

## Supported Base Models

Unsloth-optimized models (fastest):

| Model | Size | VRAM Required | Best For |
|-------|------|---------------|----------|
| `unsloth/llama-3.1-8b-bnb-4bit` | 8B | ~8GB | General purpose |
| `unsloth/Qwen2.5-7B-bnb-4bit` | 7B | ~6GB | General purpose |
| `unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit` | 7B | ~6GB | Code generation |
| `unsloth/mistral-7b-instruct-bnb-4bit` | 7B | ~6GB | Instruction following |
| `unsloth/gemma-2-9b-bnb-4bit` | 9B | ~10GB | Reasoning tasks |
| `unsloth/Phi-3.5-mini-instruct-bnb-4bit` | 3.8B | ~4GB | Fast inference |

## Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 3 | Number of training passes through the dataset |
| `--batch_size` | 2 | Examples processed simultaneously (reduce if OOM) |
| `--learning_rate` | 2e-4 | How fast the model learns (0.0002) |
| `--lora_r` | 16 | LoRA rank - complexity of adaptation |
| `--lora_alpha` | 16 | LoRA scaling factor |
| `--max_seq_length` | 2048 | Maximum input length in tokens |
| `--gradient_accumulation` | 8 | Effective batch size multiplier |

### Recommended Settings by Dataset Size

| Dataset Size | Epochs | Batch Size | Learning Rate |
|--------------|--------|------------|---------------|
| < 100 examples | 5-10 | 1-2 | 1e-4 |
| 100-1000 examples | 3-5 | 2-4 | 2e-4 |
| 1000+ examples | 1-3 | 4-8 | 2e-4 |

## Exporting to Ollama

### Automatic (during training)

```bash
python scripts/train_unsloth.py \
    --model_name unsloth/llama-3.1-8b-bnb-4bit \
    --dataset data/my_data.json \
    --save_gguf
```

### Manual Export

```bash
# After training, import to Ollama
cd outputs/my-model
ollama create my-custom-model -f Modelfile

# Test the model
ollama run my-custom-model
```

### Custom Modelfile

Create a `Modelfile` for custom system prompts:

```dockerfile
FROM ./model.gguf

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 4096

SYSTEM You are a helpful assistant for ACME Corp. You help customers with product questions and support requests.
```

## Monitoring Training

### GPU Usage

```bash
# In a separate terminal
watch -n 1 nvidia-smi
```

### Training Metrics

Training logs are saved to `outputs/MODEL_NAME/` and can be viewed with TensorBoard:

```bash
tensorboard --logdir outputs/
# Open http://localhost:6006
```

### Weights & Biases

If you have a W&B account, training metrics are automatically logged:

```bash
wandb login YOUR_API_KEY
```

## Synology NAS Backup

To automatically backup trained models to Synology:

```bash
python scripts/train_unsloth.py \
    --model_name unsloth/llama-3.1-8b-bnb-4bit \
    --dataset data/my_data.json \
    --save_gguf \
    --backup_synology
```

Configure the backup path in `scripts/train_unsloth.py`:

```python
SYNOLOGY_BACKUP_PATH = "/mnt/nas/projects/ai-models"
```

## Troubleshooting

### Out of Memory (OOM)

```bash
# Reduce batch size
--batch_size 1

# Reduce sequence length
--max_seq_length 1024

# Use a smaller model
--model_name unsloth/Phi-3.5-mini-instruct-bnb-4bit
```

### GGUF Conversion Fails

If automatic GGUF conversion fails, use Ollama's direct import:

```bash
# Create a Modelfile pointing to safetensors
cat > outputs/my-model/Modelfile << 'EOF'
FROM ./model.safetensors
PARAMETER temperature 0.7
SYSTEM You are a helpful assistant.
EOF

# Import directly
ollama create my-model -f outputs/my-model/Modelfile
```

### Slow Training

- Ensure you're using Unsloth models (`unsloth/...`)
- Check GPU utilization with `nvidia-smi`
- Increase batch size if VRAM allows
- Enable gradient accumulation for effective larger batches

## Requirements

- Python 3.10+
- NVIDIA GPU with 8GB+ VRAM (RTX 3090/4090 recommended)
- CUDA 11.8+
- 50GB+ disk space for models

### Python Dependencies

```txt
torch>=2.1.0
transformers>=4.36.0
datasets>=2.16.0
accelerate>=0.25.0
peft>=0.7.0
trl>=0.7.4
unsloth>=2024.1
bitsandbytes>=0.41.0
wandb
tensorboard
```

## Examples

### Training a Customer Support Bot

```bash
# 1. Prepare your FAQ data
cat > data/support_faq.json << 'EOF'
{"instruction": "How do I reset my password?", "output": "To reset your password, click 'Forgot Password' on the login page, enter your email, and follow the instructions sent to your inbox."}
{"instruction": "What are your support hours?", "output": "Our support team is available Monday through Friday, 9 AM to 5 PM EST. For urgent issues, use our 24/7 emergency line."}
EOF

# 2. Train the model
python scripts/train_unsloth.py \
    --model_name unsloth/llama-3.1-8b-bnb-4bit \
    --dataset data/support_faq.json \
    --output_dir outputs/support-bot \
    --epochs 5 \
    --save_gguf

# 3. Import to Ollama
ollama create support-bot -f outputs/support-bot/Modelfile

# 4. Test it
ollama run support-bot "How do I reset my password?"
```

### Training a Code Assistant

```bash
python scripts/train_unsloth.py \
    --model_name unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit \
    --dataset data/code_examples.json \
    --output_dir outputs/code-assistant \
    --epochs 3 \
    --max_seq_length 4096 \
    --save_gguf
```

## License

MIT License - see LICENSE file for details.

## Author

Quinn Vidal / AIQSO

## Resources

- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [Ollama Documentation](https://ollama.ai/docs)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
