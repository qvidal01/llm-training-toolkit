#!/usr/bin/env python3
"""
AIQSO Interactive LLM Training Assistant
Guides you through fine-tuning models step-by-step

Usage:
    ./interactive-trainer.py

Author: Quinn Vidal / AIQSO
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path
from datetime import datetime

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Navigation constants
NAV_BACK = 'back'
NAV_QUIT = 'quit'
NAV_RESTART = 'restart'

class NavigationException(Exception):
    """Exception for navigation actions"""
    def __init__(self, action):
        self.action = action

def clear_screen():
    os.system('clear')

def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(60)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")

def print_nav_help():
    """Print navigation options reminder"""
    print(f"{Colors.CYAN}Navigation: [b]ack  [r]estart  [q]uit{Colors.ENDC}")

def print_step(num, text):
    print(f"{Colors.CYAN}{Colors.BOLD}[Step {num}]{Colors.ENDC} {Colors.CYAN}{text}{Colors.ENDC}")

def print_info(text):
    print(f"{Colors.BLUE}ℹ {text}{Colors.ENDC}")

def print_success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.ENDC}")

def print_warning(text):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.ENDC}")

def print_error(text):
    print(f"{Colors.RED}✗ {text}{Colors.ENDC}")

def print_tip(text):
    print(f"{Colors.YELLOW}💡 TIP: {text}{Colors.ENDC}")

def print_command(text):
    print(f"{Colors.GREEN}$ {text}{Colors.ENDC}")

def print_bold(text):
    print(f"{Colors.BOLD}{text}{Colors.ENDC}")

def check_nav(value):
    """Check if input is a navigation command"""
    val = value.strip().lower()
    if val in ('b', 'back'):
        raise NavigationException(NAV_BACK)
    elif val in ('q', 'quit', 'exit'):
        raise NavigationException(NAV_QUIT)
    elif val in ('r', 'restart', 'start over'):
        raise NavigationException(NAV_RESTART)
    return value

def wait_for_enter(prompt="Press Enter to continue..."):
    result = input(f"\n{Colors.BOLD}{prompt}{Colors.ENDC} ")
    check_nav(result)

def ask_yes_no(prompt, default='y'):
    while True:
        choice = input(f"{prompt} [{'Y/n' if default == 'y' else 'y/N'}]: ").strip().lower()
        check_nav(choice)
        if not choice:
            return default == 'y'
        if choice in ('y', 'yes'):
            return True
        if choice in ('n', 'no'):
            return False

def ask_choice(prompt, options, default=0):
    print(f"\n{prompt}")
    for i, opt in enumerate(options):
        marker = "→" if i == default else " "
        print(f"  {marker} [{i+1}] {opt}")
    print_nav_help()
    while True:
        choice = input(f"Enter choice [1-{len(options)}] (default: {default+1}): ").strip()
        check_nav(choice)
        if not choice:
            return default
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return idx
        except ValueError:
            pass
        print_error("Invalid choice. Try again.")

def get_input(prompt, default=None):
    if default:
        result = input(f"{prompt} [{default}]: ").strip()
        check_nav(result)
        return result if result else default
    result = input(f"{prompt}: ").strip()
    check_nav(result)
    return result

def check_gpu():
    """Check GPU status and return info"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,memory.free,memory.used,utilization.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(', ')
            return {
                'name': parts[0],
                'total': int(parts[1]),
                'free': int(parts[2]),
                'used': int(parts[3]),
                'util': int(parts[4])
            }
    except:
        pass
    return None

def check_venv():
    """Check if we're in the training venv"""
    return 'cyberque-finetune' in sys.prefix

def show_gpu_status():
    """Display GPU status"""
    gpu = check_gpu()
    if gpu:
        print_bold("\nGPU Status:")
        print(f"  Name: {gpu['name']}")
        print(f"  VRAM: {gpu['used']}MB / {gpu['total']}MB ({gpu['free']}MB free)")
        print(f"  Utilization: {gpu['util']}%")
        
        # VRAM warning
        if gpu['free'] < 8000:
            print_warning(f"Low VRAM! Only {gpu['free']}MB free. May need smaller model.")
        else:
            print_success(f"{gpu['free']}MB VRAM available - good for training")
    else:
        print_error("Could not detect GPU")

def list_datasets():
    """List available datasets"""
    data_dir = Path('/aidata/projects/cyberque-finetune/data')
    datasets = list(data_dir.glob('*.jsonl')) + list(data_dir.glob('*.json'))
    return datasets

def list_outputs():
    """List trained model outputs"""
    output_dir = Path('/aidata/projects/cyberque-finetune/outputs')
    outputs = [d for d in output_dir.iterdir() if d.is_dir()]
    return outputs

def create_sample_dataset():
    """Create a sample dataset for testing"""
    sample_data = [
        {"instruction": "What services does AIQSO provide?", "output": "AIQSO provides enterprise AI automation consulting, including custom AI models, workflow automation, cloud infrastructure, and security solutions."},
        {"instruction": "How can I contact AIQSO?", "output": "You can reach AIQSO through our website at aiqso.io, by email, or by scheduling a consultation call."},
        {"instruction": "What is the pricing for custom AI models?", "output": "AIQSO offers tiered pricing starting at $5,000 for basic custom assistants up to $50,000+ for enterprise solutions. Contact us for a custom quote."},
        {"instruction": "Do you offer on-premise deployment?", "output": "Yes! AIQSO specializes in on-premise AI deployment for clients with data privacy requirements. Your data never leaves your network."},
        {"instruction": "How long does it take to train a custom model?", "output": "Typical projects take 2-4 weeks from kickoff to deployment, depending on complexity and data requirements."},
    ]
    
    output_path = Path('/aidata/projects/cyberque-finetune/data/sample_dataset.jsonl')
    with open(output_path, 'w') as f:
        for item in sample_data:
            f.write(json.dumps(item) + '\n')
    
    return output_path

# ============================================================================
# MAIN MENU
# ============================================================================

def main_menu():
    while True:
        clear_screen()
        print_header("AIQSO Interactive LLM Trainer")
        
        show_gpu_status()
        
        if not check_venv():
            print_warning("Not in training venv! Run: source venv/bin/activate")
        else:
            print_success("Training environment active")
        
        print_bold("\nWhat would you like to do?")
        print("  [1] 🚀 Start New Training (Guided)")
        print("  [2] 📊 Prepare Dataset")
        print("  [3] 📦 Export Model to Ollama")
        print("  [4] 📋 View Training Checklist")
        print("  [5] 🔍 Check System Status")
        print("  [6] 📚 Quick Reference Guide")
        print("  [7] 🧪 Test Run (Sample Dataset)")
        print("  [q] Exit")
        
        choice = input(f"\n{Colors.BOLD}Enter choice: {Colors.ENDC}").strip().lower()
        
        try:
            if choice == '1':
                guided_training()
            elif choice == '2':
                prepare_dataset()
            elif choice == '3':
                export_to_ollama()
            elif choice == '4':
                show_checklist()
            elif choice == '5':
                check_system()
            elif choice == '6':
                quick_reference()
            elif choice == '7':
                test_run()
            elif choice in ('q', 'quit', 'exit'):
                print("\nGoodbye! Happy training! 🦙\n")
                sys.exit(0)
        except NavigationException as e:
            if e.action == NAV_QUIT:
                print("\nGoodbye! Happy training! 🦙\n")
                sys.exit(0)
            elif e.action in (NAV_BACK, NAV_RESTART):
                # Return to main menu
                continue

# ============================================================================
# GUIDED TRAINING
# ============================================================================

def guided_training():
    while True:
        try:
            _guided_training_flow()
            break
        except NavigationException as e:
            if e.action == NAV_QUIT:
                raise
            elif e.action == NAV_RESTART:
                continue
            elif e.action == NAV_BACK:
                return  # Go back to main menu

def _guided_training_flow():
    clear_screen()
    print_header("Guided Training Session")
    
    print("This wizard will guide you through training a custom LLM.")
    print("We'll go step-by-step, explaining each decision.")
    print()
    print(f"{Colors.YELLOW}In your second terminal, you can monitor with:{Colors.ENDC}")
    print("  watch -n 1 nvidia-smi")
    print()
    print_nav_help()
    
    wait_for_enter()
    
    # Step 1: Choose base model
    clear_screen()
    print_header("Step 1: Choose Base Model")
    
    print("The base model determines your starting capabilities.")
    print("Choose based on your use case and available VRAM.")
    print()
    
    gpu = check_gpu()
    vram = gpu['free'] if gpu else 24000
    
    models = [
        ("unsloth/llama-3.1-8b-bnb-4bit", "8B", "~8GB", "Best balance of quality and speed"),
        ("unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit", "7B", "~6GB", "Specialized for code"),
        ("unsloth/mistral-7b-instruct-bnb-4bit", "7B", "~6GB", "Great instruction following"),
        ("unsloth/gemma-2-9b-bnb-4bit", "9B", "~10GB", "Strong reasoning"),
        ("unsloth/Phi-3.5-mini-instruct-bnb-4bit", "3.8B", "~4GB", "Fast, efficient"),
    ]
    
    print_bold(f"Available VRAM: {vram}MB")
    print()
    print(f"{'Model':<50} {'Size':<8} {'VRAM':<10} {'Best For'}")
    print("-" * 100)
    
    for i, (model, size, vram_req, desc) in enumerate(models):
        print(f"[{i+1}] {model:<47} {size:<8} {vram_req:<10} {desc}")
    
    choice = ask_choice("\nSelect a model:", [m[0] for m in models], default=0)
    selected_model = models[choice][0]
    
    print_success(f"Selected: {selected_model}")
    
    print_tip("Unsloth models are pre-quantized for 2x faster training!")
    wait_for_enter()
    
    # Step 2: Select dataset
    clear_screen()
    print_header("Step 2: Select Dataset")
    
    print("Your dataset teaches the model your specific knowledge.")
    print("It should be in JSONL format with 'instruction' and 'output' fields.")
    print()
    
    datasets = list_datasets()
    
    if datasets:
        print_bold("Available datasets:")
        print()
        for i, ds in enumerate(datasets):
            size = ds.stat().st_size / 1024
            lines = sum(1 for _ in open(ds))
            print(f"  [{i+1}] {ds.name:<40} ({lines} examples, {size:.1f}KB)")
        
        print(f"  [{len(datasets)+1}] Create sample dataset")
        print(f"  [{len(datasets)+2}] Enter custom path")
        
        print_nav_help()
        choice = get_input(f"\nSelect dataset [1-{len(datasets)+2}]", "1")
        
        try:
            idx = int(choice) - 1
            if idx < len(datasets):
                dataset_path = str(datasets[idx])
            elif idx == len(datasets):
                dataset_path = str(create_sample_dataset())
                print_success(f"Created sample dataset: {dataset_path}")
            else:
                dataset_path = get_input("Enter full path to dataset")
        except:
            dataset_path = get_input("Enter full path to dataset")
    else:
        print_warning("No datasets found in data/ directory")
        if ask_yes_no("Create sample dataset for testing?"):
            dataset_path = str(create_sample_dataset())
            print_success(f"Created: {dataset_path}")
        else:
            dataset_path = get_input("Enter full path to dataset")
    
    # Show sample of dataset
    print_bold("\nDataset preview:")
    try:
        with open(dataset_path) as f:
            for i, line in enumerate(f):
                if i >= 2:
                    break
                data = json.loads(line)
                print(f"\n  Example {i+1}:")
                print(f"    Instruction: {data.get('instruction', 'N/A')[:60]}...")
                print(f"    Output: {data.get('output', 'N/A')[:60]}...")
    except Exception as e:
        print_error(f"Could not preview: {e}")
    
    wait_for_enter()
    
    # Step 3: Training parameters
    clear_screen()
    print_header("Step 3: Training Parameters")
    
    print("These parameters control how the model learns.")
    print("We'll use sensible defaults, but you can customize.")
    print()
    
    print_bold("Parameter explanations:")
    print("  • Epochs: How many times to go through the dataset (more = better fit, risk of overfitting)")
    print("  • Batch size: Examples processed at once (higher = faster, more VRAM)")
    print("  • Learning rate: How fast to learn (too high = unstable, too low = slow)")
    print("  • LoRA rank: Complexity of adaptation (higher = more parameters)")
    
    print_bold("\nRecommended defaults for your setup:")
    print_nav_help()
    
    epochs = int(get_input("Number of epochs", "3"))
    batch_size = int(get_input("Batch size", "2"))
    learning_rate = float(get_input("Learning rate", "2e-4"))
    lora_r = int(get_input("LoRA rank (r)", "16"))
    max_seq_length = int(get_input("Max sequence length", "2048"))
    
    # Step 4: Output location
    clear_screen()
    print_header("Step 4: Output Configuration")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    default_name = f"model_{timestamp}"
    print_nav_help()
    model_name = get_input("Name for your model", default_name)
    output_dir = f"/aidata/projects/cyberque-finetune/outputs/{model_name}"
    
    save_gguf = ask_yes_no("Export to GGUF for Ollama after training?", 'y')
    backup_synology = ask_yes_no("Backup model to Synology NAS?", 'y')
    
    # Step 5: Confirmation
    clear_screen()
    print_header("Step 5: Review & Start")
    
    print_bold("Training Configuration:")
    print()
    print(f"  Base Model:    {selected_model}")
    print(f"  Dataset:       {dataset_path}")
    print(f"  Output:        {output_dir}")
    print(f"  Epochs:        {epochs}")
    print(f"  Batch Size:    {batch_size}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  LoRA Rank:     {lora_r}")
    print(f"  Max Seq Len:   {max_seq_length}")
    print(f"  Export GGUF:   {'Yes' if save_gguf else 'No'}")
    print(f"  Backup to NAS: {'Yes' if backup_synology else 'No'}")
    
    # Generate command
    cmd = f"""python scripts/train_unsloth.py \\
    --model_name {selected_model} \\
    --dataset {dataset_path} \\
    --output_dir {output_dir} \\
    --epochs {epochs} \\
    --batch_size {batch_size} \\
    --learning_rate {learning_rate} \\
    --lora_r {lora_r} \\
    --max_seq_length {max_seq_length}"""
    
    if save_gguf:
        cmd += " \\\n    --save_gguf"
    if backup_synology:
        cmd += " \\
    --backup_synology"
    
    print_bold("\nCommand to run:")
    print(f"{Colors.GREEN}{cmd}{Colors.ENDC}")
    
    print(f"\n{Colors.YELLOW}In your other terminal, monitor with:{Colors.ENDC}")
    print("  watch -n 1 nvidia-smi")
    
    print_nav_help()
    if ask_yes_no("\nStart training now?", 'y'):
        print_bold("\nStarting training...")
        print("="*60)
        
        # Build actual command
        train_cmd = [
            'python', 'scripts/train_unsloth.py',
            '--model_name', selected_model,
            '--dataset', dataset_path,
            '--output_dir', output_dir,
            '--epochs', str(epochs),
            '--batch_size', str(batch_size),
            '--learning_rate', str(learning_rate),
            '--lora_r', str(lora_r),
            '--max_seq_length', str(max_seq_length),
        ]
        if save_gguf:
            train_cmd.append('--save_gguf')
        if backup_synology:
            train_cmd.append("--backup_synology")
        
        # Run training
        os.chdir('/aidata/projects/cyberque-finetune')
        result = subprocess.run(train_cmd)
        
        print("\n" + "="*60)
        if result.returncode == 0:
            print_success("Training completed successfully!")
            print(f"\nModel saved to: {output_dir}")
            
            if save_gguf:
                print_bold("\nTo import to Ollama:")
                print(f"  ollama create {model_name} -f {output_dir}/Modelfile")
        else:
            print_error("Training failed. Check the error messages above.")
    else:
        print("\nTraining cancelled. You can copy the command above and run it manually.")
    
    wait_for_enter()

# ============================================================================
# DATASET PREPARATION
# ============================================================================

def prepare_dataset():
    while True:
        try:
            _prepare_dataset_flow()
            break
        except NavigationException as e:
            if e.action == NAV_QUIT:
                raise
            elif e.action == NAV_RESTART:
                continue
            elif e.action == NAV_BACK:
                return

def _prepare_dataset_flow():
    clear_screen()
    print_header("Dataset Preparation")
    
    print("This tool helps you prepare data for training.")
    print()
    print_bold("Supported input formats:")
    print("  1. Alpaca format (instruction, input, output)")
    print("  2. Conversations format (ShareGPT style)")
    print("  3. Plain text (will be chunked)")
    print("  4. CSV (question/answer columns)")
    print()
    print_nav_help()
    
    choice = ask_choice("Select input format:", [
        "Alpaca JSON (instruction/output pairs)",
        "Conversations JSON (ShareGPT)",
        "Plain text file",
        "CSV file",
    ])
    
    input_path = get_input("Path to input file")
    output_name = get_input("Output filename (without extension)", "prepared_dataset")
    output_path = f"/aidata/projects/cyberque-finetune/data/{output_name}.jsonl"
    
    format_map = ['alpaca', 'conversations', 'text', 'csv']
    
    cmd = f"""python scripts/prepare_dataset.py \\
    --input {input_path} \\
    --output {output_path} \\
    --format {format_map[choice]}"""
    
    print_bold("\nCommand:")
    print_command(cmd)
    
    if ask_yes_no("\nRun this command?", 'y'):
        subprocess.run([
            'python', 'scripts/prepare_dataset.py',
            '--input', input_path,
            '--output', output_path,
            '--format', format_map[choice]
        ])
    
    wait_for_enter()

# ============================================================================
# EXPORT TO OLLAMA
# ============================================================================

def export_to_ollama():
    while True:
        try:
            _export_to_ollama_flow()
            break
        except NavigationException as e:
            if e.action == NAV_QUIT:
                raise
            elif e.action == NAV_RESTART:
                continue
            elif e.action == NAV_BACK:
                return

def _export_to_ollama_flow():
    clear_screen()
    print_header("Export Model to Ollama")
    
    outputs = list_outputs()
    
    if not outputs:
        print_error("No trained models found in outputs/")
        wait_for_enter()
        return
    
    print_bold("Available trained models:")
    print()
    for i, out in enumerate(outputs):
        print(f"  [{i+1}] {out.name}")
    
    print_nav_help()
    choice = int(get_input("Select model", "1")) - 1
    selected = outputs[choice]
    
    model_name = get_input("Name for Ollama model", selected.name)
    
    # Check for GGUF
    gguf_files = list(selected.glob("*.gguf"))
    
    if gguf_files:
        print_success(f"Found GGUF: {gguf_files[0].name}")
        gguf_path = str(gguf_files[0])
    else:
        print_warning("No GGUF found. Need to convert first.")
        # Would need to run conversion here
        wait_for_enter()
        return
    
    # Create Modelfile
    system_prompt = get_input("System prompt", "You are a helpful AI assistant.")
    
    modelfile_content = f"""FROM {gguf_path}
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 4096
SYSTEM {system_prompt}
"""
    
    modelfile_path = selected / "Modelfile"
    with open(modelfile_path, 'w') as f:
        f.write(modelfile_content)
    
    print_bold("\nImporting to Ollama...")
    result = subprocess.run(['ollama', 'create', model_name, '-f', str(modelfile_path)])
    
    if result.returncode == 0:
        print_success(f"Model '{model_name}' created!")
        print(f"\nTest it with: ollama run {model_name}")
    
    wait_for_enter()

# ============================================================================
# CHECKLIST
# ============================================================================

def show_checklist():
    try:
        _show_checklist_flow()
    except NavigationException as e:
        if e.action == NAV_QUIT:
            raise

def _show_checklist_flow():
    clear_screen()
    print_header("Pre-Training Checklist")
    
    checks = []
    
    # Check 1: GPU
    gpu = check_gpu()
    if gpu and gpu['free'] > 8000:
        checks.append(("✓", "GPU VRAM available", f"{gpu['free']}MB free"))
    else:
        checks.append(("✗", "GPU VRAM", "Low or unavailable"))
    
    # Check 2: Venv
    if check_venv():
        checks.append(("✓", "Virtual environment", "Active"))
    else:
        checks.append(("✗", "Virtual environment", "Not activated - run: source venv/bin/activate"))
    
    # Check 3: Dataset
    datasets = list_datasets()
    if datasets:
        checks.append(("✓", "Datasets available", f"{len(datasets)} found"))
    else:
        checks.append(("✗", "Datasets", "None found in data/"))
    
    # Check 4: Disk space
    try:
        result = subprocess.run(['df', '-h', '/aidata'], capture_output=True, text=True)
        if 'T' in result.stdout:
            checks.append(("✓", "Disk space", "Sufficient"))
    except:
        checks.append(("?", "Disk space", "Could not check"))
    
    # Check 5: Scripts
    scripts_dir = Path('/aidata/projects/cyberque-finetune/scripts')
    if (scripts_dir / 'train_unsloth.py').exists():
        checks.append(("✓", "Training scripts", "Present"))
    else:
        checks.append(("✗", "Training scripts", "Missing"))
    
    print(f"{'Status':<8} {'Check':<25} {'Details'}")
    print("-" * 60)
    for status, check, details in checks:
        color = Colors.GREEN if status == "✓" else Colors.RED if status == "✗" else Colors.YELLOW
        print(f"{color}{status:<8}{Colors.ENDC} {check:<25} {details}")
    
    all_pass = all(c[0] == "✓" for c in checks)
    
    if all_pass:
        print(f"\n{Colors.GREEN}{Colors.BOLD}All checks passed! Ready to train.{Colors.ENDC}")
    else:
        print(f"\n{Colors.YELLOW}Some checks need attention before training.{Colors.ENDC}")
    
    print_nav_help()
    wait_for_enter()

# ============================================================================
# SYSTEM CHECK
# ============================================================================

def check_system():
    try:
        _check_system_flow()
    except NavigationException as e:
        if e.action == NAV_QUIT:
            raise

def _check_system_flow():
    clear_screen()
    print_header("System Status")
    
    # GPU
    print_bold("GPU:")
    subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free,memory.used,temperature.gpu,utilization.gpu',
                   '--format=csv'])
    
    # Disk
    print_bold("\nDisk Space (/aidata):")
    subprocess.run(['df', '-h', '/aidata'])
    
    # Ollama
    print_bold("\nOllama Status:")
    result = subprocess.run(['systemctl', 'is-active', 'ollama'], capture_output=True, text=True)
    print(f"  Service: {result.stdout.strip()}")
    
    # Models
    print_bold("\nOllama Models:")
    subprocess.run(['ollama', 'list'])
    
    # Python
    print_bold("\nPython Environment:")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  Prefix: {sys.prefix}")
    
    try:
        import torch
        print(f"  PyTorch: {torch.__version__}")
        print(f"  CUDA: {torch.version.cuda}")
        print(f"  CUDA Available: {torch.cuda.is_available()}")
    except ImportError:
        print("  PyTorch: Not imported")
    
    print_nav_help()
    wait_for_enter()

# ============================================================================
# QUICK REFERENCE
# ============================================================================

def quick_reference():
    try:
        _quick_reference_flow()
    except NavigationException as e:
        if e.action == NAV_QUIT:
            raise

def _quick_reference_flow():
    clear_screen()
    print_header("Quick Reference Guide")
    
    print_bold("TRAINING COMMANDS")
    print("─────────────────")
    print("# Quick training with Unsloth (recommended)")
    print("python scripts/train_unsloth.py \\")
    print("    --model_name unsloth/llama-3.1-8b-bnb-4bit \\")
    print("    --dataset data/my_data.jsonl \\")
    print("    --epochs 3 --save_gguf")
    print()
    print("# Standard QLoRA training")
    print("python scripts/train_lora.py \\")
    print("    --model_name mistralai/Mistral-7B-v0.1 \\")
    print("    --dataset data/my_data.jsonl")
    print()
    
    print_bold("DATASET FORMAT")
    print("───────────────")
    print('# Alpaca format (recommended)')
    print('{"instruction": "What is X?", "output": "X is..."}')
    print()
    print('# With input field')
    print('{"instruction": "Summarize", "input": "Long text...", "output": "Summary..."}')
    print()
    
    print_bold("MONITORING")
    print("───────────")
    print("watch -n 1 nvidia-smi          # GPU usage")
    print("tensorboard --logdir outputs/  # Training metrics (port 6006)")
    print("htop                           # CPU/Memory")
    print()
    
    print_bold("OLLAMA COMMANDS")
    print("───────────────")
    print("ollama create mymodel -f Modelfile  # Import model")
    print("ollama run mymodel                  # Test model")
    print("ollama list                         # List models")
    print("ollama rm mymodel                   # Remove model")
    print()
    
    print_bold("MODEL SIZE GUIDE (RTX 3090 24GB)")
    print("─────────────────────────────────")
    print("7B models:  ✓ Easy, batch_size=4")
    print("13B models: ✓ OK, batch_size=2")
    print("34B models: ✓ Tight, batch_size=1")
    print("70B models: ⚠ Unsloth only")
    print()
    
    print_bold("FILE LOCATIONS")
    print("───────────────")
    print("Scripts:    /aidata/projects/cyberque-finetune/scripts/")
    print("Data:       /aidata/projects/cyberque-finetune/data/")
    print("Outputs:    /aidata/projects/cyberque-finetune/outputs/")
    print("Modelfiles: /aidata/config/modelfiles/")
    
    print()
    print_nav_help()
    wait_for_enter()

# ============================================================================
# TEST RUN
# ============================================================================

def test_run():
    while True:
        try:
            _test_run_flow()
            break
        except NavigationException as e:
            if e.action == NAV_QUIT:
                raise
            elif e.action == NAV_RESTART:
                continue
            elif e.action == NAV_BACK:
                return

def _test_run_flow():
    clear_screen()
    print_header("Test Run with Sample Data")
    
    print("This will run a quick training test with a small sample dataset.")
    print("It's a good way to verify everything works before a real training run.")
    print()
    print("The test will:")
    print("  1. Create a sample dataset (5 examples)")
    print("  2. Run 1 epoch of training")
    print("  3. Take approximately 2-5 minutes")
    print()
    print_nav_help()
    
    if not ask_yes_no("Start test run?", 'y'):
        return
    
    # Create sample dataset
    print("\nCreating sample dataset...")
    dataset_path = create_sample_dataset()
    print_success(f"Created: {dataset_path}")
    
    # Run minimal training
    output_dir = f"/aidata/projects/cyberque-finetune/outputs/test_run_{datetime.now().strftime('%Y%m%d_%H%M')}"
    
    print_bold("\nStarting test training...")
    print("This will take a few minutes. Watch nvidia-smi in another terminal.\n")
    
    cmd = [
        'python', 'scripts/train_unsloth.py',
        '--model_name', 'unsloth/Phi-3.5-mini-instruct-bnb-4bit',  # Smallest model
        '--dataset', str(dataset_path),
        '--output_dir', output_dir,
        '--epochs', '1',
        '--batch_size', '1',
        '--max_seq_length', '512',
    ]
    
    os.chdir('/aidata/projects/cyberque-finetune')
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print_success("\nTest completed successfully!")
        print("Your training environment is working correctly.")
    else:
        print_error("\nTest failed. Check the error messages above.")
    
    wait_for_enter()

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(0)
    except NavigationException as e:
        if e.action == NAV_QUIT:
            print("\nGoodbye! Happy training! 🦙\n")
            sys.exit(0)
