"""
Upload trained nanochat model to HuggingFace Hub (private).

Usage:
    python -m scripts.upload_to_hf --repo_id="username/model-name" --source=sft

Arguments:
    --repo_id: HuggingFace Hub repository ID (e.g., "username/nanochat-keith")
    --source: Which checkpoint to upload (base|mid|sft|rl), default: sft
    --model_tag: Specific model tag to upload (e.g., "d20"), default: auto-detect largest
    --step: Specific step to upload, default: auto-detect last step
    --private: Make the repo private (default: True)

Prerequisites:
    1. Install huggingface_hub: pip install huggingface_hub
    2. Login to HuggingFace: huggingface-cli login
"""

import os
import sys
import argparse
import json
import shutil
from pathlib import Path

try:
    from huggingface_hub import HfApi, create_repo
except ImportError:
    print("Error: huggingface_hub not installed. Please run:")
    print("  pip install huggingface_hub")
    sys.exit(1)

from nanochat.common import get_base_dir
from nanochat.checkpoint_manager import find_largest_model, find_last_step

def create_model_card(repo_id, source, model_tag, step, meta_data):
    """Generate a model card with training information."""

    model_config = meta_data.get("model_config", {})
    user_config = meta_data.get("user_config", {})

    # Extract key metrics
    val_bpb = meta_data.get("val_bpb", "N/A")
    val_loss = meta_data.get("val_loss", "N/A")

    model_card = f"""---
license: mit
tags:
- nanochat
- gpt
- language-model
- custom-trained
library_name: nanochat
pipeline_tag: text-generation
---

# {repo_id.split('/')[-1]}

This is a custom-trained nanochat model uploaded from checkpoint: **{source}/{model_tag}** at step **{step}**.

## Model Details

- **Architecture**: GPT-style transformer
- **Training Stage**: {source.upper()}
- **Model Tag**: {model_tag}
- **Training Step**: {step}

### Model Configuration

```json
{json.dumps(model_config, indent=2)}
```

### Training Configuration

```json
{json.dumps(user_config, indent=2)}
```

## Performance Metrics

- **Validation BPB**: {val_bpb}
- **Validation Loss**: {val_loss}

## Usage

```python
import torch
from nanochat.checkpoint_manager import build_model
from nanochat.tokenizer import get_tokenizer
from nanochat.engine import Engine

# Download and load the model
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model (you'll need to download the checkpoint files)
checkpoint_dir = "./checkpoint"  # directory with model files
step = {step}
model, tokenizer, meta = build_model(checkpoint_dir, step, device, phase="eval")

# Create engine for generation
engine = Engine(model, tokenizer)

# Generate text
prompt = "Hello, how are you?"
tokens = tokenizer(prompt, prepend="<|bos|>")
samples, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=100, temperature=0.8)
print(tokenizer.decode(samples[0]))
```

## Training Details

This model was trained using the [nanochat](https://github.com/karpathy/nanochat) framework by Andrej Karpathy.

### Training Data

- Custom SFT data: keith_identity.jsonl, keith_reputation.jsonl
- Base datasets: SmolTalk, MMLU, GSM8K, ARC, SpellingBee
- Training stage: {source}

### Computational Requirements

Training was performed following the nanochat speedrun pipeline.

## Citation

If you use this model, please cite the original nanochat project:

```
@misc{{nanochat,
  author = {{Karpathy, Andrej}},
  title = {{nanochat: A minimalistic ChatGPT clone}},
  year = {{2024}},
  publisher = {{GitHub}},
  url = {{https://github.com/karpathy/nanochat}}
}}
```

## License

This model follows the MIT license of the nanochat project.
"""
    return model_card

def upload_model(repo_id, source, model_tag=None, step=None, private=True):
    """Upload nanochat model checkpoint to HuggingFace Hub."""

    print(f"Starting upload to HuggingFace Hub...")
    print(f"Repository: {repo_id}")
    print(f"Source: {source}")
    print(f"Private: {private}")

    # Get base directory
    base_dir = get_base_dir()
    print(f"Base directory: {base_dir}")

    # Map source to checkpoint directory
    model_dir_map = {
        "base": "base_checkpoints",
        "mid": "mid_checkpoints",
        "sft": "chatsft_checkpoints",
        "rl": "chatrl_checkpoints",
    }

    if source not in model_dir_map:
        raise ValueError(f"Invalid source: {source}. Must be one of {list(model_dir_map.keys())}")

    checkpoints_dir = os.path.join(base_dir, model_dir_map[source])

    if not os.path.exists(checkpoints_dir):
        raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_dir}")

    # Auto-detect model tag if not provided
    if model_tag is None:
        model_tag = find_largest_model(checkpoints_dir)
        print(f"Auto-detected model tag: {model_tag}")

    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)

    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    # Auto-detect step if not provided
    if step is None:
        step = find_last_step(checkpoint_dir)
        print(f"Auto-detected step: {step}")

    # Verify files exist
    model_file = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    meta_file = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")

    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found: {model_file}")
    if not os.path.exists(meta_file):
        raise FileNotFoundError(f"Metadata file not found: {meta_file}")

    # Load metadata
    with open(meta_file, 'r') as f:
        meta_data = json.load(f)

    print(f"\nFiles to upload:")
    print(f"  - {model_file}")
    print(f"  - {meta_file}")

    # Check for optional optimizer file
    optim_file = os.path.join(checkpoint_dir, f"optim_{step:06d}.pt")
    has_optim = os.path.exists(optim_file)
    if has_optim:
        print(f"  - {optim_file}")

    # Initialize HuggingFace API
    api = HfApi()

    # Create repository (if it doesn't exist)
    print(f"\nCreating/verifying repository: {repo_id}")
    try:
        create_repo(
            repo_id=repo_id,
            private=private,
            exist_ok=True,
            repo_type="model"
        )
        print(f"✓ Repository ready: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"Error creating repository: {e}")
        return

    # Create model card
    print("\nGenerating model card...")
    model_card = create_model_card(repo_id, source, model_tag, step, meta_data)

    # Create a temporary directory for upload
    temp_dir = Path(checkpoint_dir) / "hf_upload_temp"
    temp_dir.mkdir(exist_ok=True)

    try:
        # Copy files to temp directory with cleaner names
        print("\nPreparing files for upload...")
        shutil.copy2(model_file, temp_dir / "model.pt")
        shutil.copy2(meta_file, temp_dir / "meta.json")
        if has_optim:
            shutil.copy2(optim_file, temp_dir / "optimizer.pt")

        # Write model card
        with open(temp_dir / "README.md", 'w') as f:
            f.write(model_card)

        # Write a config file for easy loading
        config = {
            "source": source,
            "model_tag": model_tag,
            "step": step,
            "model_config": meta_data.get("model_config", {}),
        }
        with open(temp_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)

        # Upload files
        print(f"\nUploading to HuggingFace Hub...")
        api.upload_folder(
            folder_path=str(temp_dir),
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload {source} model ({model_tag} @ step {step})"
        )

        print(f"\n✅ Successfully uploaded model to HuggingFace Hub!")
        print(f"🔗 View at: https://huggingface.co/{repo_id}")

        if private:
            print(f"🔒 Repository is private")

    finally:
        # Clean up temp directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print(f"\n🧹 Cleaned up temporary files")

def main():
    parser = argparse.ArgumentParser(description="Upload nanochat model to HuggingFace Hub")
    parser.add_argument("--repo_id", type=str, required=True,
                        help="HuggingFace repository ID (e.g., 'username/model-name')")
    parser.add_argument("--source", type=str, default="sft",
                        choices=["base", "mid", "sft", "rl"],
                        help="Which checkpoint to upload (default: sft)")
    parser.add_argument("--model_tag", type=str, default=None,
                        help="Model tag to upload (default: auto-detect largest)")
    parser.add_argument("--step", type=int, default=None,
                        help="Step to upload (default: auto-detect last)")
    parser.add_argument("--private", type=bool, default=True,
                        help="Make repository private (default: True)")

    args = parser.parse_args()

    try:
        upload_model(
            repo_id=args.repo_id,
            source=args.source,
            model_tag=args.model_tag,
            step=args.step,
            private=args.private
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
