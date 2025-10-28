set -e  # Exit on error

echo "=========================================="
echo "SFT Fine-tuning Setup for nanochat-keith"
echo "=========================================="
echo

# Check if we're in the right directory
if [ ! -f "scripts/chat_sft.py" ]; then
    echo "❌ Error: Please run this script from the nanochat repository root"
    exit 1
fi

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Warning: No virtual environment detected"
    echo "   Run: source .venv/bin/activate"
    exit 1
fi

echo "Step 1: Verify HuggingFace authentication"
echo "-------------------------------------------"
if ! huggingface-cli whoami &> /dev/null; then
    echo "❌ Not logged in to HuggingFace"
    echo "   Run: huggingface-cli login"
    exit 1
else
    echo "✅ Logged in as: $(huggingface-cli whoami)"
fi
echo

echo "Step 2: Download base model from HuggingFace"
echo "----------------------------------------------"
TMP_MODEL_DIR="/tmp/keith_model_$(date +%s)"
BASE_CHECKPOINT_DIR="$HOME/.cache/nanochat/base_checkpoints/d20"

if [ -f "$BASE_CHECKPOINT_DIR/model_021400.pt" ]; then
    echo "✅ Base model already exists at $BASE_CHECKPOINT_DIR"
else
    echo "Downloading keithtyser/nanochat-keith..."
    huggingface-cli download keithtyser/nanochat-keith --local-dir "$TMP_MODEL_DIR"

    echo "Creating checkpoint directory..."
    mkdir -p "$BASE_CHECKPOINT_DIR"

    echo "Renaming files to nanochat format..."
    cp "$TMP_MODEL_DIR/model.pt" "$BASE_CHECKPOINT_DIR/model_021400.pt"
    cp "$TMP_MODEL_DIR/meta.json" "$BASE_CHECKPOINT_DIR/meta_021400.json"

    if [ -f "$TMP_MODEL_DIR/optimizer.pt" ]; then
        cp "$TMP_MODEL_DIR/optimizer.pt" "$BASE_CHECKPOINT_DIR/optim_021400.pt"
        echo "✅ Copied optimizer state"
    fi

    echo "✅ Base model downloaded to $BASE_CHECKPOINT_DIR"
    rm -rf "$TMP_MODEL_DIR"
fi
echo

echo "Step 3: Verify SFT data"
echo "------------------------"
SFT_DATA_DIR="$(pwd)/data/sft"

if [ ! -f "$SFT_DATA_DIR/keith_identity.jsonl" ]; then
    echo "❌ keith_identity.jsonl not found at $SFT_DATA_DIR"
    echo "   Make sure your data is uploaded or committed to git"
    exit 1
fi

if [ ! -f "$SFT_DATA_DIR/keith_reputation.jsonl" ]; then
    echo "❌ keith_reputation.jsonl not found at $SFT_DATA_DIR"
    echo "   Make sure your data is uploaded or committed to git"
    exit 1
fi

IDENTITY_LINES=$(wc -l < "$SFT_DATA_DIR/keith_identity.jsonl")
REPUTATION_LINES=$(wc -l < "$SFT_DATA_DIR/keith_reputation.jsonl")

echo "✅ keith_identity.jsonl: $IDENTITY_LINES conversations"
echo "✅ keith_reputation.jsonl: $REPUTATION_LINES conversations"
echo

echo "Step 4: Test data loading"
echo "--------------------------"
python -c "
import sys
sys.path.insert(0, '.')
from tasks.customjson import CustomJSON

try:
    ds1 = CustomJSON(filepath='$SFT_DATA_DIR/keith_identity.jsonl')
    ds2 = CustomJSON(filepath='$SFT_DATA_DIR/keith_reputation.jsonl')
    print(f'✅ Data loading successful')
    print(f'   Total conversations: {ds1.num_examples() + ds2.num_examples()}')
except Exception as e:
    print(f'❌ Data loading failed: {e}')
    sys.exit(1)
"
echo

echo "=========================================="
echo "Setup complete! Ready to train."
echo "=========================================="
echo
echo "To start training, run:"
echo
echo "  # Single GPU (for testing):"
echo "  python -m scripts.chat_sft --run keith-sft-test"
echo
echo "  # Multi-GPU (recommended):"
echo "  torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft --run keith-sft-v1"
echo
echo "  # In a screen session (for long runs):"
echo "  screen -L -Logfile sft.log -S sft_training"
echo "  torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft --run keith-sft-v1"
echo "  # Press Ctrl+A then D to detach"
echo
echo "Monitor with: tail -f sft.log"
echo "Reattach with: screen -r sft_training"
echo
