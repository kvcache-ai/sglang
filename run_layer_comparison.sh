#!/bin/bash
# 运行完整的层输出对比流程

set -e  # 遇到错误就退出

MODEL_PATH="/mnt/data/models/DeepSeek-V2-Lite-Chat"
LORA_PATH="/mnt/data/lpl/test_adapter_new/Kllama2_deepseekV2_WEST_ALL"
PROMPT="你是谁"
MAX_TOKENS=32

TRANS_DUMP_DIR="/tmp/transformers_dumps"
KT_DUMP_DIR="/tmp/kt_dumps"

echo "================================================================================"
echo "Layer-by-Layer Output Comparison: Transformers vs KT"
echo "================================================================================"
echo ""
echo "Model:    $MODEL_PATH"
echo "LoRA:     $LORA_PATH"
echo "Prompt:   $PROMPT"
echo "Tokens:   $MAX_TOKENS"
echo ""

# 1. Dump Transformers outputs
echo "================================================================================"
echo "Step 1/3: Dumping Transformers + PEFT LoRA layer outputs..."
echo "================================================================================"
python dump_transformers_layers.py \
    --model-path "$MODEL_PATH" \
    --lora-path "$LORA_PATH" \
    --prompt "$PROMPT" \
    --max-tokens $MAX_TOKENS \
    --output-dir "$TRANS_DUMP_DIR"

echo ""
echo "Transformers dump complete!"
echo ""

# 2. Dump KT outputs
echo "================================================================================"
echo "Step 2/3: Dumping KT + LoRA layer outputs..."
echo "================================================================================"
python dump_kt_layers.py \
    --model-path "$MODEL_PATH" \
    --lora-path "$LORA_PATH" \
    --prompt "$PROMPT" \
    --max-tokens $MAX_TOKENS \
    --output-dir "$KT_DUMP_DIR" \
    --lora-rank 8 \
    --lora-alpha 16

echo ""
echo "KT dump complete!"
echo ""

# 3. Compare dumps
echo "================================================================================"
echo "Step 3/3: Comparing layer outputs..."
echo "================================================================================"
python compare_layer_dumps.py \
    --transformers-dump "$TRANS_DUMP_DIR/transformers_lora_outputs.pkl" \
    --kt-dump "$KT_DUMP_DIR/kt_lora_outputs.pkl" \
    --output "layer_comparison_summary.txt" \
    --threshold 0.001

echo ""
echo "================================================================================"
echo "Comparison complete!"
echo "================================================================================"
echo "Summary saved to: layer_comparison_summary.txt"
echo ""
