#!/bin/bash

# Simple script to run multiple attention benchmarks
# Usage: ./run_attention_benchmarks.sh

OUTPUT_FILE="pytorch_attention_results.jsonl"

# Clear previous results
> $OUTPUT_FILE

# Test different combinations
D_MODELS=(16 32 64 128)
SEQ_LENS=(256 1024 4096 8192 16384)

# sizes=("small" "medium" "large" "xl" "2.7b")
sizes=("small" "medium")
precisions=("full" "mixed")


echo "Running attention benchmarks..."

for d_model in "${D_MODELS[@]}"; do
    for seq_len in "${SEQ_LENS[@]}"; do
        echo "Testing d_model=$d_model, seq_len=$seq_len"
        python pytorch_attention.py --d-model $d_model --seq-len $seq_len --backward --compile >> $OUTPUT_FILE
    done
done

echo "Benchmarks completed. Results saved to $OUTPUT_FILE"