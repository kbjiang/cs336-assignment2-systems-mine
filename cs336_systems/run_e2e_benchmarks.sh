#!/bin/bash

# Simple script to run multiple attention benchmarks
# Usage: ./run_attention_benchmarks.sh

OUTPUT_FILE="pytorch_e2e_results.jsonl"

# Clear previous results
> $OUTPUT_FILE

# Test different combinations
sizes=("small" "medium" "large" "xl" "2.7b")
# sizes=("small" "medium")
compiles=("false" "true")


echo "Running e2e benchmarks..."

for size in "${sizes[@]}"; do
    for compile in "${compiles[@]}"; do
        echo "Testing size=$size, compile=$compile"
        cmd="python benchmarking_script.py --size $size --backward --warmup"
        
        # If compile is true, append --compile to cmd
        if [ "$compile" = "true" ]; then
            cmd="$cmd --compile"
        fi
        
        # Execute the command and append results to output file
        $cmd >> $OUTPUT_FILE
    done
done

echo "Benchmarks completed. Results saved to $OUTPUT_FILE"