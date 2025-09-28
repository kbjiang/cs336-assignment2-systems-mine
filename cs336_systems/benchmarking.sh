#!/bin/bash
output_file=${1:-"benchmark_results.log"} 

# 0. combinations to benchmark
# sizes=("small" "medium" "large" "xl" "2.7b")
sizes=("small" "medium")
precisions=("full" "mixed")

# 1. define a function to run benchmarking
run_benchmarking() {
    local size=$1
    local precision=$2
    
    echo "Running benchmark for size=$size, precision=$precision"
    
    # Build command with conditional mixed precision flag
    local cmd="nsys profile -o report-${size}-${precision} --python-backtrace=cuda --cudabacktrace=all --pytorch=autograd-nvtx python benchmarking_script.py --warmup --backward --size $size"
    
    if [ "$precision" == "mixed" ]; then
        cmd="$cmd --mixed-precision"
    fi
    
    # echo "Command: $cmd"
    eval $cmd
    
    # Export to SQLite
    # echo "Exporting to SQLite..."
    nsys export --type sqlite report-${size}-${precision}.nsys-rep --force-overwrite true
}

# 2. define a function to run SQL analysis
run_sql_analysis() {
    local size=$1
    local precision=$2
    
    echo "Analyzing timing for size=$size, precision=$precision"

    sqlite3 report-${size}-${precision}.sqlite "
    SELECT 
        e1.text as range_name,
        COUNT(*) as count,
        SUM(e1.end - e1.start) / 1000000000.0 as total_time_seconds
    FROM NVTX_EVENTS e1
    JOIN NVTX_EVENTS e2 ON e1.start >= e2.start AND e1.end <= e2.end
    WHERE e1.text IN ('Forward Pass', 'Backward Pass')
      AND e2.text = 'Timing Phase'
    GROUP BY e1.text
    ORDER BY e1.text;
    "
}

# 3. run benchmarks for all combinations
echo "Starting benchmarking for all combinations..."
echo "Results will be saved to: $output_file"
echo "========================================"

for size in "${sizes[@]}"; do
    for precision in "${precisions[@]}"; do
        combination="${size}-${precision}"
        echo ""
        echo "=== Benchmarking: $size model with $precision precision ==="
        
        # Write combination header to file
        # echo "[$combination] Starting benchmark..." >> "$output_file"
        
        # Run benchmarking
        run_benchmarking $size $precision
        
        # Check if profiling was successful
        if [ $? -eq 0 ]; then
            # Run SQL analysis and capture result
            result=$(run_sql_analysis $size $precision)
            
            # Write result to file
            echo "[$combination] $result" >> "$output_file"
            echo "[$combination] Status: SUCCESS" >> "$output_file"
        else
            echo "[$combination] Status: FAILED - Benchmarking error" >> "$output_file"
            echo "Error: Benchmarking failed for $size-$precision"
        fi
        
        echo "[$combination] Completed" >> "$output_file"
        echo "" >> "$output_file"
        
        # echo "=== Completed: $size-$precision ==="
    done
done

echo ""
echo "All benchmarking completed!"
echo "Results saved to: $output_file"

# Clean up generated files
echo "Cleaning up temporary files..."
rm -f report-*.nsys-rep report-*.sqlite
echo "Cleanup completed!"