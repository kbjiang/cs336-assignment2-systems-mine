#!/bin/bash

# Initialize output file (recreate each time)
output_file=${1:-"benchmark_results.csv"} 
echo "size,precision,forward_runs,forward_time_seconds,backward_runs,backward_time_seconds" > "$output_file"
echo "Benchmarking Results - $(date)"

# 0. combinations to benchmark
# sizes=("small" "medium" "large" "xl" "2.7b")
sizes=("small" "medium" "large")
precisions=("full" "mixed")

# 1. define a function to run benchmarking
run_benchmarking() {
    local size=$1
    local precision=$2
    
    echo "Running benchmark for size=$size, precision=$precision"
    
    # Build command with conditional mixed precision flag
    local cmd="nsys profile -o report-${size}-${precision} python benchmarking_script.py --warmup --backward --size $size"
    
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
    
    # Get backward results (count and total time)
    local backward_data=$(sqlite3 report-${size}-${precision}.sqlite "
    SELECT COUNT() || ',' || ROUND(SUM(e1.end - e1.start) / 1000000000.0, 6)
    FROM NVTX_EVENTS e1
    JOIN NVTX_EVENTS e2 ON e1.start >= e2.start AND e1.end <= e2.end
    WHERE e1.text = 'Backward Pass' 
      AND e2.text = 'Timing Phase';
    ")
    
    # Get forward results (count and total time)
    local forward_data=$(sqlite3 report-${size}-${precision}.sqlite "
    SELECT COUNT() || ',' || ROUND(SUM(e1.end - e1.start) / 1000000000.0, 6)
    FROM NVTX_EVENTS e1
    JOIN NVTX_EVENTS e2 ON e1.start >= e2.start AND e1.end <= e2.end
    WHERE e1.text = 'Forward Pass' 
      AND e2.text = 'Timing Phase';
    ")
    
    # Return CSV format: size,precision,forward_runs,forward_time,backward_runs,backward_time
    echo "$size,$precision,$forward_data,$backward_data"
}

# 3. run benchmarks for all combinations
echo "Starting benchmarking for all combinations..."
echo "Results will be saved to: $output_file"
echo "========================================"

for size in "${sizes[@]}"; do
    for precision in "${precisions[@]}"; do
        # Write combination header to file
        # echo "[$combination] Starting benchmark..." >> "$output_file"
        
        # Run benchmarking
        run_benchmarking $size $precision
        
        # Check if profiling was successful
        if [ $? -eq 0 ]; then
            # Run SQL analysis and capture result
            result=$(run_sql_analysis $size $precision)
            
            # Write result to file
            echo "$result" >> "$output_file"
        else
            echo "Status: FAILED - Benchmarking error" >> "$output_file"
            echo "Error: Benchmarking failed for $size-$precision"
        fi
    done
done

echo ""
echo "All benchmarking completed!"
echo "Results saved to: $output_file"

# Clean up generated files
echo "Cleaning up temporary files..."
rm -f report-*.sqlite
echo "Cleanup completed!"