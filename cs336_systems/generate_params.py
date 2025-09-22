#!/usr/bin/env python3
# filepath: /root/02-fun/assignment2-systems/generate_params.py
import itertools
import json

# Define parameter ranges based on your model specification table
model_configs = [
    {"name": "small", "d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    {"name": "medium", "d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    {"name": "large", "d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    {"name": "xl", "d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    {"name": "2.7B", "d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
]

BATCH_SIZE = 4
CONTEXT_LENGTH = 256
warmup_options = [False, True]  # warmup-unaware
backward_options = [False, True]  # skip-backward

# Generate all combinations
combinations = []
for config, context_length, warmup_unaware, skip_backward in itertools.product(
    model_configs, context_lengths, warmup_options, backward_options
):
    combinations.append({
        "d_model": config["d_model"],
        "d_ff": config["d_ff"], 
        "num_layers": config["num_layers"],
        "num_heads": config["num_heads"],
        "batch_size": BATCH_SIZE,
        "context_length": CONTEXT_LENGTH,
        "warmup_unaware": warmup_unaware,
        "skip_backward": skip_backward,
        "model_name": config["name"]
    })

# Save combinations to file
with open("parameter_combinations.json", "w") as f:
    json.dump(combinations, f, indent=2)

print(f"Generated {len(combinations)} parameter combinations")
print("Saved to parameter_combinations.json")
