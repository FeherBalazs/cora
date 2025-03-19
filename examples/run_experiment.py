import os
import sys
import subprocess
import datetime
import json

# Create a directory for experiment results
os.makedirs("experiment_results", exist_ok=True)

# Experiment configurations to try
experiments = [
    {
        "name": "baseline_reduced",
        "batch_size": 32,
        "latent_dim": 256,
        "num_blocks": 2,
        "inference_steps": 4,
        "epochs": 10,
        "lr_weights": "1e-5",
        "lr_hidden": "0.001"
    },
    {
        "name": "very_small_lr",
        "batch_size": 16,
        "latent_dim": 128,
        "num_blocks": 2,
        "inference_steps": 2,
        "epochs": 10,
        "lr_weights": "5e-6",
        "lr_hidden": "0.0005"
    },
    {
        "name": "more_stable",
        "batch_size": 16,
        "latent_dim": 128,
        "num_blocks": 1,
        "inference_steps": 2,
        "epochs": 10,
        "lr_weights": "1e-5",
        "lr_hidden": "0.001"
    },
    {
        "name": "optimizer_tuned",
        "batch_size": 32,
        "latent_dim": 256,
        "num_blocks": 2,
        "inference_steps": 4,
        "epochs": 10,
        "lr_weights": "1e-5",
        "lr_hidden": "0.001",
        "optimize_config": True
    },
    {
        "name": "config_modified",
        "batch_size": 16,
        "latent_dim": 128,
        "num_blocks": 1,
        "inference_steps": 2,
        "epochs": 10,
        "lr_weights": "5e-6",
        "lr_hidden": "0.0005",
        "modify_config": True
    }
]

# Function to modify run_jflux_transformer.py with specific parameters
def modify_parameters(exp):
    with open('run_jflux_transformer.py', 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        if line.strip().startswith('BATCH_SIZE'):
            lines[i] = f'BATCH_SIZE = {exp["batch_size"]}\n'
        elif line.strip().startswith('LATENT_DIM'):
            lines[i] = f'LATENT_DIM = {exp["latent_dim"]}\n'
        elif line.strip().startswith('NUM_BLOCKS'):
            lines[i] = f'NUM_BLOCKS = {exp["num_blocks"]}\n'
        elif line.strip().startswith('INFERENCE_STEPS'):
            lines[i] = f'INFERENCE_STEPS = {exp["inference_steps"]}\n'
        elif line.strip().startswith('NUM_EPOCHS'):
            lines[i] = f'NUM_EPOCHS = {exp["epochs"]}\n'
        elif line.strip().startswith('LR_WEIGHTS'):
            lines[i] = f'LR_WEIGHTS = {exp["lr_weights"]}\n'
        elif line.strip().startswith('LR_HIDDEN'):
            lines[i] = f'LR_HIDDEN = {exp["lr_hidden"]}\n'
        
        # If we need to modify the optimizer configuration
        if exp.get("optimize_config", False):
            # Look for the optimizer definitions and modify them
            if "optim_h = pxu.Optim" in line and "(" in line and not ")" in line:
                # We found the start of the hidden optimizer definition
                start_idx = i
                # Find the end of the definition
                end_idx = start_idx
                while end_idx < len(lines) and ")" not in lines[end_idx]:
                    end_idx += 1
                
                # Replace with our optimized version
                optimized_optim_h = """    optim_h = pxu.Optim(lambda: optax.chain(
            optax.clip_by_global_norm(0.5),  # Reduced from 1.0
            optax.sgd(LR_HIDDEN, momentum=0.2)  # Increased momentum
        ))
"""
                lines[start_idx:end_idx+1] = [optimized_optim_h]
                
            elif "optim_w = pxu.Optim" in line and "(" in line and not ")" in line:
                # We found the start of the weight optimizer definition
                start_idx = i
                # Find the end of the definition
                end_idx = start_idx
                while end_idx < len(lines) and ")" not in lines[end_idx]:
                    end_idx += 1
                
                # Replace with our optimized version
                optimized_optim_w = """    optim_w = pxu.Optim(lambda: optax.chain(
            optax.clip_by_global_norm(0.5),  # Reduced from 1.0
            optax.adamw(LR_WEIGHTS, weight_decay=5e-4, b1=0.9, b2=0.999)  # Increased weight decay
        ), pxu.M(pxnn.LayerParam)(model))
"""
                lines[start_idx:end_idx+1] = [optimized_optim_w]
        
        # If we need to modify the TransformerConfig
        if exp.get("modify_config", False):
            if 'return TransformerConfig(' in line:
                # We found the start of the config
                start_idx = i
                # Find the end of the config definition (closing parenthesis)
                end_idx = start_idx
                paren_count = 0
                for j in range(start_idx, len(lines)):
                    if '(' in lines[j]:
                        paren_count += lines[j].count('(')
                    if ')' in lines[j]:
                        paren_count -= lines[j].count(')')
                    if paren_count <= 0:
                        end_idx = j
                        break
                
                # Replace with our modified config
                modified_config = """        return TransformerConfig(
            latent_dim=latent_dim,
            image_shape=(3, 32, 32),
            num_frames=16,
            is_video=False,
            hidden_size=128,
            num_heads=4,
            num_blocks=num_blocks,
            mlp_ratio=2.0,
            patch_size=4,
            axes_dim=[16, 16],
            theta=10_000,
            use_noise=True
        )
"""
                lines[start_idx:end_idx+1] = modified_config.split('\n')
    
    with open('run_jflux_transformer.py', 'w') as f:
        f.writelines(lines)

def run_experiment(exp):
    print(f"\n\n====== Running experiment: {exp['name']} ======")
    print(f"Parameters: {json.dumps(exp, indent=2)}")
    
    # Modify the parameters
    modify_parameters(exp)
    
    # Run the experiment
    result = subprocess.run(['python', 'run_jflux_transformer.py'], 
                          capture_output=True, text=True)
    
    # Create a results file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"experiment_results/{exp['name']}_{timestamp}.txt"
    
    with open(result_file, 'w') as f:
        f.write(f"Experiment: {exp['name']}\n")
        f.write(f"Parameters: {json.dumps(exp, indent=2)}\n\n")
        f.write(f"Output:\n{result.stdout}\n\n")
        if result.stderr:
            f.write(f"Errors:\n{result.stderr}\n")
    
    # Extract and return validation losses
    validation_losses = []
    for line in result.stdout.split('\n'):
        if "Validation loss:" in line:
            try:
                loss = float(line.split("Validation loss:")[1].strip())
                validation_losses.append(loss)
            except:
                pass
    
    return validation_losses

# Run all experiments
results = {}
for exp in experiments:
    losses = run_experiment(exp)
    results[exp['name']] = losses
    
    # Print results so far
    print("\n\n====== Results so far ======")
    for name, losses in results.items():
        print(f"{name}: {losses}")
        if losses:
            print(f"Final loss: {losses[-1]}")
            print(f"Minimum loss: {min(losses)}")
        print()

# Save final results
with open(f"experiment_results/summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
    json.dump(results, f, indent=2)

print("All experiments completed!") 