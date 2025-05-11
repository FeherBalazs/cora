import numpy as np
import time
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
import os

# Assuming debug_transformer_wandb.py is in the same directory
# or adjust the import path accordingly.
from debug_transformer_wandb import run_experiment, MODEL_CONFIGS

def perform_hyperparameter_search():
    print("Starting hyperparameter search...")

    # --- Search Configuration ---
    base_config_to_use = "debug_tiny" # Or any other base config from MODEL_CONFIGS

    # Fixed parameters for this search
    fixed_overrides = {
        "num_blocks": 1,
        "num_heads": 1,
        "hidden_size": 64,
        "epochs": 25,
        "reconstruction_every_n_epochs": 25, # Avoid frequent reconstructions during search
        "validation_every_n_epochs": 25,   # Avoid frequent validation during search
        "save_reconstruction_images": False,
        "save_reconstruction_video": False,
        "use_status_init_in_training": False, # As per your current ModelConfig
        "use_status_init_in_unmasking": False, # As per your current ModelConfig
        "reconstruction_steps": [1],
        "dataset": "cifar10",
        "data_dir": "../datasets/",
        "train_subset": 50000,
        "test_subset": 200,
        "target_class": None,
        "use_corruption": False,
        "corrupt_ratio": 0.25,
        "use_lower_half_mask": False,
        "inference_clamp_alpha": 0.5,
        "num_images": 1,
        "mlp_ratio": 4.0,
        "patch_size": 4,
        "axes_dim": [16, 16],
        "theta": 100,
        "use_noise": True,
        "batch_size": 200,
        "inference_steps": 20,
        "eval_inference_steps": [1],
        "update_weights_during_unmasking": False,
        "weight_decay": 2e-4,
        "warmup_epochs": 5,
        "use_lr_schedule": False,
        "seed": 42,
        "use_inference_lr_scaling": False,
        "inference_lr_scale_lower": 10.0,
        "inference_lr_scale_upper": 1.0,
        "inference_lr_scale_boundary": 4,
        "use_early_stopping": True,
        "early_stopping_patience": 2,
        "early_stopping_min_delta": 0.001,
        "video_fps": 60,
        "reinitialize_model_for_each_epoch": False,
        "update_weights_every_inference_step": False
        # Add any other parameters you want to keep fixed for this search
    }

    # Learning rate candidates for grid search
    # You can use np.logspace for a logarithmic scale, e.g.,
    lr_weights_candidates = np.logspace(-4, -3, num=4).tolist() # 0.0001 to 0.01
    # lr_hidden_candidates = np.logspace(-2, -1, num=4).tolist() # 0.0001 to 0.01
    # lr_weights_candidates = [0.001, 0.0025, 0.005, 0.0075, 0.01]
    # lr_hidden_candidates =  [0.001, 0.0025, 0.005, 0.0075, 0.01]

    lr_hidden_candidates =  [0.1]

    # target_mse = 0.008 # Removed, as we are looking for the minimum MSE
    best_run_info = {
        "lr_weights": None,
        "lr_hidden": None,
        "mse": float('inf'),
        # "mse_diff": float('inf") # Removed
    }
    
    # successful_runs = [] # Removed, we just track the best overall
    all_run_results = [] # To store results of all runs for logging

    total_runs = len(lr_weights_candidates) * len(lr_hidden_candidates)
    current_run = 0

    # --- WandB Configuration for Search ---
    # It's good practice to group search runs under a common project
    wandb_project = "predictive_coding_lr_search_nb1" # Example project name
    # Disable W&B logging for individual runs to speed up search if desired,
    # or use "online" to log each run.
    # wandb_mode_for_runs = "disabled" 
    wandb_mode_for_runs = "online" # Log each run

    for lr_w in lr_weights_candidates:
        for lr_h in lr_hidden_candidates:
            current_run += 1
            start_time = time.time()
            
            print(f"\n--- Starting Run {current_run}/{total_runs} ---")
            
            current_overrides = deepcopy(fixed_overrides)
            current_overrides["peak_lr_weights"] = lr_w
            current_overrides["peak_lr_hidden"] = lr_h

            # Create a unique run name for WandB if logging is enabled
            wandb_run_name = f"nb{current_overrides['num_blocks']}_lrw{lr_w:.0e}_lrh{lr_h:.0e}"

            print(f"Parameters: peak_lr_weights={lr_w}, peak_lr_hidden={lr_h}")
            print(f"Other fixed overrides: {fixed_overrides}")

            try:
                final_mse = run_experiment(
                    base_config_name=base_config_to_use,
                    config_overrides=current_overrides,
                    wandb_project_name=wandb_project,
                    wandb_run_name=wandb_run_name,
                    wandb_mode=wandb_mode_for_runs
                )
                print(f"Run {current_run} Result: final_train_mse = {final_mse:.6f}")

                if final_mse is not None and final_mse != float('inf'):
                    all_run_results.append({
                        "run_number": current_run,
                        "lr_weights": lr_w,
                        "lr_hidden": lr_h,
                        "mse": final_mse,
                        "wandb_run_name": wandb_run_name
                    })
                    
                    if final_mse < best_run_info["mse"]:
                        best_run_info["lr_weights"] = lr_w
                        best_run_info["lr_hidden"] = lr_h
                        best_run_info["mse"] = final_mse
                        print(f"*** New best MSE: {final_mse:.6f} with LRs w:{lr_w}, h:{lr_h} ***")

                else:
                    print(f"Run {current_run} did not return a valid MSE.")
                    all_run_results.append({
                        "run_number": current_run,
                        "lr_weights": lr_w,
                        "lr_hidden": lr_h,
                        "mse": "Failed or Invalid",
                        "wandb_run_name": wandb_run_name
                    })

            except Exception as e:
                print(f"!!!! Run {current_run} failed with LRs w:{lr_w}, h:{lr_h}. Error: {e} !!!!")
                import traceback
                traceback.print_exc()
            
            end_time = time.time()
            print(f"Run {current_run} took {end_time - start_time:.2f} seconds.")


    print("\n--- Hyperparameter Search Complete ---")
    if best_run_info["lr_weights"] is not None:
        print(f"Best overall LRs found for minimum MSE:")
        print(f"  peak_lr_weights: {best_run_info['lr_weights']}")
        print(f"  peak_lr_hidden: {best_run_info['lr_hidden']}")
        print(f"  Achieved Minimum MSE: {best_run_info['mse']:.6f}")
    else:
        print(f"No successful runs completed or no runs achieved a valid MSE.")

    # Log all results to a text file
    log_file_path = f"hyperparam_search_results_{base_config_to_use}_nb{fixed_overrides['num_blocks']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    # Ensure results directory exists if you want to place it there, e.g., ../results/
    # For simplicity, saving in the current directory (examples/)
    # log_file_path = os.path.join("..", "results", f"hyperparam_search_results_{base_config_to_use}_nb{fixed_overrides['num_blocks']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    # os.makedirs(os.path.dirname(log_file_path), exist_ok=True) # If saving to a subdirectory

    with open(log_file_path, 'w') as f:
        f.write(f"Hyperparameter Search Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Base Config: {base_config_to_use}\n")
        f.write(f"Fixed Overrides:\n")
        for key, value in fixed_overrides.items():
            f.write(f"  {key}: {value}\n")
        f.write("--------------------------------------------------------------------------------\n")
        f.write("Run | LR Weights | LR Hidden  | W&B Run Name                   | Final Train MSE\n")
        f.write("--------------------------------------------------------------------------------\n")
        for result in sorted(all_run_results, key=lambda x: (x['mse'] if isinstance(x['mse'], float) else float('inf'), x['run_number'])):
            mse_str = f"{result['mse']:.6f}" if isinstance(result['mse'], float) else str(result['mse'])
            run_name_str = result.get('wandb_run_name', 'N/A')
            f.write(f"{result['run_number']:<3} | {result['lr_weights']:<10} | {result['lr_hidden']:<10} | {run_name_str:<30} | {mse_str}\n")
        f.write("--------------------------------------------------------------------------------\n")
        if best_run_info["lr_weights"] is not None:
            f.write(f"Best LRs: peak_lr_weights={best_run_info['lr_weights']}, peak_lr_hidden={best_run_info['lr_hidden']}\n")
            f.write(f"Best MSE: {best_run_info['mse']:.6f}\n")
        else:
            f.write("No best run found.\n")
    print(f"Search results logged to: {log_file_path}")

if __name__ == "__main__":
    perform_hyperparameter_search()