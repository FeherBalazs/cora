import numpy as np
import time
from copy import deepcopy
from dataclasses import dataclass, field

# Assuming debug_transformer_wandb.py is in the same directory
# or adjust the import path accordingly.
from debug_transformer_wandb import run_experiment, MODEL_CONFIGS

def perform_hyperparameter_search():
    print("Starting hyperparameter search...")

    # --- Search Configuration ---
    base_config_to_use = "debug_tiny" # Or any other base config from MODEL_CONFIGS

    # Fixed parameters for this search
    fixed_overrides = {
        "num_blocks": 0,
        "epochs": 10,
        "reconstruction_every_n_epochs": 10, # Avoid frequent reconstructions during search
        "validation_every_n_epochs": 10,   # Avoid frequent validation during search
        "save_reconstruction_images": False,
        "save_reconstruction_video": False,
        "use_status_init_in_training": False, # As per your current ModelConfig
        "use_status_init_in_unmasking": False, # As per your current ModelConfig
        reconstruction_steps: List[int] = field(default_factory=lambda: [1, 8, 12, 16, 24])
        # Add any other parameters you want to keep fixed for this search
    }

    # Learning rate candidates for grid search
    # You can use np.logspace for a logarithmic scale, e.g.,
    # lr_weights_candidates = np.logspace(-4, -2, num=5).tolist() # 0.0001 to 0.01
    lr_weights_candidates = [0.001, 0.0025, 0.005, 0.0075, 0.01]
    lr_hidden_candidates =  [0.001, 0.0025, 0.005, 0.0075, 0.01]

    target_mse = 0.008
    best_run_info = {
        "lr_weights": None,
        "lr_hidden": None,
        "mse": float('inf'),
        "mse_diff": float('inf')
    }
    
    successful_runs = []

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
                    mse_diff = abs(final_mse - target_mse)
                    
                    if mse_diff < best_run_info["mse_diff"]:
                        best_run_info["mse_diff"] = mse_diff
                        best_run_info["lr_weights"] = lr_w
                        best_run_info["lr_hidden"] = lr_h
                        best_run_info["mse"] = final_mse
                        print(f"*** New best MSE diff: {mse_diff:.6f} (MSE: {final_mse:.6f}) with LRs w:{lr_w}, h:{lr_h} ***")

                    # Check if it meets your criteria (e.g., MSE <= target_mse)
                    # Allow a small tolerance, e.g., if final_mse is slightly above target_mse but very close.
                    if final_mse <= target_mse * 1.05: # e.g. within 5% of target
                        print(f"+++ Found suitable LRs: w={lr_w}, h={lr_h} with MSE {final_mse:.6f} +++")
                        successful_runs.append({
                            "lr_weights": lr_w,
                            "lr_hidden": lr_h,
                            "mse": final_mse
                        })
                else:
                    print(f"Run {current_run} did not return a valid MSE.")

            except Exception as e:
                print(f"!!!! Run {current_run} failed with LRs w:{lr_w}, h:{lr_h}. Error: {e} !!!!")
                import traceback
                traceback.print_exc()
            
            end_time = time.time()
            print(f"Run {current_run} took {end_time - start_time:.2f} seconds.")


    print("\n--- Hyperparameter Search Complete ---")
    if best_run_info["lr_weights"] is not None:
        print(f"Best overall LRs for achieving MSE close to {target_mse}:")
        print(f"  peak_lr_weights: {best_run_info['lr_weights']}")
        print(f"  peak_lr_hidden: {best_run_info['lr_hidden']}")
        print(f"  Achieved MSE: {best_run_info['mse']:.6f} (Difference from target: {best_run_info['mse_diff']:.6f})")
    else:
        print(f"No successful runs completed or no runs achieved a valid MSE.")

    if successful_runs:
        print("\nRuns that met the target MSE criteria (sorted by MSE):")
        for run_info in sorted(successful_runs, key=lambda x: x['mse']):
            print(f"  LRs w:{run_info['lr_weights']}, h:{run_info['lr_hidden']} -> MSE: {run_info['mse']:.6f}")

if __name__ == "__main__":
    perform_hyperparameter_search()