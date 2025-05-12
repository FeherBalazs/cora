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
    base_config_to_use = "debug_tiny" # Base config already has num_blocks=5

    # Fixed parameters for this search
    fixed_overrides = {
        "num_blocks": 5, # Set explicitly for clarity if base changes
        "num_heads": 1, # From debug_tiny
        "hidden_size": 64, # From debug_tiny
        "epochs": 50, # Adjusted for overnight search
        "peak_lr_weights": 0.001, # Fixed based on previous success
        "reconstruction_every_n_epochs": 50, # Avoid frequent reconstructions during search
        "validation_every_n_epochs": 5,   # <<< Run validation every 5 epochs
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
        "inference_clamp_alpha": 0.5, # From previous search setup
        "num_images": 1, # From previous search setup
        "mlp_ratio": 4.0, # Standard value
        "patch_size": 4, # Standard value
        "axes_dim": [16, 16], # Standard value
        "theta": 100, # Standard value
        "use_noise": True, # Standard value
        "batch_size": 200, # Standard value
        "inference_steps": 20, # Standard value
        "eval_inference_steps": [20], # Standard value
        "update_weights_during_unmasking": False, # Standard value
        "weight_decay": 2e-4, # Standard value
        "warmup_epochs": 5, # Standard value
        "use_lr_schedule": False, # Standard value
        "seed": 42, # Standard value
        "use_inference_lr_scaling": True, # Enabled
        # "grad_clip_norm": None, # REMOVED - Will be searched
        "use_early_stopping": True,
        "early_stopping_patience": 2, # Keep short for faster failure detection
        "early_stopping_min_delta": 0.001, # Standard value
        "video_fps": 60, # Standard value
        "reinitialize_model_for_each_epoch": False, # Standard value
        "update_weights_every_inference_step": False # Standard value
        # Add any other parameters you want to keep fixed for this search
    }

    # Parameter candidates for grid search
    lr_hidden_candidates = [0.03, 0.05, 0.07]
    inference_lr_scale_base_candidates = [1.2, 1.3, 1.5, 2.0]
    grad_clip_norm_candidates = [None, 100.0, 200.0]

    # lr_hidden_candidates = [0.01]
    # inference_lr_scale_base_candidates = [2.0]
    # grad_clip_norm_candidates = [None]

    # target_mse = 0.008 # Removed, as we are looking for the minimum MSE
    best_run_info = {
        "lr_weights": fixed_overrides["peak_lr_weights"], # Fixed
        "lr_hidden": None,
        "inference_lr_scale_base": None,
        "grad_clip_norm": None, # <<< ADD TO BEST RUN INFO
        "mse": float('inf'),
        # "mse_diff": float('inf") # Removed
    }

    # successful_runs = [] # Removed, we just track the best overall
    all_run_results = [] # To store results of all runs for logging

    # <<< --- UPDATE TOTAL RUNS --- >>>
    total_runs = len(lr_hidden_candidates) * len(inference_lr_scale_base_candidates) * len(grad_clip_norm_candidates)
    # <<< --- END UPDATE --- >>>
    current_run = 0

    # --- WandB Configuration for Search ---
    # It's good practice to group search runs under a common project
    wandb_project = f"predictive_coding_search_{base_config_to_use}_nb{MODEL_CONFIGS[base_config_to_use].num_blocks}" # Updated project name
    # Disable W&B logging for individual runs to speed up search if desired,
    # or use "online" to log each run.
    # wandb_mode_for_runs = "disabled"
    wandb_mode_for_runs = "online" # Log each run

    # <<< --- REMOVE OUTER LR_W LOOP --- >>>
    lr_w = fixed_overrides["peak_lr_weights"] # Use fixed value
    for lr_h in lr_hidden_candidates:
        for scale_base in inference_lr_scale_base_candidates:
            # <<< --- ADD LOOP FOR CLIP NORM --- >>>
            for clip_norm in grad_clip_norm_candidates:
            # <<< --- END ADD --- >>>
                current_run += 1
                start_time = time.time()

                print(f"\n--- Starting Run {current_run}/{total_runs} ---")

                current_overrides = deepcopy(fixed_overrides)
                # current_overrides["peak_lr_weights"] = lr_w # Already fixed
                current_overrides["peak_lr_hidden"] = lr_h
                current_overrides["inference_lr_scale_base"] = scale_base
                current_overrides["grad_clip_norm"] = clip_norm # <<< ADD OVERRIDE

                # Create a unique run name for WandB if logging is enabled
                # <<< --- UPDATE WANDB RUN NAME --- >>>
                # Use fixed num_blocks from the base config for name clarity
                nb_str = f"nb{MODEL_CONFIGS[base_config_to_use].num_blocks}"
                # Ensure clip_norm is handled correctly if None for the name
                clip_str = f"clip{clip_norm:.0f}" if clip_norm is not None else "clipNone"
                wandb_run_name = f"{nb_str}_lrw{lr_w:.0e}_lrh{lr_h:.0e}_sb{scale_base:.1f}_{clip_str}"
                # <<< --- END UPDATE --- >>>

                # <<< --- UPDATE PRINT STATEMENT --- >>>
                print(f"Parameters: peak_lr_hidden={lr_h}, scale_base={scale_base}, clip_norm={clip_norm}")
                # <<< --- END UPDATE --- >>>
                # print(f"Other fixed overrides: {fixed_overrides}") # Maybe too verbose

                try:
                    # <<< Capture tuple from run_experiment >>>
                    final_mse, early_stop_reason = run_experiment(
                        base_config_name=base_config_to_use,
                        config_overrides=current_overrides,
                        wandb_project_name=wandb_project,
                        wandb_run_name=wandb_run_name,
                        wandb_mode=wandb_mode_for_runs
                    )
                    print(f"Run {current_run} Result: final_train_mse = {final_mse:.8f}, Stop Reason: {early_stop_reason}")
                    
                    # <<< DEBUG PRINT >>>
                    print(f"DEBUG: Received from run_experiment - final_mse: {final_mse}, Type: {type(final_mse)}, Reason: {early_stop_reason}")

                    # Ensure final_mse is a number before comparison
                    is_valid_mse = isinstance(final_mse, (int, float)) and not np.isnan(final_mse) and final_mse != float('inf')

                    run_result_data = {
                        "run_number": current_run,
                        "lr_weights": lr_w,
                        "lr_hidden": lr_h,
                        "inference_lr_scale_base": scale_base,
                        "grad_clip_norm": clip_norm, # <<< ADD TO RESULTS
                        "mse": final_mse if is_valid_mse else "Failed or Invalid",
                        "early_stop_reason": early_stop_reason,
                        "wandb_run_name": wandb_run_name
                    }
                    all_run_results.append(run_result_data)

                    # <<< DEBUG PRINT: Check validity check >>>
                    print(f"DEBUG: is_valid_mse = {is_valid_mse}")

                    if is_valid_mse:
                        if final_mse < best_run_info["mse"]:
                            best_run_info["lr_weights"] = lr_w
                            best_run_info["lr_hidden"] = lr_h
                            best_run_info["inference_lr_scale_base"] = scale_base
                            best_run_info["grad_clip_norm"] = clip_norm # <<< ADD TO BEST RUN INFO
                            best_run_info["mse"] = final_mse
                            # <<< --- UPDATE PRINT STATEMENT --- >>>
                            print(f"*** New best MSE: {final_mse:.6f} with Params: lr_h={lr_h}, scale={scale_base}, clip={clip_norm} ***")
                            # <<< --- END UPDATE --- >>>
                    # else: # Already handled by mse value assignment above
                    #     print(f"Run {current_run} did not return a valid MSE.")

                    # <<< DEBUG PRINT: Check log formatting >>>
                    mse_val_debug = run_result_data['mse']
                    stop_reason_debug = run_result_data.get('early_stop_reason')
                    status_str_debug = ""
                    if isinstance(mse_val_debug, (int, float)) and not np.isnan(mse_val_debug) and mse_val_debug != float('inf'):
                        mse_str_debug = f"{mse_val_debug:.8f}"
                        if stop_reason_debug == 'Validation': status_str_debug = "(ES)"
                        elif stop_reason_debug is None: status_str_debug = "(Done)"
                    elif stop_reason_debug == 'NaN': mse_str_debug = "NaN"; status_str_debug = "(ES)"
                    elif stop_reason_debug == 'Exception': mse_str_debug = "Exception"; status_str_debug = "(Fail)"
                    else: mse_str_debug = str(mse_val_debug); status_str_debug = "(Fail)"
                    print(f"DEBUG: For log file - mse_str: '{mse_str_debug}', status_str: '{status_str_debug}'")

                except Exception as e:
                    # <<< --- UPDATE PRINT STATEMENT --- >>>
                    print(f"!!!! Run {current_run} failed with Params: lr_h={lr_h}, scale={scale_base}, clip={clip_norm}. Error: {e} !!!!")
                    # <<< --- END UPDATE --- >>>
                    import traceback
                    traceback.print_exc()
                    # Log failure in results
                    all_run_results.append({
                        "run_number": current_run,
                        "lr_weights": lr_w,
                        "lr_hidden": lr_h,
                        "inference_lr_scale_base": scale_base,
                        "grad_clip_norm": clip_norm,
                        "mse": "Exception",
                        "early_stop_reason": "Exception", # Indicate exception caused stop
                        "wandb_run_name": wandb_run_name
                    })

                end_time = time.time()
                print(f"Run {current_run} took {end_time - start_time:.2f} seconds.")
    # <<< --- END INNER LOOPS --- >>>


    print("\n--- Hyperparameter Search Complete ---")
    if best_run_info["lr_hidden"] is not None: # Check a searched param
        print(f"Best overall parameters found for minimum MSE:")
        print(f"  peak_lr_weights: {best_run_info['lr_weights']} (Fixed)")
        print(f"  peak_lr_hidden: {best_run_info['lr_hidden']}")
        print(f"  inference_lr_scale_base: {best_run_info['inference_lr_scale_base']}")
        print(f"  grad_clip_norm: {best_run_info['grad_clip_norm']}") # <<< ADD TO SUMMARY
        print(f"  Achieved Minimum MSE: {best_run_info['mse']:.6f}")
    else:
        print(f"No successful runs completed or no runs achieved a valid MSE.")

    # Log all results to a text file
    log_file_path = f"hyperparam_search_results_{base_config_to_use}_nb{MODEL_CONFIGS[base_config_to_use].num_blocks}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    # Ensure results directory exists if you want to place it there, e.g., ../results/
    # For simplicity, saving in the current directory (examples/)
    # log_file_path = os.path.join("..", "results", f"hyperparam_search_results_{base_config_to_use}_nb{fixed_overrides['num_blocks']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    # os.makedirs(os.path.dirname(log_file_path), exist_ok=True) # If saving to a subdirectory

    with open(log_file_path, 'w') as f:
        f.write(f"Hyperparameter Search Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Base Config: {base_config_to_use} (num_blocks={MODEL_CONFIGS[base_config_to_use].num_blocks})\n")
        f.write(f"Fixed Overrides (excluding searched params):\n")
        # Print fixed overrides neatly, excluding searched ones
        fixed_to_print = {k: v for k, v in fixed_overrides.items() if k not in ['peak_lr_hidden', 'inference_lr_scale_base', 'grad_clip_norm']}
        for key, value in fixed_to_print.items():
            f.write(f"  {key}: {value}\n")
        f.write("-------------------------------------------------------------------------------------------------\n")
        # <<< --- UPDATE LOG HEADER --- >>>
        f.write("Run | LR Hid | Scale | Clip  | W&B Run Name                             | Final Train MSE | Status\n")
        # <<< --- END UPDATE --- >>>
        f.write("-------------------------------------------------------------------------------------------------\n")
        # Sort results for clarity (e.g., by MSE then run number)
        all_run_results.sort(key=lambda x: (x['mse'] if isinstance(x['mse'], (int, float)) and not np.isnan(x['mse']) else float('inf'), x['run_number']))
        for result in all_run_results:
            # Format MSE string based on validity and early stopping
            mse_val = result['mse']
            stop_reason = result.get('early_stop_reason')
            status_str = ""
            if isinstance(mse_val, (int, float)) and not np.isnan(mse_val) and mse_val != float('inf'):
                mse_str = f"{mse_val:.8f}"
                if stop_reason == 'Validation':
                    status_str = "(ES)"
                elif stop_reason is None:
                    status_str = "(Done)" # Indicate normal completion
                # Keep status_str empty if NaN triggered stop but MSE is somehow valid (unlikely)
            elif stop_reason == 'NaN':
                mse_str = "NaN"
                status_str = "(ES)"
            elif stop_reason == 'Exception':
                mse_str = "Exception"
                status_str = "(Fail)"
            else:
                mse_str = str(mse_val) # Handle other "Failed or Invalid" cases
                status_str = "(Fail)"

            run_name_str = result.get('wandb_run_name', 'N/A')
            # <<< --- UPDATE LOG LINE --- >>>
            clip_norm_str = f"{result['grad_clip_norm']:.1f}" if result['grad_clip_norm'] is not None else "None"
            f.write(f"{result['run_number']:<3} | {result['lr_hidden']:<6} | {result['inference_lr_scale_base']:<5} | {clip_norm_str:<5} | {run_name_str:<42} | {mse_str:<15} | {status_str:<6}\n")
            # <<< --- END UPDATE --- >>>
        f.write("-------------------------------------------------------------------------------------------------\n")
        if best_run_info["lr_hidden"] is not None:
            # <<< --- UPDATE BEST RUN SUMMARY --- >>>
            clip_norm_best_str = f"{best_run_info['grad_clip_norm']:.1f}" if best_run_info['grad_clip_norm'] is not None else "None"
            f.write(f"Best Params: peak_lr_hidden={best_run_info['lr_hidden']}, inference_lr_scale_base={best_run_info['inference_lr_scale_base']}, grad_clip_norm={clip_norm_best_str} (lr_weights={best_run_info['lr_weights']} fixed)\n")
            f.write(f"Best MSE: {best_run_info['mse']:.6f}\n")
        else:
            f.write("No best run found.\n")
    print(f"Search results logged to: {log_file_path}")

if __name__ == "__main__":
    perform_hyperparameter_search()