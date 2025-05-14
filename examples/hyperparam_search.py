import numpy as np
import time
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
import os


from debug_transformer_wandb import run_experiment, MODEL_CONFIGS

def perform_hyperparameter_search():
    print("Starting hyperparameter search...")

    base_config_to_use = "debug_tiny" # Base config already has num_blocks=5

    # Fixed parameters for this new focused search (50 epochs)
    fixed_overrides = {
        "num_blocks": 5, 
        "num_heads": 1,
        "hidden_size": 64,
        "epochs": 50, # Longer epochs for focused search
        "peak_lr_weights": 0.001, 
        "peak_lr_hidden": 0.1, # Best found hidden LR
        "reconstruction_every_n_epochs": 25, # Adjust for longer runs
        "validation_every_n_epochs": 1,    
        "use_inference_lr_scaling": True,
        "h_grad_clip_norm": 1000.0, 
        "w_grad_clip_norm": 500.0,   
        "use_status_init_in_training": False, 
        "use_status_init_in_unmasking": False 
    }

    # Parameter candidates for the new focused grid search (50 epochs)
    lr_hidden_candidates = [0.1, 1.1, 1.2] # Fixed to best found
    inference_lr_scale_base_candidates = [1.15, 1.20] # Best performing scales
    h_grad_clip_norm_candidates = [fixed_overrides["h_grad_clip_norm"]] # Fixed
    w_grad_clip_norm_candidates = [fixed_overrides["w_grad_clip_norm"]] # Fixed
    seed_candidates = [42, 123, 1024] # Seeds for robustness check

    # --- End Search Configuration ---

    best_run_info = {
        "lr_weights": fixed_overrides["peak_lr_weights"], # Fixed
        "lr_hidden": None, 
        "inference_lr_scale_base": None,
        "h_grad_clip_norm": None,
        "w_grad_clip_norm": None,
        "seed": None, # Added seed to best_run_info
        "mse": float('inf'),
    }

    all_run_results = [] # To store results of all runs for logging

    # <<< --- UPDATE TOTAL RUNS --- >>>
    total_runs = len(lr_hidden_candidates) * len(inference_lr_scale_base_candidates) * \
                 len(h_grad_clip_norm_candidates) * len(w_grad_clip_norm_candidates) * \
                 len(seed_candidates) # Added seeds to total runs
    # <<< --- END UPDATE --- >>>
    current_run = 0

    # --- WandB Configuration for Search ---
    # It's good practice to group search runs under a common project
    # Use the names-generator from docker for creative run names like "eloquent_einstein"
    try:
        from names_generator import generate_name
        creative_name = generate_name().replace('_', '-')
    except ImportError:
        # Fallback to a simple random adjective-noun combination if library not available
        import random
        adjectives = ["vibrant", "cosmic", "elegant", "swift", "radiant", "noble", "serene", "mighty", "gentle", "bold"]
        nouns = ["voyager", "nexus", "phoenix", "horizon", "quantum", "nebula", "zenith", "atlas", "aurora", "titan"]
        creative_name = f"{random.choice(adjectives)}-{random.choice(nouns)}"
    
    wandb_project = f"pc-search-{creative_name}-{base_config_to_use}_nb{MODEL_CONFIGS[base_config_to_use].num_blocks}"
    # Disable W&B logging for individual runs to speed up search if desired,
    # or use "online" to log each run.
    # wandb_mode_for_runs = "disabled"
    wandb_mode_for_runs = "online" # Log each run

    # <<< --- REMOVE OUTER LR_W LOOP --- >>>
    lr_w = fixed_overrides["peak_lr_weights"] # Use fixed value
    for lr_h in lr_hidden_candidates:
        for scale_base in inference_lr_scale_base_candidates:
            # <<< --- ADD LOOP FOR CLIP NORM --- >>>
            for h_clip in h_grad_clip_norm_candidates:
                for w_clip in w_grad_clip_norm_candidates:
            # <<< --- END ADD --- >>>
                    for seed_val in seed_candidates: # New loop for seeds
                        current_run += 1
                        start_time = time.time()

                        print(f"\n--- Starting Run {current_run}/{total_runs} ---")

                        current_overrides = deepcopy(fixed_overrides)
                        current_overrides["peak_lr_hidden"] = lr_h
                        current_overrides["inference_lr_scale_base"] = scale_base
                        # current_overrides["grad_clip_norm"] = clip_norm # Old combined
                        current_overrides["h_grad_clip_norm"] = h_clip
                        current_overrides["w_grad_clip_norm"] = w_clip
                        current_overrides["seed"] = seed_val # Override seed

                        # Create a unique run name for WandB if logging is enabled
                        nb_str = f"nb{MODEL_CONFIGS[base_config_to_use].num_blocks}"
                        h_clip_str = f"hclip{h_clip:.0f}" if h_clip is not None else "hclipNone"
                        w_clip_str = f"wclip{w_clip:.0f}" if w_clip is not None else "wclipNone"
                        # Corrected formatting for scale_base: .2f and replace . with p
                        scale_base_str = f"sb{scale_base:.2f}".replace('.', 'p')
                        wandb_run_name = f"{nb_str}_lrw{lr_w:.0e}_lrh{lr_h:.0e}_{scale_base_str}_{h_clip_str}_{w_clip_str}_e{fixed_overrides['epochs']}_seed{seed_val}"

                        print(f"Parameters: lr_h={lr_h}, scale={scale_base}, h_clip={h_clip}, w_clip={w_clip}, epochs={fixed_overrides['epochs']}, seed={seed_val}")

                        try:
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
                                # "grad_clip_norm": clip_norm, # Old combined
                                "h_grad_clip_norm": h_clip,
                                "w_grad_clip_norm": w_clip,
                                "seed": seed_val, # Log seed
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
                                    # best_run_info["grad_clip_norm"] = clip_norm # Old combined
                                    best_run_info["h_grad_clip_norm"] = h_clip
                                    best_run_info["w_grad_clip_norm"] = w_clip
                                    best_run_info["seed"] = seed_val # Store best seed
                                    best_run_info["mse"] = final_mse
                                    print(f"*** New best MSE: {final_mse:.6f} with Params: lr_h={lr_h}, scale={scale_base}, h_clip={h_clip}, w_clip={w_clip}, seed={seed_val} ***")
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
                            print(f"!!!! Run {current_run} failed with Params: lr_h={lr_h}, scale={scale_base}, h_clip={h_clip}, w_clip={w_clip}, seed={seed_val}. Error: {e} !!!!")
                            # <<< --- END UPDATE --- >>>
                            import traceback
                            traceback.print_exc()
                            # Log failure in results
                            all_run_results.append({
                                "run_number": current_run,
                                "lr_weights": lr_w,
                                "lr_hidden": lr_h,
                                "inference_lr_scale_base": scale_base,
                                # "grad_clip_norm": clip_norm, # Old combined
                                "h_grad_clip_norm": h_clip,
                                "w_grad_clip_norm": w_clip,
                                "seed": seed_val, # Log seed for failed run
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
        # print(f"  grad_clip_norm: {best_run_info['grad_clip_norm']}") # Old combined
        print(f"  h_grad_clip_norm: {best_run_info['h_grad_clip_norm']}")
        print(f"  w_grad_clip_norm: {best_run_info['w_grad_clip_norm']}")
        print(f"  seed: {best_run_info['seed']}") # Print best seed
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
        fixed_to_print = {k: v for k, v in fixed_overrides.items() if k not in [
            'peak_lr_hidden', 
            'inference_lr_scale_base', 
            'h_grad_clip_norm', 
            'w_grad_clip_norm',
            'seed' # Exclude seed from fixed if it's part of the search
            # 'grad_clip_norm' # Old combined
        ]}
        for key, value in fixed_to_print.items():
            f.write(f"  {key}: {value}\n")
        f.write("-------------------------------------------------------------------------------------------------\n")
        # <<< --- UPDATE LOG HEADER --- >>>
        f.write("Run | LR Hid | Scale | H Clip | W Clip | Seed | W&B Run Name                                 | Final Train MSE | Status\n")
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
            h_clip_str = f"{result['h_grad_clip_norm']:.1f}" if result['h_grad_clip_norm'] is not None else "None"
            w_clip_str = f"{result['w_grad_clip_norm']:.1f}" if result['w_grad_clip_norm'] is not None else "None"
            seed_str = str(result.get('seed', 'N/A')) # Get seed string
            f.write(f"{result['run_number']:<3} | {result['lr_hidden']:<6} | {result['inference_lr_scale_base']:<5} | {h_clip_str:<6} | {w_clip_str:<6} | {seed_str:<4} | {run_name_str:<51} | {mse_str:<15} | {status_str:<6}\n") # Added seed to log line
            # <<< --- END UPDATE --- >>>
        f.write("-------------------------------------------------------------------------------------------------\n")
        if best_run_info["lr_hidden"] is not None:
            # <<< --- UPDATE BEST RUN SUMMARY --- >>>
            h_clip_best_str = f"{best_run_info['h_grad_clip_norm']:.1f}" if best_run_info['h_grad_clip_norm'] is not None else "None"
            w_clip_best_str = f"{best_run_info['w_grad_clip_norm']:.1f}" if best_run_info['w_grad_clip_norm'] is not None else "None"
            f.write(f"Best Params: peak_lr_hidden={best_run_info['lr_hidden']}, inference_lr_scale_base={best_run_info['inference_lr_scale_base']}, h_grad_clip_norm={h_clip_best_str}, w_grad_clip_norm={w_clip_best_str}, seed={best_run_info['seed']} (lr_weights={best_run_info['lr_weights']} fixed)\n")
            f.write(f"Best MSE: {best_run_info['mse']:.6f}\n")
        else:
            f.write("No best run found.\n")
    print(f"Search results logged to: {log_file_path}")

if __name__ == "__main__":
    perform_hyperparameter_search()