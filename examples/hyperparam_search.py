import numpy as np
import time
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
import os


from debug_transformer_wandb import run_experiment, MODEL_CONFIGS

def perform_hyperparameter_search():
    print("Starting hyperparameter search...")

    base_config_to_use = "6block" 

    fixed_overrides = {
        "num_blocks": 6, 
        "num_heads": 1,
        "hidden_size": 64,
        "epochs": 75,
        "batch_size": 200, 
        "peak_lr_weights": 0.001, 
        # peak_lr_hidden will be set by candidates
        "reconstruction_every_n_epochs": 25, 
        "validation_every_n_epochs": 10,    
        "use_inference_lr_scaling": True,
        "use_lr_schedule": True,
        # inference_steps will be set by candidates
        # warmup_steps will be set by candidates
        # w_grad_clip_norm will be set by candidates
        "use_vode_state_layernorm": False, 
        "use_vode_grad_norm": False, # Ensuring this is True for the search
        # vode_grad_norm_target will be set by candidates
    }

    # lr_hidden_candidates = [0.095, 0.085]
    # inference_lr_scale_base_candidates = [1.21, 1.23, 1.25]
    # hidden_momentum_candidates = [0.4, 0.3]
    # h_grad_clip_norm_candidates = [5000]
    # seed_candidates = [50, 42] 

    lr_hidden_candidates = [0.095]
    inference_lr_scale_base_candidates = [1.25]
    hidden_momentum_candidates = [0.4]
    h_grad_clip_norm_candidates = [500, 1000, 2000]
    seed_candidates = [50, 42]
    inference_steps_candidates = [20]
    warmup_steps_candidates = [0]

    w_grad_clip_norm_candidates = [500.0] 
    vode_grad_norm_target_candidates = [50]

    # --- End Search Configuration ---

    best_run_info = {
        "lr_weights": fixed_overrides["peak_lr_weights"], 
        "lr_hidden": None, 
        "inference_lr_scale_base": None,
        "inference_steps": None, 
        "warmup_steps": None, 
        "h_grad_clip_norm": None,
        "w_grad_clip_norm": None,
        "vode_grad_norm_target": None, 
        "hidden_momentum": None, 
        "seed": None, 
        "overall_best_val_mse": float('inf'), 
        "run_best_train_mse_of_best_val_run": float('inf'), # Best train MSE of the run that got best val MSE
        "final_train_mse_of_best_val_run": float('inf') 
    }

    all_run_results = [] 

    total_runs = len(lr_hidden_candidates) * \
                 len(inference_lr_scale_base_candidates) * \
                 len(inference_steps_candidates) * \
                 len(warmup_steps_candidates) * \
                 len(h_grad_clip_norm_candidates) * \
                 len(w_grad_clip_norm_candidates) * \
                 len(vode_grad_norm_target_candidates) * \
                 len(hidden_momentum_candidates) * \
                 len(seed_candidates) 
    current_run = 0

    # --- WandB Configuration for Search ---
    try:
        from names_generator import generate_name
        creative_name = generate_name().replace('_', '-')
    except ImportError:
        import random
        adjectives = ["vibrant", "cosmic", "elegant", "swift", "radiant", "noble", "serene", "mighty", "gentle", "bold"]
        nouns = ["voyager", "nexus", "phoenix", "horizon", "quantum", "nebula", "zenith", "atlas", "aurora", "titan"]
        creative_name = f"{random.choice(adjectives)}-{random.choice(nouns)}"
    
    # Ensure MODEL_CONFIGS[base_config_to_use] is valid before accessing num_blocks
    if base_config_to_use not in MODEL_CONFIGS:
        print(f"Error: base_config_to_use '{base_config_to_use}' not found in MODEL_CONFIGS.")
        return

    wandb_project = f"pc-search-{creative_name}-{base_config_to_use}_nb{MODEL_CONFIGS[base_config_to_use].num_blocks}"
    wandb_mode_for_runs = "online" # Changed to online for active search, can be 'offline'

    lr_w = fixed_overrides["peak_lr_weights"] 
    for lr_h in lr_hidden_candidates:
        for scale_base in inference_lr_scale_base_candidates:
            for inf_steps in inference_steps_candidates: 
                for ws_val in warmup_steps_candidates: 
                    for h_clip in h_grad_clip_norm_candidates:
                        for w_clip in w_grad_clip_norm_candidates:
                            for norm_target_val in vode_grad_norm_target_candidates:
                                for hm_val in hidden_momentum_candidates:
                                    for seed_val in seed_candidates: 
                                    
                                        current_run += 1
                                        start_time = time.time()

                                        print(f"\n--- Starting Run {current_run}/{total_runs} ---")

                                        current_overrides = deepcopy(fixed_overrides)
                                        current_overrides["peak_lr_hidden"] = lr_h
                                        current_overrides["inference_lr_scale_base"] = scale_base # Will be 1.0 if use_inference_lr_scaling is False
                                        current_overrides["inference_steps"] = inf_steps 
                                        current_overrides["eval_inference_steps"] = [inf_steps] 
                                        current_overrides["reconstruction_steps"] = [inf_steps] 
                                        current_overrides["warmup_steps"] = ws_val 
                                        current_overrides["h_grad_clip_norm"] = h_clip
                                        current_overrides["w_grad_clip_norm"] = w_clip
                                        current_overrides["vode_grad_norm_target"] = norm_target_val # Added
                                        current_overrides["hidden_momentum"] = hm_val  # Added
                                        current_overrides["seed"] = seed_val 

                                        # Ensure num_blocks from fixed_overrides is used for consistency in run name
                                        nb_str = f"nb{fixed_overrides['num_blocks']}" 
                                        h_clip_str = f"hclip{h_clip:.0f}" if h_clip is not None else "hclipNone"
                                        w_clip_str = f"wclip{w_clip:.0f}" if w_clip is not None else "wclipNone"
                                        scale_base_str = f"sb{scale_base:.2f}".replace('.', 'p') if fixed_overrides.get("use_inference_lr_scaling", False) else "sbOFF"
                                        inf_steps_str = f"is{inf_steps}" 
                                        ws_str = f"ws{ws_val}" 
                                        norm_target_str = f"nt{norm_target_val:.1f}".replace('.', 'p') # Added
                                        hm_str = f"hm{hm_val:.2f}".replace('.', 'p') # Added
                                        lr_w_str = f"lrw{lr_w:.2e}"  
                                        lr_h_str = f"lrh{lr_h:.2e}"  
                                        wandb_run_name = f"{nb_str}_{lr_w_str}_{lr_h_str}_{scale_base_str}_{inf_steps_str}_{ws_str}_{norm_target_str}_{hm_str}_{h_clip_str}_{w_clip_str}_e{current_overrides['epochs']}_seed{seed_val}"

                                        print(f"Parameters: lr_h={lr_h}, scale={scale_base if fixed_overrides.get('use_inference_lr_scaling', False) else 'OFF'}, inf_steps={inf_steps}, warmup_steps={ws_val}, norm_target={norm_target_val}, hidden_momentum={hm_val}, h_clip={h_clip}, w_clip={w_clip}, epochs={current_overrides['epochs']}, seed={seed_val}")

                                        try:
                                            achieved_best_val_mse, run_best_train_mse, final_train_mse, early_stop_reason = run_experiment(
                                                base_config_name=base_config_to_use,
                                                config_overrides=current_overrides,
                                                wandb_project_name=wandb_project,
                                                wandb_run_name=wandb_run_name,
                                                wandb_mode=wandb_mode_for_runs
                                            )
                                            print(f"Run {current_run} Result: Best Val MSE = {achieved_best_val_mse:.8f}, Run Best Train MSE = {run_best_train_mse:.8f}, Final Train MSE = {final_train_mse:.8f}, Stop Reason: {early_stop_reason}")
                                            
                                            is_valid_val_mse = isinstance(achieved_best_val_mse, (int, float)) and not np.isnan(achieved_best_val_mse) and achieved_best_val_mse != float('inf')
                                            is_valid_run_best_train_mse = isinstance(run_best_train_mse, (int, float)) and not np.isnan(run_best_train_mse) and run_best_train_mse != float('inf')
                                            is_valid_final_train_mse = isinstance(final_train_mse, (int, float)) and not np.isnan(final_train_mse) and final_train_mse != float('inf')

                                            run_result_data = {
                                                "run_number": current_run,
                                                "lr_weights": lr_w,
                                                "lr_hidden": lr_h,
                                                "inference_lr_scale_base": scale_base if fixed_overrides.get("use_inference_lr_scaling", False) else "N/A",
                                                "inference_steps": inf_steps, 
                                                "warmup_steps": ws_val, 
                                                "h_grad_clip_norm": h_clip,
                                                "w_grad_clip_norm": w_clip,
                                                "vode_grad_norm_target": norm_target_val, 
                                                "hidden_momentum": hm_val, 
                                                "seed": seed_val, 
                                                "best_val_mse": achieved_best_val_mse if is_valid_val_mse else "Failed or Invalid",
                                                "run_best_train_mse": run_best_train_mse if is_valid_run_best_train_mse else "Failed or Invalid",
                                                "final_train_mse": final_train_mse if is_valid_final_train_mse else "Failed or Invalid",
                                                "early_stop_reason": early_stop_reason,
                                                "wandb_run_name": wandb_run_name
                                            }
                                            all_run_results.append(run_result_data)

                                            if is_valid_val_mse: 
                                                if achieved_best_val_mse < best_run_info["overall_best_val_mse"]:
                                                    best_run_info["lr_weights"] = lr_w
                                                    best_run_info["lr_hidden"] = lr_h
                                                    best_run_info["inference_lr_scale_base"] = scale_base if fixed_overrides.get("use_inference_lr_scaling", False) else "N/A"
                                                    best_run_info["inference_steps"] = inf_steps 
                                                    best_run_info["warmup_steps"] = ws_val 
                                                    best_run_info["h_grad_clip_norm"] = h_clip
                                                    best_run_info["w_grad_clip_norm"] = w_clip
                                                    best_run_info["vode_grad_norm_target"] = norm_target_val 
                                                    best_run_info["hidden_momentum"] = hm_val 
                                                    best_run_info["seed"] = seed_val 
                                                    best_run_info["overall_best_val_mse"] = achieved_best_val_mse
                                                    best_run_info["run_best_train_mse_of_best_val_run"] = run_best_train_mse if is_valid_run_best_train_mse else "N/A"
                                                    best_run_info["final_train_mse_of_best_val_run"] = final_train_mse if is_valid_final_train_mse else "N/A"
                                                    print(f"*** New best overall_best_val_mse: {achieved_best_val_mse:.6f} (Run Best Train MSE: {best_run_info['run_best_train_mse_of_best_val_run'] if isinstance(best_run_info['run_best_train_mse_of_best_val_run'], float) else 'N/A'}, Final Train MSE: {best_run_info['final_train_mse_of_best_val_run'] if isinstance(best_run_info['final_train_mse_of_best_val_run'], float) else 'N/A'}) with Params: lr_h={lr_h}, scale={scale_base if fixed_overrides.get('use_inference_lr_scaling', False) else 'OFF'}, inf_steps={inf_steps}, warmup_steps={ws_val}, norm_target={norm_target_val}, hidden_momentum={hm_val}, h_clip={h_clip}, w_clip={w_clip}, seed={seed_val} ***")
                                            # Fallback logic if no valid val_mse has been found yet, but we have a valid run_best_train_mse
                                            elif is_valid_run_best_train_mse and best_run_info["overall_best_val_mse"] == float('inf'):
                                                if run_best_train_mse < best_run_info["run_best_train_mse_of_best_val_run"]: # Compare with existing best train mse as a fallback
                                                    best_run_info["lr_weights"] = lr_w
                                                    best_run_info["lr_hidden"] = lr_h
                                                    best_run_info["inference_lr_scale_base"] = scale_base if fixed_overrides.get("use_inference_lr_scaling", False) else "N/A"
                                                    best_run_info["inference_steps"] = inf_steps 
                                                    best_run_info["warmup_steps"] = ws_val 
                                                    best_run_info["h_grad_clip_norm"] = h_clip
                                                    best_run_info["w_grad_clip_norm"] = w_clip
                                                    best_run_info["vode_grad_norm_target"] = norm_target_val 
                                                    best_run_info["hidden_momentum"] = hm_val 
                                                    best_run_info["seed"] = seed_val 
                                                    best_run_info["run_best_train_mse_of_best_val_run"] = run_best_train_mse
                                                    best_run_info["final_train_mse_of_best_val_run"] = final_train_mse if is_valid_final_train_mse else "N/A"
                                                    print(f"*** No valid validation MSEs yet. New best run_best_train_mse: {run_best_train_mse:.6f} (Final Train MSE: {best_run_info['final_train_mse_of_best_val_run'] if isinstance(best_run_info['final_train_mse_of_best_val_run'], float) else 'N/A'}) with Params: lr_h={lr_h}, scale={scale_base if fixed_overrides.get('use_inference_lr_scaling', False) else 'OFF'}, inf_steps={inf_steps}, warmup_steps={ws_val}, norm_target={norm_target_val}, hidden_momentum={hm_val}, h_clip={h_clip}, w_clip={w_clip}, seed={seed_val} ***")

                                        except Exception as e:
                                            print(f"!!!! Run {current_run} failed with Params: lr_h={lr_h}, scale={scale_base if fixed_overrides.get('use_inference_lr_scaling', False) else 'OFF'}, inf_steps={inf_steps}, warmup_steps={ws_val}, norm_target={norm_target_val}, hidden_momentum={hm_val}, h_clip={h_clip}, w_clip={w_clip}, seed={seed_val}. Error: {e} !!!!")
                                            import traceback
                                            traceback.print_exc()
                                            all_run_results.append({
                                                "run_number": current_run,
                                                "lr_weights": lr_w,
                                                "lr_hidden": lr_h,
                                                "inference_lr_scale_base": scale_base if fixed_overrides.get('use_inference_lr_scaling', False) else "N/A",
                                                "inference_steps": inf_steps, 
                                                "warmup_steps": ws_val, 
                                                "h_grad_clip_norm": h_clip,
                                                "w_grad_clip_norm": w_clip,
                                                "vode_grad_norm_target": norm_target_val, # Added
                                                "hidden_momentum": hm_val, # Added
                                                "seed": seed_val, 
                                                "best_val_mse": "Exception",
                                                "run_best_train_mse": "Exception",
                                                "final_train_mse": "Exception", 
                                                "early_stop_reason": "Exception", 
                                                "wandb_run_name": wandb_run_name
                                            })

                                        end_time = time.time()
                                        print(f"Run {current_run} took {end_time - start_time:.2f} seconds.")

    print("\n--- Hyperparameter Search Complete ---")
    if best_run_info["lr_hidden"] is not None: 
        print(f"Best overall parameters found for minimum MSE:")
        print(f"  peak_lr_weights: {best_run_info['lr_weights']} (Fixed)")
        print(f"  peak_lr_hidden: {best_run_info['lr_hidden']}")
        print(f"  inference_lr_scale_base: {best_run_info['inference_lr_scale_base']}")
        print(f"  inference_steps: {best_run_info['inference_steps']}") 
        print(f"  warmup_steps: {best_run_info['warmup_steps']}") 
        print(f"  vode_grad_norm_target: {best_run_info['vode_grad_norm_target']}") # Added
        print(f"  hidden_momentum: {best_run_info['hidden_momentum']}") # Added
        print(f"  h_grad_clip_norm: {best_run_info['h_grad_clip_norm']}")
        print(f"  w_grad_clip_norm: {best_run_info['w_grad_clip_norm']}")
        print(f"  seed: {best_run_info['seed']}") 
        print(f"  Achieved Overall Best Validation MSE: {best_run_info['overall_best_val_mse']:.6f}")
        print(f"  Run Best Train MSE of Best Val Run: {best_run_info['run_best_train_mse_of_best_val_run'] if isinstance(best_run_info['run_best_train_mse_of_best_val_run'], float) else 'N/A'}")
        print(f"  Final Train MSE of Best Val Run: {best_run_info['final_train_mse_of_best_val_run'] if isinstance(best_run_info['final_train_mse_of_best_val_run'], float) else 'N/A'}")
    else:
        print(f"No successful runs completed or no runs achieved a valid MSE.")

    # Use num_blocks from fixed_overrides for log file name consistency
    log_file_nb_str = fixed_overrides['num_blocks']
    log_file_path = f"hyperparam_search_results_{base_config_to_use}_nb{log_file_nb_str}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    with open(log_file_path, 'w') as f:
        f.write(f"Hyperparameter Search Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Base Config: {base_config_to_use} (num_blocks={log_file_nb_str})\n") # Use fixed num_blocks
        f.write(f"Fixed Overrides (excluding searched params):\n")
        
        # Define keys that are part of the search space
        searched_keys = [
            'peak_lr_hidden', 'inference_lr_scale_base', 'inference_steps', 
            'eval_inference_steps', 'reconstruction_steps', 'warmup_steps', 
            'h_grad_clip_norm', 'w_grad_clip_norm', 'vode_grad_norm_target', 'hidden_momentum', 'seed'
        ]
        # Add keys that are dependent on other fixed_overrides for printing
        # For example, if use_inference_lr_scaling is False, inference_lr_scale_base is effectively fixed
        # but it's simpler to just exclude all searched_keys from the fixed_to_print.

        fixed_to_print = {k: v for k, v in fixed_overrides.items() if k not in searched_keys}
        # Special handling for use_inference_lr_scaling to show its fixed value if it is fixed
        if 'use_inference_lr_scaling' in fixed_overrides:
             fixed_to_print['use_inference_lr_scaling'] = fixed_overrides['use_inference_lr_scaling']


        for key, value in fixed_to_print.items():
            f.write(f"  {key}: {value}\n")
        f.write("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n")
        
        # Adjusted header_parts and col_widths to include Vode Grad Norm Target
        header_parts = [
            "Run", "LR Hid", "Scale", "Inf Steps", "Warmup Steps", "Norm Target", "Hid Mom", "H Clip", "W Clip", "Seed", 
            "Best Val MSE", "Run Best Train MSE", "Final Train MSE", "Status", "W&B Run Name"
        ]
        col_widths = {
            "Run": 3, "LR Hid": 6, "Scale": 7, "Inf Steps": 9, "Warmup Steps": 12, 
            "Norm Target": 11, "Hid Mom": 7, "H Clip": 6, "W Clip": 6, "Seed": 4, 
            "Best Val MSE": 14, "Run Best Train MSE": 18 , "Final Train MSE": 15, "Status": 8, "W&B Run Name": 40
        }
        
        # Filter out keys from fixed_overrides that are part of the new normalization params but not explicitly searched
        # (use_vode_state_layernorm is fixed False, use_vode_grad_norm is fixed True)
        # These are already in fixed_to_print if they are not in searched_keys.
        
        header_str = " | ".join([f"{h:<{col_widths[h]}}" for h in header_parts])
        f.write(header_str + "\n")
        f.write("-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n") # Adjusted separator length
        
        all_run_results.sort(key=lambda x: (x['best_val_mse'] if isinstance(x['best_val_mse'], (int, float)) and not np.isnan(x['best_val_mse']) else float('inf'), 
                                          x['run_best_train_mse'] if isinstance(x['run_best_train_mse'], (int, float)) and not np.isnan(x['run_best_train_mse']) else float('inf'),
                                          x['final_train_mse'] if isinstance(x['final_train_mse'], (int, float)) and not np.isnan(x['final_train_mse']) else float('inf'), 
                                          x['run_number']))
        
        for result in all_run_results:
            best_val_mse_val = result['best_val_mse']
            run_best_train_mse_val = result['run_best_train_mse']
            final_train_mse_val = result['final_train_mse']
            stop_reason = result.get('early_stop_reason')
            status_str = ""

            if isinstance(best_val_mse_val, (int, float)) and not np.isnan(best_val_mse_val) and best_val_mse_val != float('inf'):
                best_val_mse_str = f"{best_val_mse_val:.8f}"
                if stop_reason == 'Validation': status_str = "(ES-Val)" # More specific ES
                elif stop_reason is None: status_str = "(Done)"
            elif stop_reason == 'NaN' and result['best_val_mse'] == 'Failed or Invalid': # Check if specifically val mse was NaN
                best_val_mse_str = "NaN-Val"; status_str = "(ES-NaN)"
            elif stop_reason == 'Exception': best_val_mse_str = "Exception"; status_str = "(Fail)"
            else: best_val_mse_str = str(best_val_mse_val); status_str = "(ValFail)"

            if isinstance(run_best_train_mse_val, (int, float)) and not np.isnan(run_best_train_mse_val) and run_best_train_mse_val != float('inf'):
                run_best_train_mse_str = f"{run_best_train_mse_val:.8f}"
            else: run_best_train_mse_str = str(run_best_train_mse_val)

            if isinstance(final_train_mse_val, (int, float)) and not np.isnan(final_train_mse_val) and final_train_mse_val != float('inf'):
                final_train_mse_str = f"{final_train_mse_val:.8f}"
                # Status string might already be set by best_val_mse logic, append if necessary or make it more specific
                if not status_str: # If status not set by val_mse (e.g. val_mse was inf but train_mse is valid)
                    if stop_reason == 'Validation': status_str = "(ES)" # Should not happen if val_mse was inf
                    elif stop_reason is None: status_str = "(Done)"
                elif stop_reason == 'NaN' and not status_str.endswith("(ES-NaN)"): # If final_train_mse is NaN and not already marked
                     final_train_mse_str = "NaN"; 
                     if status_str: 
                         status_str += "/TrNaN"
                     else: 
                         status_str = "(TrNaN)"
                elif result.get('early_stop_reason') == 'Exception' and not status_str.endswith("(Fail)") : # check 'result' directly as final_train_mse_val might be 'Exception' string
                    final_train_mse_str = "Exception"
                    if status_str: 
                        status_str += "/TrFail"
                    else: 
                        status_str = "(TrFail)"
                else:
                    final_train_mse_str = str(final_train_mse_val)
                    if not status_str: status_str = "(TrFail)"


            run_name_str = result.get('wandb_run_name', 'N/A')
            h_clip_str = f"{result['h_grad_clip_norm']:.1f}" if result['h_grad_clip_norm'] is not None else "None"
            w_clip_str = f"{result['w_grad_clip_norm']:.1f}" if result['w_grad_clip_norm'] is not None else "None"
            inf_steps_log_str = str(result.get('inference_steps', 'N/A')) 
            warmup_steps_log_str = str(result.get('warmup_steps', 'N/A')) 
            norm_target_log_str = f"{result.get('vode_grad_norm_target', 'N/A'):.1f}" # Added
            hm_log_str = f"{result.get('hidden_momentum', 'N/A'):.2f}" # Added
            seed_str = str(result.get('seed', 'N/A')) 
            
            scale_log_str = result.get('inference_lr_scale_base', 'N/A')
            if isinstance(scale_log_str, float): scale_log_str = f"{scale_log_str:.2f}"


            log_line_parts = {
                "Run": result['run_number'],
                "LR Hid": f"{result['lr_hidden']:.2e}", # Format LR for consistency
                "Scale": scale_log_str,
                "Inf Steps": inf_steps_log_str,
                "Warmup Steps": warmup_steps_log_str,
                "Norm Target": norm_target_log_str, # Added
                "Hid Mom": hm_log_str, # Added
                "H Clip": h_clip_str,
                "W Clip": w_clip_str,
                "Seed": seed_str,
                "Best Val MSE": best_val_mse_str,
                "Run Best Train MSE": run_best_train_mse_str,
                "Final Train MSE": final_train_mse_str,
                "Status": status_str,
                "W&B Run Name": run_name_str
            }
            log_line_str = " | ".join([f"{str(log_line_parts[h]):<{col_widths[h]}}" for h in header_parts])
            f.write(log_line_str + "\n") 
        f.write("-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n") # Adjusted separator length
        if best_run_info["lr_hidden"] is not None:
            h_clip_best_str = f"{best_run_info['h_grad_clip_norm']:.1f}" if best_run_info['h_grad_clip_norm'] is not None else "None"
            w_clip_best_str = f"{best_run_info['w_grad_clip_norm']:.1f}" if best_run_info['w_grad_clip_norm'] is not None else "None"
            norm_target_best_str = f"{best_run_info['vode_grad_norm_target']:.1f}" if best_run_info['vode_grad_norm_target'] is not None else "None" 
            hm_best_str = f"{best_run_info['hidden_momentum']:.2f}" if best_run_info['hidden_momentum'] is not None else "None" 
            
            best_scale_str = best_run_info['inference_lr_scale_base']
            if isinstance(best_scale_str, float): best_scale_str = f"{best_scale_str:.2f}"


            f.write(f"Best Params: peak_lr_hidden={best_run_info['lr_hidden']:.2e}, inference_lr_scale_base={best_scale_str}, inference_steps={best_run_info['inference_steps']}, warmup_steps={best_run_info['warmup_steps']}, vode_grad_norm_target={norm_target_best_str}, hidden_momentum={hm_best_str}, h_grad_clip_norm={h_clip_best_str}, w_grad_clip_norm={w_clip_best_str}, seed={best_run_info['seed']}\n")
            f.write(f"Achieved Overall Best Validation MSE: {best_run_info['overall_best_val_mse']:.6f}\n")
            f.write(f"Run Best Train MSE of Best Val Run: {best_run_info['run_best_train_mse_of_best_val_run'] if isinstance(best_run_info['run_best_train_mse_of_best_val_run'], float) else 'N/A'}\n")
            f.write(f"Final Train MSE of Best Val Run: {best_run_info['final_train_mse_of_best_val_run'] if isinstance(best_run_info['final_train_mse_of_best_val_run'], float) else 'N/A'}\n")
            f.write(f" (lr_weights={best_run_info['lr_weights']:.2e} fixed)\n") 
    print(f"Search results logged to: {log_file_path}")

if __name__ == "__main__":
    perform_hyperparameter_search()