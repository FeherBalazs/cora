import numpy as np
import time
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
import os


from debug_transformer_wandb import run_experiment, MODEL_CONFIGS

def perform_hyperparameter_search():
    print("Starting hyperparameter search...")

    base_config_to_use = "6block" # Base for non-overridden/non-searched params

    # Fixed overrides: Non-searched parameters, taking cues from the "6block" base
    fixed_overrides = {
        "epochs": 75,
        "theta": 10_000,
        "use_ssl_augmentations": True,
        "use_cifar10_norm": False,
        "num_images": 3,
        "test_subset": 200,
        "train_subset": 50000,
        "peak_lr_weights": 0.001,
        "hidden_lr_inference": 0.095,
        "reconstruction_every_n_epochs": 75,
        "validation_every_n_epochs": 75,
        "use_inference_lr_scaling": True,
        "use_lr_schedule_w": True,
        "use_lr_schedule_h": True,
        "weight_decay": 2e-4,
        "mlp_ratio": 4.0,
        "patch_size": 4,
        "use_noise": True,
        "update_weights_every_inference_step": False,

        "use_early_stopping": True,
        "early_stopping_patience": 50,
        "early_stopping_min_delta": 0.001,
        "early_stopping_metric": "train_mse",
        "save_model_train_mse_threshold": 0.01,
        "model_saving_metric": "train_mse",
        
        "use_vode_grad_norm": False,
        "use_adamw_for_hidden_optimizer": False,
        "corrupt_ratio": 0.25,
        "use_lower_half_mask": False,
        "inference_clamp_alpha": 1.0,
        "save_reconstruction_images": True,
        "save_reconstruction_video": True,
        "video_fps": 5,
        "reinitialize_model_for_each_epoch": False,
        "use_status_init_in_training": False,
        "use_status_init_in_unmasking": False,
        "lr_schedule_min_lr_factor": 0.5
    }

    # --- Architectural Search Space ---
    num_blocks_candidates = [6]
    batch_size_candidates = [200]
    hidden_size_candidates = [64]
    num_heads_candidates = [1]

    # --- Hyperparameter Search Space ---
    lr_hidden_candidates = [0.095]
    inference_lr_scale_base_candidates = [1.25]
    hidden_momentum_candidates = [0.4]
    h_grad_clip_norm_candidates = [2000]
    seed_candidates = [10]
    inference_steps_candidates = [20]
    warmup_steps_candidates = [0]
    w_grad_clip_norm_candidates = [500.0]
    use_vode_state_layernorm_candidates = [False]

    best_run_info = {
        "num_blocks": None,
        "batch_size": None,
        "hidden_size": None,
        "num_heads": None,
        "use_vode_state_layernorm": None, # New
        "lr_weights": fixed_overrides["peak_lr_weights"],
        "lr_hidden": None,
        "inference_lr_scale_base": None,
        "inference_steps": None,
        "warmup_steps": None,
        "h_grad_clip_norm": None,
        "w_grad_clip_norm": None,
        "hidden_momentum": None,
        "seed": None,
        "overall_best_val_mse": float('inf'),
        "run_best_train_mse_of_best_val_run": float('inf'),
        "final_train_mse_of_best_val_run": float('inf')
    }

    all_run_results = []

    total_runs = len(num_blocks_candidates) * \
                 len(batch_size_candidates) * \
                 len(hidden_size_candidates) * \
                 len(num_heads_candidates) * \
                 len(lr_hidden_candidates) * \
                 len(inference_lr_scale_base_candidates) * \
                 len(inference_steps_candidates) * \
                 len(warmup_steps_candidates) * \
                 len(h_grad_clip_norm_candidates) * \
                 len(w_grad_clip_norm_candidates) * \
                 len(hidden_momentum_candidates) * \
                 len(use_vode_state_layernorm_candidates) * \
                 len(seed_candidates)
    current_run = 0

    try:
        from names_generator import generate_name
        creative_name = generate_name().replace('_', '-')
    except ImportError:
        import random
        adjectives = ["vibrant", "cosmic", "elegant", "swift", "radiant", "noble", "serene", "mighty", "gentle", "bold"]
        nouns = ["voyager", "nexus", "phoenix", "horizon", "quantum", "nebula", "zenith", "atlas", "aurora", "titan"]
        creative_name = f"{random.choice(adjectives)}-{random.choice(nouns)}"

    wandb_project = f"pc-arch-search-{creative_name}"
    wandb_mode_for_runs = "online"

    lr_w = fixed_overrides["peak_lr_weights"]

    for nb_val in num_blocks_candidates:
        for bs_val in batch_size_candidates:
            for hs_val in hidden_size_candidates:
                for nh_val in num_heads_candidates:
                    for lr_h in lr_hidden_candidates:
                        for scale_base in inference_lr_scale_base_candidates:
                            for inf_steps in inference_steps_candidates:
                                for ws_val in warmup_steps_candidates:
                                    for h_clip in h_grad_clip_norm_candidates:
                                        for w_clip in w_grad_clip_norm_candidates:
                                            for hm_val in hidden_momentum_candidates:
                                                for vln_val in use_vode_state_layernorm_candidates: # New loop
                                                    for seed_val in seed_candidates:

                                                        current_run += 1
                                                        start_time = time.time()

                                                        print(f"\n--- Starting Run {current_run}/{total_runs} ---")

                                                        current_overrides = deepcopy(fixed_overrides)
                                                        current_overrides["num_blocks"] = nb_val
                                                        current_overrides["batch_size"] = bs_val
                                                        current_overrides["hidden_size"] = hs_val
                                                        current_overrides["num_heads"] = nh_val
                                                        current_overrides["peak_lr_hidden"] = lr_h
                                                        current_overrides["inference_lr_scale_base"] = scale_base
                                                        current_overrides["inference_steps"] = inf_steps
                                                        # current_overrides["eval_inference_steps"] = [inf_steps]
                                                        # current_overrides["reconstruction_steps"] = [inf_steps]
                                                        current_overrides["warmup_steps"] = ws_val
                                                        current_overrides["h_grad_clip_norm"] = h_clip
                                                        current_overrides["w_grad_clip_norm"] = w_clip
                                                        current_overrides["hidden_momentum"] = hm_val
                                                        current_overrides["use_vode_state_layernorm"] = vln_val # New
                                                        current_overrides["seed"] = seed_val

                                                        nb_str = f"nb{nb_val}"
                                                        bs_str = f"bs{bs_val}"
                                                        hs_str = f"hs{hs_val}"
                                                        nh_str = f"nh{nh_val}"
                                                        lr_h_str = f"lrh{lr_h:.3f}".replace('.', 'p')
                                                        scale_base_str = f"sb{scale_base:.2f}".replace('.', 'p') if current_overrides.get("use_inference_lr_scaling", False) else "sbOFF"
                                                        inf_steps_str = f"is{inf_steps}"
                                                        ws_str = f"ws{ws_val}"
                                                        hm_str = f"hm{hm_val:.2f}".replace('.', 'p')
                                                        h_clip_str = f"hclip{h_clip:.0f}" if h_clip is not None else "hclipNone"
                                                        w_clip_str = f"wclip{w_clip:.0f}" if w_clip is not None else "wclipNone"
                                                        vln_str = f"vln{'ON' if vln_val else 'OFF'}" # New
                                                        
                                                        wandb_run_name = f"{nb_str}_{bs_str}_{hs_str}_{nh_str}_{lr_h_str}_{scale_base_str}_{inf_steps_str}_{ws_str}_{hm_str}_{h_clip_str}_{w_clip_str}_{vln_str}_e{current_overrides['epochs']}_sd{seed_val}"
                                                        
                                                        print(f"Parameters: num_blocks={nb_val}, batch_size={bs_val}, hidden_size={hs_val}, num_heads={nh_val}, use_vode_state_layernorm={vln_val}, lr_h={lr_h}, scale={scale_base if current_overrides.get('use_inference_lr_scaling', False) else 'OFF'}, inf_steps={inf_steps}, warmup_steps={ws_val}, hidden_momentum={hm_val}, h_clip={h_clip}, w_clip={w_clip}, seed={seed_val}")

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
                                                                "num_blocks": nb_val,
                                                                "batch_size": bs_val,
                                                                "hidden_size": hs_val,
                                                                "num_heads": nh_val,
                                                                "use_vode_state_layernorm": vln_val, # New
                                                                "lr_weights": lr_w,
                                                                "lr_hidden": lr_h,
                                                                "inference_lr_scale_base": scale_base if current_overrides.get("use_inference_lr_scaling", False) else "N/A",
                                                                "inference_steps": inf_steps,
                                                                "warmup_steps": ws_val,
                                                                "h_grad_clip_norm": h_clip,
                                                                "w_grad_clip_norm": w_clip,
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
                                                                    best_run_info["num_blocks"] = nb_val
                                                                    best_run_info["batch_size"] = bs_val
                                                                    best_run_info["hidden_size"] = hs_val
                                                                    best_run_info["num_heads"] = nh_val
                                                                    best_run_info["use_vode_state_layernorm"] = vln_val # New
                                                                    best_run_info["lr_weights"] = lr_w
                                                                    best_run_info["lr_hidden"] = lr_h
                                                                    best_run_info["inference_lr_scale_base"] = scale_base if current_overrides.get("use_inference_lr_scaling", False) else "N/A"
                                                                    best_run_info["inference_steps"] = inf_steps
                                                                    best_run_info["warmup_steps"] = ws_val
                                                                    best_run_info["h_grad_clip_norm"] = h_clip
                                                                    best_run_info["w_grad_clip_norm"] = w_clip
                                                                    best_run_info["hidden_momentum"] = hm_val
                                                                    best_run_info["seed"] = seed_val
                                                                    best_run_info["overall_best_val_mse"] = achieved_best_val_mse
                                                                    best_run_info["run_best_train_mse_of_best_val_run"] = run_best_train_mse if is_valid_run_best_train_mse else "N/A"
                                                                    best_run_info["final_train_mse_of_best_val_run"] = final_train_mse if is_valid_final_train_mse else "N/A"
                                                                    print(f"*** New best overall_best_val_mse: {achieved_best_val_mse:.6f} (Run Best Train MSE: {best_run_info['run_best_train_mse_of_best_val_run'] if isinstance(best_run_info['run_best_train_mse_of_best_val_run'], float) else 'N/A'}, Final Train MSE: {best_run_info['final_train_mse_of_best_val_run'] if isinstance(best_run_info['final_train_mse_of_best_val_run'], float) else 'N/A'}) with Params: nb={nb_val}, bs={bs_val}, hs={hs_val}, nh={nh_val}, vln={vln_val}, lr_h={lr_h}, scale={scale_base if current_overrides.get('use_inference_lr_scaling', False) else 'OFF'}, inf_steps={inf_steps}, warmup_steps={ws_val}, hidden_momentum={hm_val}, h_clip={h_clip}, w_clip={w_clip}, seed={seed_val} ***")
                                                            elif is_valid_run_best_train_mse and best_run_info["overall_best_val_mse"] == float('inf'):
                                                                if run_best_train_mse < best_run_info["run_best_train_mse_of_best_val_run"]:
                                                                    best_run_info["num_blocks"] = nb_val
                                                                    best_run_info["batch_size"] = bs_val
                                                                    best_run_info["hidden_size"] = hs_val
                                                                    best_run_info["num_heads"] = nh_val
                                                                    best_run_info["use_vode_state_layernorm"] = vln_val # New
                                                                    best_run_info["lr_weights"] = lr_w
                                                                    best_run_info["lr_hidden"] = lr_h
                                                                    best_run_info["inference_lr_scale_base"] = scale_base if current_overrides.get("use_inference_lr_scaling", False) else "N/A"
                                                                    best_run_info["inference_steps"] = inf_steps
                                                                    best_run_info["warmup_steps"] = ws_val
                                                                    best_run_info["h_grad_clip_norm"] = h_clip
                                                                    best_run_info["w_grad_clip_norm"] = w_clip
                                                                    best_run_info["hidden_momentum"] = hm_val
                                                                    best_run_info["seed"] = seed_val
                                                                    best_run_info["run_best_train_mse_of_best_val_run"] = run_best_train_mse
                                                                    best_run_info["final_train_mse_of_best_val_run"] = final_train_mse if is_valid_final_train_mse else "N/A"
                                                                    print(f"*** No valid validation MSEs yet. New best run_best_train_mse: {run_best_train_mse:.6f} (Final Train MSE: {best_run_info['final_train_mse_of_best_val_run'] if isinstance(best_run_info['final_train_mse_of_best_val_run'], float) else 'N/A'}) with Params: nb={nb_val}, bs={bs_val}, hs={hs_val}, nh={nh_val}, vln={vln_val}, lr_h={lr_h}, scale={scale_base if current_overrides.get('use_inference_lr_scaling', False) else 'OFF'}, inf_steps={inf_steps}, warmup_steps={ws_val}, hidden_momentum={hm_val}, h_clip={h_clip}, w_clip={w_clip}, seed={seed_val} ***")

                                                        except Exception as e:
                                                            print(f"!!!! Run {current_run} failed with Params: nb={nb_val}, bs={bs_val}, hs={hs_val}, nh={nh_val}, vln={vln_val}, lr_h={lr_h}, scale={scale_base if current_overrides.get('use_inference_lr_scaling', False) else 'OFF'}, inf_steps={inf_steps}, warmup_steps={ws_val}, hidden_momentum={hm_val}, h_clip={h_clip}, w_clip={w_clip}, seed={seed_val}. Error: {e} !!!!")
                                                            import traceback
                                                            traceback.print_exc()
                                                            all_run_results.append({
                                                                "run_number": current_run,
                                                                "num_blocks": nb_val,
                                                                "batch_size": bs_val,
                                                                "hidden_size": hs_val,
                                                                "num_heads": nh_val,
                                                                "use_vode_state_layernorm": vln_val, # New
                                                                "lr_weights": lr_w,
                                                                "lr_hidden": lr_h,
                                                                "inference_lr_scale_base": scale_base if current_overrides.get('use_inference_lr_scaling', False) else "N/A",
                                                                "inference_steps": inf_steps,
                                                                "warmup_steps": ws_val,
                                                                "h_grad_clip_norm": h_clip,
                                                                "w_grad_clip_norm": w_clip,
                                                                "hidden_momentum": hm_val,
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
        print(f"  num_blocks: {best_run_info['num_blocks']}")
        print(f"  batch_size: {best_run_info['batch_size']}")
        print(f"  hidden_size: {best_run_info['hidden_size']}")
        print(f"  num_heads: {best_run_info['num_heads']}")
        print(f"  use_vode_state_layernorm: {best_run_info['use_vode_state_layernorm']}") # New
        print(f"  peak_lr_weights: {best_run_info['lr_weights']} (Fixed)")
        print(f"  peak_lr_hidden: {best_run_info['lr_hidden']}")
        print(f"  inference_lr_scale_base: {best_run_info['inference_lr_scale_base']}")
        print(f"  inference_steps: {best_run_info['inference_steps']}")
        print(f"  warmup_steps: {best_run_info['warmup_steps']}")
        print(f"  hidden_momentum: {best_run_info['hidden_momentum']}")
        print(f"  h_grad_clip_norm: {best_run_info['h_grad_clip_norm']}")
        print(f"  w_grad_clip_norm: {best_run_info['w_grad_clip_norm']}")
        print(f"  seed: {best_run_info['seed']}")
        print(f"  Achieved Overall Best Validation MSE: {best_run_info['overall_best_val_mse']:.6f}")
        print(f"  Run Best Train MSE of Best Val Run: {best_run_info['run_best_train_mse_of_best_val_run'] if isinstance(best_run_info['run_best_train_mse_of_best_val_run'], float) else 'N/A'}")
        print(f"  Final Train MSE of Best Val Run: {best_run_info['final_train_mse_of_best_val_run'] if isinstance(best_run_info['final_train_mse_of_best_val_run'], float) else 'N/A'}")
    else:
        print(f"No successful runs completed or no runs achieved a valid MSE.")

    log_file_base_name = base_config_to_use
    # New: Define and create the target directory for hyperparameter search results
    results_dir = "../results/hyperparam_search_results"
    os.makedirs(results_dir, exist_ok=True)
    log_file_path = f"{results_dir}/hyperparam_search_results_{log_file_base_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"


    with open(log_file_path, 'w') as f:
        f.write(f"Hyperparameter Search Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Base Config for non-searched params: {base_config_to_use}\n")
        f.write(f"Fixed Overrides (excluding searched params):\n")

        searched_keys = [
            'num_blocks', 'batch_size', 'hidden_size', 'num_heads', 
            'use_vode_state_layernorm', # New
            'peak_lr_hidden', 'inference_lr_scale_base', 'inference_steps',
            'warmup_steps',
            'h_grad_clip_norm', 'w_grad_clip_norm', 'hidden_momentum', 'seed'
        ]
        
        fixed_to_print = {k: v for k, v in fixed_overrides.items() if k not in searched_keys}
        if 'use_inference_lr_scaling' in fixed_overrides:
             fixed_to_print['use_inference_lr_scaling'] = fixed_overrides['use_inference_lr_scaling']

        for key, value in fixed_to_print.items():
            f.write(f"  {key}: {value}\n")
        f.write("---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n")
        
        header_parts = [
            "Run", "NB", "BS", "HS", "NH", "VodeLN", "LR Hid", "Scale", "Inf Steps", "Warmup", "Hid Mom", "H Clip", "W Clip", "Seed",
            "Best Val MSE", "Run Best Train MSE", "Final Train MSE", "Status", "W&B Run Name"
        ]
        col_widths = {
            "Run": 3, "NB": 2, "BS": 4, "HS": 4, "NH": 2, "VodeLN": 6, "LR Hid": 9, "Scale": 7, "Inf Steps": 9, "Warmup": 6,
            "Hid Mom": 7, "H Clip": 6, "W Clip": 6, "Seed": 4,
            "Best Val MSE": 14, "Run Best Train MSE": 18 , "Final Train MSE": 15, "Status": 8, "W&B Run Name": 60 # Increased W&B name width
        }
        
        header_str = " | ".join([f"{h:<{col_widths[h]}}" for h in header_parts])
        f.write(header_str + "\n")
        f.write("---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n")
        
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
                if stop_reason == 'Validation': status_str = "(ES-Val)"
                elif stop_reason is None: status_str = "(Done)"
            elif stop_reason == 'NaN' and result['best_val_mse'] == 'Failed or Invalid':
                best_val_mse_str = "NaN-Val"; status_str = "(ES-NaN)"
            elif stop_reason == 'Exception': best_val_mse_str = "Exception"; status_str = "(Fail)"
            else: best_val_mse_str = str(best_val_mse_val); status_str = "(ValFail)"

            if isinstance(run_best_train_mse_val, (int, float)) and not np.isnan(run_best_train_mse_val) and run_best_train_mse_val != float('inf'):
                run_best_train_mse_str = f"{run_best_train_mse_val:.8f}"
            else: run_best_train_mse_str = str(run_best_train_mse_val)

            if isinstance(final_train_mse_val, (int, float)) and not np.isnan(final_train_mse_val) and final_train_mse_val != float('inf'):
                final_train_mse_str = f"{final_train_mse_val:.8f}"
                if not status_str:
                    if stop_reason == 'Validation': status_str = "(ES)"
                    elif stop_reason is None: status_str = "(Done)"
                elif stop_reason == 'NaN' and not status_str.endswith("(ES-NaN)"):
                     final_train_mse_str = "NaN";
                     if status_str: status_str += "/TrNaN"
                     else: status_str = "(TrNaN)"
                elif result.get('early_stop_reason') == 'Exception' and not status_str.endswith("(Fail)") :
                    final_train_mse_str = "Exception"
                    if status_str: status_str += "/TrFail"
                    else: status_str = "(TrFail)"
            else: 
                final_train_mse_str = str(final_train_mse_val) 
                if not status_str: status_str = "(TrFail)"


            run_name_str = result.get('wandb_run_name', 'N/A')
            h_clip_str = f"{result['h_grad_clip_norm']:.1f}" if result['h_grad_clip_norm'] is not None else "None"
            w_clip_str = f"{result['w_grad_clip_norm']:.1f}" if result['w_grad_clip_norm'] is not None else "None"
            inf_steps_log_str = str(result.get('inference_steps', 'N/A'))
            warmup_steps_log_str = str(result.get('warmup_steps', 'N/A'))
            hm_log_str = f"{result.get('hidden_momentum', 'N/A'):.2f}"
            seed_str = str(result.get('seed', 'N/A'))
            
            scale_log_str = result.get('inference_lr_scale_base', 'N/A')
            if isinstance(scale_log_str, float): scale_log_str = f"{scale_log_str:.2f}"

            nb_log_str = str(result.get('num_blocks', 'N/A'))
            bs_log_str = str(result.get('batch_size', 'N/A'))
            hs_log_str = str(result.get('hidden_size', 'N/A'))
            nh_log_str = str(result.get('num_heads', 'N/A'))
            vln_log_str = "ON" if result.get('use_vode_state_layernorm') else ("OFF" if result.get('use_vode_state_layernorm') is False else "N/A") #New

            log_line_parts = {
                "Run": result['run_number'],
                "NB": nb_log_str,
                "BS": bs_log_str,
                "HS": hs_log_str,
                "NH": nh_log_str,
                "VodeLN": vln_log_str, # New
                "LR Hid": f"{result['lr_hidden']:.3e}", 
                "Scale": scale_log_str,
                "Inf Steps": inf_steps_log_str,
                "Warmup": warmup_steps_log_str, 
                "Hid Mom": hm_log_str,
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
        f.write("---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n")
        if best_run_info["lr_hidden"] is not None:
            h_clip_best_str = f"{best_run_info['h_grad_clip_norm']:.1f}" if best_run_info['h_grad_clip_norm'] is not None else "None"
            w_clip_best_str = f"{best_run_info['w_grad_clip_norm']:.1f}" if best_run_info['w_grad_clip_norm'] is not None else "None"
            hm_best_str = f"{best_run_info['hidden_momentum']:.2f}" if best_run_info['hidden_momentum'] is not None else "None"
            vln_best_str = "ON" if best_run_info['use_vode_state_layernorm'] else ("OFF" if best_run_info['use_vode_state_layernorm'] is False else "N/A") # New
            
            best_scale_str = best_run_info['inference_lr_scale_base']
            if isinstance(best_scale_str, float): best_scale_str = f"{best_scale_str:.2f}"

            f.write(f"Best Params: num_blocks={best_run_info['num_blocks']}, batch_size={best_run_info['batch_size']}, hidden_size={best_run_info['hidden_size']}, num_heads={best_run_info['num_heads']}, use_vode_state_layernorm={vln_best_str}, peak_lr_hidden={best_run_info['lr_hidden']:.3e}, inference_lr_scale_base={best_scale_str}, inference_steps={best_run_info['inference_steps']}, warmup_steps={best_run_info['warmup_steps']}, hidden_momentum={hm_best_str}, h_grad_clip_norm={h_clip_best_str}, w_grad_clip_norm={w_clip_best_str}, seed={best_run_info['seed']}\n")
            f.write(f"Achieved Overall Best Validation MSE: {best_run_info['overall_best_val_mse']:.6f}\n")
            f.write(f"Run Best Train MSE of Best Val Run: {best_run_info['run_best_train_mse_of_best_val_run'] if isinstance(best_run_info['run_best_train_mse_of_best_val_run'], float) else 'N/A'}\n")
            f.write(f"Final Train MSE of Best Val Run: {best_run_info['final_train_mse_of_best_val_run'] if isinstance(best_run_info['final_train_mse_of_best_val_run'], float) else 'N/A'}\n")
            f.write(f" (lr_weights={best_run_info['lr_weights']:.3e} fixed)\n")
    print(f"Search results logged to: {log_file_path}")

if __name__ == "__main__":
    perform_hyperparameter_search()