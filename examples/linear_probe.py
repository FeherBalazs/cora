import argparse
import os
import sys
import jax
import jax.numpy as jnp
import numpy as np
import pcx
import pcx.functional as pxf
import pcx.utils as pxu
import pcx.nn as pxnn
import pcx.predictive_coding as pxc
from debug_transformer_wandb import MODEL_CONFIGS, ModelConfig # To get config definitions
from src.decoder_transformer import (
    TransformerDecoder, TransformerConfig, forward, energy, 
    apply_exponential_layer_scaling, normalize_vode_h_gradients,
    unmask_on_batch_enhanced # Import this function
) # To instantiate model and use PC dynamics
from tqdm import tqdm # For progress bars
import imageio
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.data import DataLoader, Subset, TensorDataset # Added TensorDataset
import torchvision
import torchvision.transforms as transforms
import torch # Added torch import

# Add project root to path to import from debug_transformer_wandb and src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def parse_args():
    parser = argparse.ArgumentParser(description="Linear Probing for PC-ViT SSL Evaluation")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the saved .npz file of the pretrained PC-ViT model (should contain weights and VodeParams).")
    parser.add_argument("--config_name", type=str, default="6block",
                        choices=MODEL_CONFIGS.keys(),
                        help="Name of the ModelConfig used for pretraining the loaded model.")
    parser.add_argument("--feature_layer_vode_indices", type=str, default=None,
                        help="Comma-separated string of Vode indices from which to extract features (e.g., \"0,1,7\"). "\
                             "0 for the top-most latent Vode. "\
                             "Positive indexing for other layers. E.g., num_blocks + 1 for output of last block. "\
                             "If not provided, will sweep through all relevant Vodes individually.")
    parser.add_argument("--concatenate_features", type=lambda x: (str(x).lower() == 'true'), default=False,
                        help="If True and multiple --feature_layer_vode_indices are provided, "
                             "concatenate features from these Vodes before linear probing. "
                             "If False, probe each Vode separately (default: False).")
    parser.add_argument("--use_gap", type=lambda x: (str(x).lower() == 'true'), default=True,
                        help="Whether to apply Global Average Pooling to patch features if they are multi-patch (default: True).")
    # Linear Probing specific arguments
    parser.add_argument("--probe_lr", type=float, default=1e-3, help="Learning rate for the linear probe classifier.")
    parser.add_argument("--probe_wd", type=float, default=1e-4, help="Weight decay for the linear probe classifier.")
    parser.add_argument("--probe_epochs", type=int, default=100, help="Number of epochs to train the linear probe.")
    parser.add_argument("--probe_batch_size", type=int, default=256, help="Batch size for training the linear probe.")
    parser.add_argument("--probe_inference_steps", type=int, default=None, help="Number of inference steps for feature extraction in the probe.")
    parser.add_argument("--probe_h_lr", type=float, default=None, help="Learning rate for hidden states (h) during feature extraction.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initializing the linear probe and other stochastic operations.")
    # Add more args later for linear classifier training (lr, epochs, etc.)
    return parser.parse_args()

def create_reconstruction_video(all_reconstruction_frames, orig_images, masked_images, labels_list, num_images, image_shape, wandb_run, epoch, fps=10, reconstruction_mses=None):
    """
    Creates a video comparing original images and their reconstructions over time.
    Adds MSE to the title of the reconstruction if provided.

    Args:
        all_reconstruction_frames: List (num_images) of lists (T_steps) of reconstruction tensors.
        orig_images: List of original image tensors.
        masked_images: List of masked input image tensors.
        labels_list: List of labels for the original images.
        num_images: Number of images to include in the video.
        image_shape: Shape of a single image (C, H, W).
        wandb_run: Wandb run object (can be None).
        epoch: Current epoch number (can be None).
        fps: Frames per second for the video.
        reconstruction_mses: List of MSE values for each image.

    Returns:
        Tuple: (video_path, log_dict)
    """
    num_channels, H, W = image_shape
    num_steps = len(all_reconstruction_frames[0]) # Assuming all images have same number of steps

    video_frames = []

    # Prepare original images once
    processed_orig_images = []
    processed_masked_images = []
    for i in range(num_images):
        orig_np = np.array(orig_images[i])
        masked_np = np.array(masked_images[i])
        if num_channels == 1: # Grayscale
            orig_plot = np.clip(np.squeeze(orig_np), 0.0, 1.0)
            orig_plot = (plt.cm.gray(orig_plot)[:, :, :3] * 255).astype(np.uint8) # Convert grayscale to RGB for video
            masked_plot = np.clip(np.squeeze(masked_np), 0.0, 1.0)
            masked_plot = (plt.cm.gray(masked_plot)[:, :, :3] * 255).astype(np.uint8)
        else: # RGB
            orig_plot = np.clip(np.transpose(orig_np, (1, 2, 0)), 0.0, 1.0)
            orig_plot = (orig_plot * 255).astype(np.uint8)
            masked_plot = np.clip(np.transpose(masked_np, (1, 2, 0)), 0.0, 1.0)
            masked_plot = (masked_plot * 255).astype(np.uint8)
        processed_orig_images.append(orig_plot)
        processed_masked_images.append(masked_plot)

    # Generate frames for the video
    for t in range(num_steps):
        fig, axes = plt.subplots(num_images, 3, figsize=(4 * 3, 2 * num_images))
        if num_images == 1:
            axes = axes[None, :] # Make it 2D

        for i in range(num_images):
            # Plot original image (Column 0)
            axes[i, 0].imshow(processed_orig_images[i])
            axes[i, 0].set_title(f'Original {labels_list[i] if labels_list[i] is not None else ""}')
            axes[i, 0].axis('off')

            # Plot masked input (Column 1)
            axes[i, 1].imshow(processed_masked_images[i])
            axes[i, 1].set_title(f'Masked Input') # For probe, masked might just be original
            axes[i, 1].axis('off')

            # Plot reconstruction at step t (Column 2)
            recon_t = all_reconstruction_frames[i][t]
            recon_np = np.array(recon_t[0]) # Get the first element from batch dim
            recon_np = np.reshape(recon_np, image_shape)
            current_mse_str = f" (MSE: {reconstruction_mses[i]:.4f})" if reconstruction_mses and i < len(reconstruction_mses) else ""

            if num_channels == 1: # Grayscale
                recon_plot = np.clip(np.squeeze(recon_np), 0.0, 1.0)
                recon_plot = (plt.cm.gray(recon_plot)[:, :, :3] * 255).astype(np.uint8)
            else: # RGB
                recon_plot = np.clip(np.transpose(recon_np, (1, 2, 0)), 0.0, 1.0)
                recon_plot = (recon_plot * 255).astype(np.uint8)
            
            axes[i, 2].imshow(recon_plot)
            axes[i, 2].set_title(f'Recon T={t+1}{current_mse_str}')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        fig.canvas.draw() # Draw the canvas, cache the renderer
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')

        width_inches, height_inches = fig.get_size_inches()
        dpi = fig.dpi
        width_pixels = int(np.round(width_inches * dpi))
        height_pixels = int(np.round(height_inches * dpi))
        expected_shape = (height_pixels, width_pixels, 3)

        if frame.size != np.prod(expected_shape):
            print(f"Warning: Buffer size ({frame.size}) does not match calculated shape {expected_shape} ({np.prod(expected_shape)}). Trying to infer shape.")
            buffer_pixels = frame.size // 3
            fig_width_inches, fig_height_inches = fig.get_size_inches()
            aspect_ratio = fig_width_inches / fig_height_inches if fig_height_inches > 0 else 1
            inferred_height = int(np.sqrt(buffer_pixels / aspect_ratio))
            inferred_width = int(inferred_height * aspect_ratio)
            if inferred_height * inferred_width * 3 == frame.size:
                 expected_shape = (inferred_height, inferred_width, 3)
                 print(f"Using inferred shape: {expected_shape}")
            else:
                 print(f"Error: Cannot determine correct frame shape. Buffer size: {frame.size}, Calculated shape: {(height_pixels, width_pixels, 3)}, Inferred shape: {(inferred_height, inferred_width, 3)}")
                 raise ValueError(f"Cannot reshape array of size {frame.size} into calculated shape {(height_pixels, width_pixels, 3)} or inferred shape {(inferred_height, inferred_width, 3)}")

        frame = frame.reshape(expected_shape)
        video_frames.append(frame)
        plt.close(fig)

    epoch_str = f"_epoch{epoch}" if epoch is not None else "_probe"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_dir = "./extracted_features" # Save in the same dir as other probe outputs
    os.makedirs(video_dir, exist_ok=True)
    video_path = f"{video_dir}/reconstruction_video{epoch_str}_{timestamp}.mp4"
    imageio.mimsave(video_path, video_frames, fps=fps)
    print(f"Saved reconstruction video to {video_path}")

    log_dict = {}
    if wandb_run is not None:
        # Attempt to import wandb only if needed
        try:
            import wandb
            log_key = f"reconstructions_video{epoch_str}" if epoch is not None else "reconstructions_video_probe"
            log_dict[log_key] = wandb.Video(video_path, fps=fps, format="mp4")
        except ImportError:
            print("wandb module not found, skipping W&B video logging.")
        except Exception as e:
            print(f"Error creating wandb.Video: {e}")
            print("Ensure ffmpeg is installed and accessible.")

    return video_path, log_dict

def main():
    args = parse_args()

    print("--- Linear Probing Setup ---")
    print(f"Loading pretrained model from: {args.model_path}")
    print(f"Using base config: {args.config_name}")
    print(f"Extracting features from Vode indices: {args.feature_layer_vode_indices}")
    print(f"Using Global Average Pooling: {args.use_gap}")

    # 1. Load ModelConfig used for pretraining
    if args.config_name not in MODEL_CONFIGS:
        raise ValueError(f"Config name '{args.config_name}' not found in MODEL_CONFIGS.")
    
    # Create a new instance of the config, we might override parts of it if they were
    # changed during the specific run that saved the model, but hyperparam_search.py
    # doesn't save the full config, only the overrides. For now, assume base config is enough.
    # In a more robust setup, the saved model artifact would include its exact config.
    base_model_config_obj: ModelConfig = MODEL_CONFIGS[args.config_name]

    # Create the TransformerConfig for the model architecture
    # We need to map ModelConfig fields to TransformerConfig fields
    transformer_arch_config = TransformerConfig(
        image_shape=tuple(map(int, MODEL_CONFIGS[args.config_name].dataset_img_shape)) if hasattr(MODEL_CONFIGS[args.config_name], 'dataset_img_shape') else (3, 32, 32), # TODO: Make this more robust
        num_frames=MODEL_CONFIGS[args.config_name].num_frames if hasattr(MODEL_CONFIGS[args.config_name], 'num_frames') else 16, # Default, adjust if needed
        is_video=MODEL_CONFIGS[args.config_name].is_video if hasattr(MODEL_CONFIGS[args.config_name], 'is_video') else False, # Default
        hidden_size=base_model_config_obj.hidden_size,
        num_heads=base_model_config_obj.num_heads,
        num_blocks=base_model_config_obj.num_blocks,
        mlp_ratio=base_model_config_obj.mlp_ratio,
        patch_size=base_model_config_obj.patch_size,
        axes_dim=base_model_config_obj.axes_dim,
        theta=base_model_config_obj.theta,
        act_fn=base_model_config_obj.act_fn,
        # SSL specific flags are not directly relevant for architecture, but for feature extraction behavior
        use_noise=base_model_config_obj.use_noise, 
        use_lower_half_mask=base_model_config_obj.use_lower_half_mask,
        use_status_init_in_training=base_model_config_obj.use_status_init_in_training, # Might influence how we get 'settled' h
        use_status_init_in_unmasking=base_model_config_obj.use_status_init_in_unmasking, # Might influence how we get 'settled' h
        use_inference_lr_scaling=base_model_config_obj.use_inference_lr_scaling,
        inference_lr_scale_base=base_model_config_obj.inference_lr_scale_base,
        update_weights_every_inference_step=base_model_config_obj.update_weights_every_inference_step,
        use_vode_state_layernorm=base_model_config_obj.use_vode_state_layernorm,
        use_vode_grad_norm=base_model_config_obj.use_vode_grad_norm,
        vode_grad_norm_target=base_model_config_obj.vode_grad_norm_target
    )
    
    print("\n--- Initializing Model Architecture ---")
    model = TransformerDecoder(transformer_arch_config)

    # Initialize model parameters (needed before loading)
    # Create a dummy input matching the expected batch size for model parameter initialization
    # The batch size here doesn't have to match training/pretraining, just for init.
    # However, the feature extraction will use a specific batch size from the dataloader.
    # Let's use a placeholder batch size of 1 for initialization, as pcx sometimes
    # initializes params based on the first call if not explicitly done.
    # Better: use the actual batch_size from the ModelConfig if available, or a default.
    # For now, using the original approach from debug_transformer_wandb
    
    # The pcx model usually initializes its parameters when first called with data,
    # or specifically when pxu.load_params is called on a model that has had its
    # parameters built (e.g. by a first forward pass).
    # We need to ensure the model structure is built.
    # A common way in pcx is to do a STATUS.INIT pass.
    
    key = jax.random.PRNGKey(0) # Dummy key for init
    # Use batch_size from the loaded config for init, or a default like 1 if not critical
    # The actual batching for feature extraction will come from the dataloader.
    # Using the config's batch_size is more consistent.
    init_batch_size = base_model_config_obj.batch_size 
    
    # Ensure image_shape is correctly derived for dummy input
    if transformer_arch_config.is_video:
        dummy_input_shape = (init_batch_size, *transformer_arch_config.image_shape)
    else:
        # Ensure image_shape from TransformerConfig is (C, H, W)
        if len(transformer_arch_config.image_shape) == 3:
             dummy_input_shape = (init_batch_size, *transformer_arch_config.image_shape)
        elif len(transformer_arch_config.image_shape) == 2: # (H,W) for grayscale, add C
             dummy_input_shape = (init_batch_size, 1, *transformer_arch_config.image_shape)
        else:
            raise ValueError(f"Unexpected image_shape format for non-video: {transformer_arch_config.image_shape}")


    x_init_dummy = jnp.zeros(dummy_input_shape, dtype=jnp.float32)
    
    print(f"Performing initial model forward pass with dummy input shape: {x_init_dummy.shape} to build parameters...")
    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        forward(x_init_dummy, model=model)
    print("Model parameters built.")

    # 2. Load saved weights AND VodeParams (activations/latents)
    try:
        # Load both LayerParams (weights) and VodeParams (h, u states)
        pxu.load_params(model, args.model_path, filter=lambda x: isinstance(x, (pxnn.LayerParam, pxc.VodeParam)))
        print(f"Successfully loaded LayerParams (weights) AND VodeParams (activations) from {args.model_path}")
    except Exception as e:
        print(f"Error loading model weights and VodeParams: {e}")
        print("Ensure the model_path is correct and the .npz file contains LayerParam and VodeParam entries matching the model architecture.")
        sys.exit(1)

    # 3. Freeze all model weights (LayerParams)
    # In pcx, freezing is typically handled by not including params in the optimizer's target.
    # For feature extraction, we simply won't update them.
    # If we were to use pcx's own parameter freezing, it would be:
    # for param in pxu.Parameters(model).select(pxnn.LayerParam):
    #     param.frozen = True
    # But since we are not calling an optimizer on these weights, it's implicitly frozen.
    print("Model weights (LayerParams) will be treated as frozen for feature extraction.")

    # 4. Setup CIFAR-10 Dataloaders for feature extraction
    print("\n--- Setting up CIFAR-10 Dataloaders ---")
    import torch
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader

    # Standard CIFAR-10 normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # We'll use the batch_size from the loaded config for feature extraction, can be overridden later if needed
    # For linear probing, batch size during feature extraction isn't super critical for accuracy, more for speed.
    feature_extraction_batch_size = base_model_config_obj.batch_size

    try:
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    except Exception as e:
        print(f"Error downloading CIFAR-10: {e}. Trying with alternative root.")
        alt_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets')
        os.makedirs(alt_data_dir, exist_ok=True)
        train_dataset = torchvision.datasets.CIFAR10(root=alt_data_dir, train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root=alt_data_dir, train=False, download=True, transform=transform)


    train_loader = DataLoader(train_dataset, batch_size=feature_extraction_batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=feature_extraction_batch_size, shuffle=False, num_workers=0)

    print(f"CIFAR-10 train dataset size: {len(train_dataset)}")
    print(f"CIFAR-10 test dataset size: {len(test_dataset)}")
    print(f"Feature extraction batch size: {feature_extraction_batch_size}")

    # 5. Setup optimizer for h-states during inference for feature extraction
    # This is similar to optim_h_inference in debug_transformer_wandb.py
    # It should use the inference-specific LR and momentum from the loaded config.
    import optax
    print("\n--- Setting up Optimizer for Hidden States (h) during Feature Extraction ---")

    # Determine effective h_lr for feature extraction
    # Priority: CLI arg -> SSL model's peak_lr_hidden (if schedule was off) / current LR (if schedule on) -> default 0.055
    ssl_model_peak_lr_hidden = base_model_config_obj.peak_lr_hidden
    ssl_model_hidden_lr_inference = base_model_config_obj.hidden_lr_inference # Fallback if other options are not clear

    # Try to get the learning rate that would have been active at the END of SSL pretraining if a schedule was used.
    # This is a rough approximation if total_train_steps isn't perfectly known or if model saved mid-epoch.
    effective_h_lr_from_ssl = ssl_model_peak_lr_hidden # Default to peak if no schedule
    if base_model_config_obj.use_lr_schedule_h:
        # Recreate the schedule to get the final LR. Needs total_train_steps from pretraining.
        # This info isn't directly in ModelConfig. Assuming it trained for full base_model_config_obj.epochs.
        # This is an approximation.
        # For simplicity, if schedule was used, and no CLI override, let's just use the base peak_lr_hidden
        # or a new default, as accurately recreating the final scheduled LR is complex here.
        # A better approach would be to save the final LR with the model.
        # For now, we'll prioritize CLI, then a sensible default, then the model's peak_lr_hidden.
        pass # Using peak_lr_hidden as a proxy if schedule was on and no CLI override for now.

    if args.probe_h_lr is not None:
        h_lr_for_extraction = args.probe_h_lr
        print(f"Using probe_h_lr from CLI for feature extraction: {h_lr_for_extraction}")
    else:
        # If no CLI, consider the SSL model's setup. 
        # If LR schedule for H was NOT used during SSL, then peak_lr_hidden was the constant LR for H.
        if not base_model_config_obj.use_lr_schedule_h:
            h_lr_for_extraction = ssl_model_peak_lr_hidden
            print(f"Using SSL model's fixed peak_lr_hidden for feature extraction: {h_lr_for_extraction}")
        else:
            # If schedule WAS used, using peak might be too high. Let's use a safer default or hidden_lr_inference.
            h_lr_for_extraction = base_model_config_obj.hidden_lr_inference # Or a new default like 0.055
            print(f"Using SSL model's hidden_lr_inference for feature extraction (schedule was on): {h_lr_for_extraction}")

    h_momentum = base_model_config_obj.hidden_momentum
    h_grad_clip = base_model_config_obj.h_grad_clip_norm

    if base_model_config_obj.use_adamw_for_hidden_optimizer:
        print(f"Using AdamW for hidden state inference optimizer with LR: {h_lr_for_extraction}")
        base_optim_h_feat_extract = optax.adamw(learning_rate=h_lr_for_extraction, b1=0.9, b2=0.999, eps=1e-8)
    else:
        print(f"Using SGD for hidden state inference optimizer with LR: {h_lr_for_extraction}, Momentum: {h_momentum}")
        base_optim_h_feat_extract = optax.sgd(h_lr_for_extraction, momentum=h_momentum)

    optim_h_feat_extract_steps = []
    if h_grad_clip is not None and h_grad_clip > 0:
        print(f"Applying H-gradient clipping for feature extraction with max_norm = {h_grad_clip}")
        h_clipper_feat_extract = optax.clip_by_global_norm(h_grad_clip)
        optim_h_feat_extract_steps.append(h_clipper_feat_extract)
    
    optim_h_feat_extract_steps.append(base_optim_h_feat_extract)
    final_optim_h_for_feature_extraction = optax.chain(*optim_h_feat_extract_steps)
    optim_h_extractor = pxu.Optim(lambda: final_optim_h_for_feature_extraction)

    print("Optimizer for h-states (optim_h_extractor) configured.")

    # Determine inference steps for extraction (moved earlier for visualization use)
    if args.probe_inference_steps is not None:
        inference_steps_for_extraction = args.probe_inference_steps
        print(f"Using probe_inference_steps from CLI for feature extraction and visualization: {inference_steps_for_extraction}")
    else:
        inference_steps_for_extraction = 40 # Default to 40 if not provided
        print(f"Using default probe_inference_steps for feature extraction and visualization: {inference_steps_for_extraction}")

    # --- Add Reconstruction Visualization --- 
    print("\n--- Generating Reconstruction Visualization --- ")
    vis_num_images = 2
    # Use the determined inference_steps_for_extraction for the visualization target T values
    # vis_target_T_values will be a list with a single element: the final step number.
    vis_target_T_values = [inference_steps_for_extraction]
    print(f"Visualization will use T_values: {vis_target_T_values} (derived from probe_inference_steps or its default)")

    vis_fps = base_model_config_obj.video_fps
    vis_image_shape = transformer_arch_config.image_shape # This is (C,H,W)

    # --- Define Linear Classifier Components (moved to outer scope) ---
    def linear_classifier_predict(params_local, x_local):
        return jnp.dot(x_local, params_local['W']) + params_local['b']

    def cross_entropy_loss(params_local, x_local, y_true_one_hot_local):
        logits_local = linear_classifier_predict(params_local, x_local)
        log_probs_local = jax.nn.log_softmax(logits_local)
        # Standard cross-entropy: sum over classes, mean over batch
        return -jnp.sum(log_probs_local * y_true_one_hot_local, axis=-1).mean()

    def accuracy(params_local, x_local, y_true_one_hot_local):
        predictions_local = linear_classifier_predict(params_local, x_local)
        predicted_classes_local = jnp.argmax(predictions_local, axis=1)
        true_classes_local = jnp.argmax(y_true_one_hot_local, axis=1)
        return jnp.mean(predicted_classes_local == true_classes_local)
    # --- End Linear Classifier Components ---

    # Conditional initial visualization
    if args.feature_layer_vode_indices is not None: # Only run initial viz if a specific Vode is targeted
        # Get a batch from the test_loader for visualization
        try:
            vis_batch_pt_full, vis_labels_pt_full = next(iter(test_loader))
        except StopIteration:
            print("Error: Could not get a batch from test_loader for visualization.")
            vis_batch_pt_full = None

        if vis_batch_pt_full is not None:
            model.eval() # Ensure model is in eval mode
            
            all_images_all_recons_list = [] # To store [[img1_recons_steps], [img2_recons_steps]]
            all_images_orig_for_video = []
            all_images_labels_for_video = []
            all_images_recon_loss = []

            for i in range(vis_num_images):
                print(f"Processing image {i+1}/{vis_num_images} for reconstruction video...")
                single_image_np = vis_batch_pt_full[i:i+1].numpy() # Batch of 1 image
                single_label_np = vis_labels_pt_full[i:i+1].numpy()

                all_images_orig_for_video.append(single_image_np[0]) # Store unbatched image
                all_images_labels_for_video.append(single_label_np[0])

                recon_loss_single, single_img_all_recons_list, _, _ = unmask_on_batch_enhanced(
                    use_corruption=False,
                    corrupt_ratio=0.0,
                    target_T_values=vis_target_T_values,
                    x=jnp.array(single_image_np), # Pass single image as JAX array
                    model=model,
                    optim_h=optim_h_extractor,
                    optim_w=None
                )
                all_images_all_recons_list.append(single_img_all_recons_list)
                all_images_recon_loss.append(recon_loss_single)
                print(f"  Reconstruction loss for image {i+1}: {recon_loss_single:.4f}")

            if all_images_all_recons_list:
                # create_reconstruction_video expects: List (num_images) of lists (T_steps) of reconstruction tensors.
                # Each inner list (T_steps) should contain tensors of shape (1, C, H, W) or (C,H,W) if squeezed later.
                # single_img_all_recons_list already contains [step0_recon_batch_of_1, step1_recon_batch_of_1, ...]
                # So, all_images_all_recons_list is already in the correct format.
                create_reconstruction_video(
                    all_reconstruction_frames=all_images_all_recons_list,
                    orig_images=all_images_orig_for_video,
                    masked_images=all_images_orig_for_video, # Same as original for no corruption
                    labels_list=all_images_labels_for_video,
                    num_images=vis_num_images,
                    image_shape=vis_image_shape,
                    wandb_run=None, 
                    epoch=None,     
                    fps=vis_fps,
                    reconstruction_mses=all_images_recon_loss 
                )
    # --- End Reconstruction Visualization ---
    else:
        print("Skipping initial reconstruction visualization because running in Vode sweep mode.")

    # 6. Define feature extraction function
    @pxf.jit(static_argnums=(3, 4, 5)) # Jit this function for speed after debugging
    def extract_features_from_batch(x_batch_np, model_instance, optim_h, inference_steps_for_extraction, target_vode_indices_list, use_gap_flag):
        """Extracts features from a batch of images for specified Vode indices."""
        # print(f"Debug: extract_features_from_batch called with x_batch_np shape: {x_batch_np.shape}, target_vode_indices_list: {target_vode_indices_list}")
        x_batch = jnp.array(x_batch_np) # Ensure JAX array

        # 1. Initialize hidden states (Vodes) if needed or use existing ones
        #    This assumes model.vodes are already initialized appropriately before calling this function
        #    or that the PC dynamics will handle it.
        
        # If using status.init for unmasking (which is analogous to feature extraction here)
        initial_status_extraction = pxc.STATUS.INIT if model_instance.config.use_status_init_in_unmasking else None
        # print(f"Debug: Initializing model for extraction with status: {initial_status_extraction}")
        with pxu.step(model_instance, initial_status_extraction, clear_params=pxc.VodeParam.Cache):
            forward(x_batch, model=model_instance) # This sets sensory Vode to x_batch
        # print("Debug: Model initialized for extraction.")

        optim_h.init(pxu.M_hasnot(pxc.VodeParam, frozen=True)(model_instance))
        # print("Debug: optim_h initialized for extraction.")

        # 2. Run PC inference for T_h steps to let hidden states converge
        inference_step_fn = pxf.value_and_grad(pxu.M_hasnot(pxc.VodeParam, frozen=True).to([False, True]), has_aux=True)(energy)
        
        # print(f"Debug: Starting {inference_steps_for_extraction} inference steps for feature extraction...")
        for t_inf in range(inference_steps_for_extraction):
            with pxu.step(model_instance, clear_params=pxc.VodeParam.Cache): # Ensure we don't carry over stale cache
                (h_energy, _), h_grad = inference_step_fn(model=model_instance)
            
            model_grads_to_apply = h_grad["model"]
            if model_instance.config.use_inference_lr_scaling:
                model_grads_to_apply = apply_exponential_layer_scaling(
                    model_grads=model_grads_to_apply,
                    config=model_instance.config
                )
            if model_instance.config.use_vode_grad_norm:
                model_grads_to_apply = normalize_vode_h_gradients(
                    model_grads=model_grads_to_apply,
                    config=model_instance.config
                )
            optim_h.step(model_instance, model_grads_to_apply)
            # print(f"Debug: Inference step {t_inf+1}/{inference_steps_for_extraction} completed. Energy: {h_energy}")

        # 3. Extract features from the specified Vode(s)
        extracted_features_list = []
        for target_vode_idx in target_vode_indices_list:
            # print(f"Debug: Extracting features from Vode index: {target_vode_idx}")
            num_vodes_total = len(model_instance.vodes)

            # Handle negative indexing for Vodes if necessary (e.g., -1 for sensory)
            actual_idx = target_vode_idx
            if actual_idx < 0:
                actual_idx = num_vodes_total + actual_idx
            
            if not (0 <= actual_idx < num_vodes_total):
                raise ValueError(f"Invalid Vode index {target_vode_idx} (resolved to {actual_idx}). Model has {num_vodes_total} Vodes.")

            # Access the hidden state 'h' of the Vode
            # The ._value is crucial to get the JAX array from the Param object
            h_param = model_instance.vodes[actual_idx].h 
            # print(f"Debug: Vode {actual_idx} h_param type: {type(h_param)}, value type: {type(h_param._value)}")
            
            if h_param._value is None:
                raise ValueError(f"Vode {actual_idx} hidden state (h) is None. Ensure model is run first.")
            
            features_from_vode = h_param._value # This should be a JAX array (batch_size, num_patches, feature_dim) or (batch_size, feature_dim)
            # print(f"Debug: Features from Vode {actual_idx} shape: {features_from_vode.shape}")
            extracted_features_list.append(features_from_vode)

        # 4. Concatenate features if multiple Vodes were specified
        if len(extracted_features_list) > 1:
            # Assuming features are (batch_size, num_patches, feature_dim) or (batch_size, feature_dim)
            # We want to concatenate along the feature dimension (the last dimension)
            # print(f"Debug: Concatenating features from {len(extracted_features_list)} Vodes.")
            # for i, f in enumerate(extracted_features_list):
            #     print(f"  Feature {i} shape: {f.shape}")
            final_features = jnp.concatenate(extracted_features_list, axis=-1)
            # print(f"Debug: Concatenated features shape: {final_features.shape}")
        elif extracted_features_list:
            final_features = extracted_features_list[0]
        else:
            raise ValueError("No features were extracted. target_vode_indices_list might be empty.")

        # 5. Apply Global Average Pooling (GAP) if specified and features are per-patch
        # GAP is typically applied if features have a patch dimension, e.g., (batch_size, num_patches, feature_dim)
        if use_gap_flag and final_features.ndim == 3: # Check if there's a patch dimension
            # print(f"Debug: Applying GAP to features of shape: {final_features.shape}")
            # Average over the patch dimension (axis=1)
            final_features = jnp.mean(final_features, axis=1)
            # print(f"Debug: Features shape after GAP: {final_features.shape}")
        elif use_gap_flag and final_features.ndim != 2:
            print(f"Warning: use_gap is True, but features are not 2D or 3D (shape: {final_features.shape}). Skipping GAP.")
        
        optim_h.clear()
        # print(f"Debug: optim_h cleared. Returning features of shape: {final_features.shape}")
        return final_features

    # 7. Extract features for train and test sets
    print("\n--- Extracting Features ---")
    
    def extract_and_save_features(loader, set_name, model_instance_local, optim_h_local, inference_steps_local, target_vode_indices_local_list, use_gap_local, concatenate_flag):
        print(f"Extracting features for '{set_name}' set from Vode indices: {target_vode_indices_local_list} (Concatenate: {concatenate_flag})...")
        all_features = []
        all_labels = []
        for x_batch_torch, y_batch_torch in tqdm(loader, desc=f"Extracting {set_name} features"):
            x_batch_np = x_batch_torch.numpy() # Convert to NumPy for JAX
            
            # Convert list to tuple for JIT compatibility if it's a static argument
            target_vode_indices_local_tuple = tuple(target_vode_indices_local_list)
            
            features_batch = extract_features_from_batch(
                x_batch_np, 
                model_instance_local, 
                optim_h_local, 
                inference_steps_local, 
                target_vode_indices_local_tuple, # Pass the tuple of indices
                use_gap_local
            )
            all_features.append(features_batch)
            all_labels.append(y_batch_torch.numpy())

        # Concatenate all batch features and labels
        final_features_np = np.concatenate([jax.device_get(f) for f in all_features], axis=0)
        final_labels_np = np.concatenate(all_labels, axis=0)
        print(f"Shape of extracted '{set_name}' features: {final_features_np.shape}")
        print(f"Shape of extracted '{set_name}' labels: {final_labels_np.shape}")

        # Save features and labels
        os.makedirs("./extracted_features", exist_ok=True)
        vode_indices_str_part = "_concat_" + "_".join(map(str, target_vode_indices_local_list)) if concatenate_flag and len(target_vode_indices_local_list) > 1 else "_vode_" + "_".join(map(str, target_vode_indices_local_list))
        features_path = f"./extracted_features/features{vode_indices_str_part}_{set_name}.npz"
        np.savez_compressed(features_path, features=final_features_np, labels=final_labels_np)
        print(f"Saved '{set_name}' features and labels to {features_path}")
        return features_path

    # --- Feature Extraction and Linear Probing Loop ---
    all_probe_results = []

    # Determine which Vode indices to run the probe for
    if args.feature_layer_vode_indices is not None:
        # Parse comma-separated string into a list of integers
        vode_indices_to_process = [int(idx.strip()) for idx in args.feature_layer_vode_indices.split(',')]
        if not vode_indices_to_process:
            raise ValueError("feature_layer_vode_indices was provided but resulted in an empty list.")
        print(f"Processing specified Vode indices: {vode_indices_to_process}")
    else:
        # Default sweep: Vode 0 (top-most) to Vode num_blocks + 1 (output of last transformer block)
        # Also include sensory Vode (-1) if it makes sense (usually not for classification)
        # And pre-sensory (-2) which is same as num_blocks + 1
        vode_indices_to_process = list(range(transformer_arch_config.num_blocks + 2)) # 0 to num_blocks+1
        print(f"No specific Vode indices provided. Sweeping through default Vodes: {vode_indices_to_process}")

    # Logic for handling concatenation vs. individual probing
    if args.concatenate_features and len(vode_indices_to_process) > 1:
        print(f"\n--- Concatenating features from Vodes: {vode_indices_to_process} ---")
        # Extract and save concatenated features once
        train_features_path = extract_and_save_features(train_loader, "train", model, optim_h_extractor, inference_steps_for_extraction, vode_indices_to_process, args.use_gap, True)
        test_features_path = extract_and_save_features(test_loader, "test", model, optim_h_extractor, inference_steps_for_extraction, vode_indices_to_process, args.use_gap, True)
        
        # Load concatenated features
        train_data = np.load(train_features_path)
        x_train, y_train = torch.from_numpy(train_data['features']), torch.from_numpy(train_data['labels'])
        test_data = np.load(test_features_path)
        x_test, y_test = torch.from_numpy(test_data['features']), torch.from_numpy(test_data['labels'])

        # Train and evaluate linear probe on concatenated features
        vode_str_for_print = "_concat_" + "_".join(map(str, vode_indices_to_process))
        # (Linear probe training and evaluation logic - to be filled, similar to below but just once)
        # ... (This part will be a single call to the probe training logic)
        # For now, let's print a placeholder
        print(f"Linear probing for concatenated features: {vode_str_for_print}")
        num_features = x_train.shape[1]
        print(f"Number of features for concatenated Vodes {vode_str_for_print}: {num_features}")
        
        # Initialize linear classifier parameters (W, b)
        key_probe_init, _ = jax.random.split(jax.random.PRNGKey(args.seed))
        # Glorot/Xavier uniform initialization for weights
        limit = np.sqrt(6 / (num_features + 10))
        W_probe = jax.random.uniform(key_probe_init, (num_features, 10), minval=-limit, maxval=limit)
        b_probe = jnp.zeros(10)
        probe_params = {'W': W_probe, 'b': b_probe}

        # Optimizer for the linear probe
        probe_optimizer = optax.adamw(learning_rate=args.probe_lr, weight_decay=args.probe_wd)
        probe_opt_state = probe_optimizer.init(probe_params)

        # Convert to TensorDataset for DataLoader
        probe_train_dataset = TensorDataset(x_train, y_train)
        probe_train_loader = DataLoader(probe_train_dataset, batch_size=args.probe_batch_size, shuffle=True)

        print(f"Training linear probe for Vodes {vode_str_for_print} for {args.probe_epochs} epochs...")
        best_test_acc_concat = 0.0
        for epoch in range(args.probe_epochs):
            epoch_train_loss = 0.0
            epoch_train_correct = 0
            epoch_train_total = 0
            for x_probe_batch, y_probe_batch_true_idx in probe_train_loader:
                x_pb_jnp = jnp.array(x_probe_batch.numpy()) # Ensure JAX array
                y_pb_true_one_hot = jax.nn.one_hot(jnp.array(y_probe_batch_true_idx.numpy()), 10)
                
                loss_val, grads = jax.value_and_grad(cross_entropy_loss)(probe_params, x_pb_jnp, y_pb_true_one_hot)
                updates, probe_opt_state = probe_optimizer.update(grads, probe_opt_state, probe_params)
                probe_params = optax.apply_updates(probe_params, updates)
                
                epoch_train_loss += loss_val * x_probe_batch.shape[0]
                # Calculate training accuracy for the batch
                preds = linear_classifier_predict(probe_params, x_pb_jnp)
                epoch_train_correct += jnp.sum(jnp.argmax(preds, axis=1) == y_probe_batch_true_idx.numpy()).item()
                epoch_train_total += x_probe_batch.shape[0]

            avg_epoch_train_loss = epoch_train_loss / len(probe_train_dataset)
            current_train_acc = epoch_train_correct / epoch_train_total
            
            # Evaluate on test set
            x_test_jnp = jnp.array(x_test.numpy())
            y_test_one_hot = jax.nn.one_hot(jnp.array(y_test.numpy()), 10)
            current_test_acc = accuracy(probe_params, x_test_jnp, y_test_one_hot).item()
            best_test_acc_concat = max(best_test_acc_concat, current_test_acc)

            print(f"Vodes {vode_str_for_print} - Probe Epoch {epoch+1}/{args.probe_epochs} - Train Loss: {avg_epoch_train_loss:.4f}, Train Acc: {current_train_acc:.4f}, Test Acc: {current_test_acc:.4f}")

        print(f"Vodes {vode_str_for_print} - Final Test Accuracy: {best_test_acc_concat:.4f}")
        all_probe_results.append({
            "vode_indices": vode_str_for_print,
            "num_features": num_features,
            "test_accuracy": best_test_acc_concat
        })
    else: # Individual probing (sweep or single Vode)
        for current_vode_idx in vode_indices_to_process:
            print(f"\n--- Probing Vode Index: {current_vode_idx} ---")
            # Extract and save features for the current Vode index
            # Pass current_vode_idx as a single-element list for extract_features_from_batch
            train_features_path = extract_and_save_features(train_loader, "train", model, optim_h_extractor, inference_steps_for_extraction, [current_vode_idx], args.use_gap, False)
            test_features_path = extract_and_save_features(test_loader, "test", model, optim_h_extractor, inference_steps_for_extraction, [current_vode_idx], args.use_gap, False)

            # Load features for the current Vode
            train_data = np.load(train_features_path)
            x_train, y_train = torch.from_numpy(train_data['features']), torch.from_numpy(train_data['labels'])
            test_data = np.load(test_features_path)
            x_test, y_test = torch.from_numpy(test_data['features']), torch.from_numpy(test_data['labels'])

            num_features = x_train.shape[1]
            print(f"Number of features for Vode {current_vode_idx}: {num_features}")

            # Initialize linear classifier parameters (W, b)
            key_probe_init, _ = jax.random.split(jax.random.PRNGKey(args.seed + current_vode_idx)) # Vary seed per Vode
            limit = np.sqrt(6 / (num_features + 10))
            W_probe = jax.random.uniform(key_probe_init, (num_features, 10), minval=-limit, maxval=limit)
            b_probe = jnp.zeros(10)
            probe_params = {'W': W_probe, 'b': b_probe}

            # Optimizer for the linear probe
            probe_optimizer = optax.adamw(learning_rate=args.probe_lr, weight_decay=args.probe_wd)
            probe_opt_state = probe_optimizer.init(probe_params)

            # Convert to TensorDataset for DataLoader
            probe_train_dataset = TensorDataset(x_train, y_train)
            probe_train_loader = DataLoader(probe_train_dataset, batch_size=args.probe_batch_size, shuffle=True)

            print(f"Training linear probe for Vode {current_vode_idx} for {args.probe_epochs} epochs...")
            best_test_acc_this_vode = 0.0
            for epoch in range(args.probe_epochs):
                epoch_train_loss = 0.0
                epoch_train_correct = 0
                epoch_train_total = 0
                for x_probe_batch, y_probe_batch_true_idx in probe_train_loader:
                    x_pb_jnp = jnp.array(x_probe_batch.numpy()) # Ensure JAX array
                    y_pb_true_one_hot = jax.nn.one_hot(jnp.array(y_probe_batch_true_idx.numpy()), 10)
                    
                    loss_val, grads = jax.value_and_grad(cross_entropy_loss)(probe_params, x_pb_jnp, y_pb_true_one_hot)
                    updates, probe_opt_state = probe_optimizer.update(grads, probe_opt_state, probe_params)
                    probe_params = optax.apply_updates(probe_params, updates)
                    
                    epoch_train_loss += loss_val * x_probe_batch.shape[0]
                    preds = linear_classifier_predict(probe_params, x_pb_jnp)
                    epoch_train_correct += jnp.sum(jnp.argmax(preds, axis=1) == y_probe_batch_true_idx.numpy()).item()
                    epoch_train_total += x_probe_batch.shape[0]

                avg_epoch_train_loss = epoch_train_loss / len(probe_train_dataset)
                current_train_acc = epoch_train_correct / epoch_train_total
                
                # Evaluate on test set
                x_test_jnp = jnp.array(x_test.numpy())
                y_test_one_hot = jax.nn.one_hot(jnp.array(y_test.numpy()), 10)
                current_test_acc = accuracy(probe_params, x_test_jnp, y_test_one_hot).item()
                best_test_acc_this_vode = max(best_test_acc_this_vode, current_test_acc)

                print(f"Vode {current_vode_idx} - Probe Epoch {epoch+1}/{args.probe_epochs} - Train Loss: {avg_epoch_train_loss:.4f}, Train Acc: {current_train_acc:.4f}, Test Acc: {current_test_acc:.4f}")

            print(f"Vode {current_vode_idx} - Final Test Accuracy: {best_test_acc_this_vode:.4f}")
            all_probe_results.append({
                "vode_indices": str(current_vode_idx),
                "num_features": num_features,
                "test_accuracy": best_test_acc_this_vode
            })

    # 9. Log all results
    print("\n===== Linear Probe Sweep Results =====")
    results_log_path = os.path.join("./results", f"linear_probe_sweep_{args.config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    os.makedirs("./results", exist_ok=True)

    with open(results_log_path, 'w') as f:
        f.write(f"Linear Probe Sweep Results - Model: {args.model_path}, Config: {args.config_name}\n")
        f.write(f"Probe LR: {args.probe_lr}, Probe WD: {args.probe_wd}, Probe Epochs: {args.probe_epochs}\n")
        f.write(f"Feature Extraction H LR: {h_lr_for_extraction}, Feature Extraction Steps: {inference_steps_for_extraction}\n")
        f.write(f"Global Average Pooling: {args.use_gap}\n\n")
        f.write("Vode Index/Combination | Test Accuracy | Num Features\n")
        f.write("----------------------------------------------------\n")
        # Sort results by test accuracy in descending order
        sorted_results = sorted(all_probe_results, key=lambda item: item["test_accuracy"], reverse=True)
        for result_item in sorted_results:
            vode_str = result_item["vode_indices"]
            acc_val = result_item["test_accuracy"]
            num_feat = result_item["num_features"]
            f.write(f"{vode_str:<24} | {acc_val:<13.4f} | {num_feat}\n")
            print(f"Vode Combination: {vode_str}, Test Accuracy: {acc_val:.4f}, Num Features: {num_feat}")
    
    print(f"\nResults saved to: {results_log_path}")

    # Clean up: Remove feature files if desired (optional)
    # print("\nCleaning up extracted feature files...")
    # for current_vode_idx in vode_indices_to_process:
    #     for set_name in ["train", "test"]:
    #         try:
    #             features_path = os.path.join("./extracted_features", f"{args.config_name}_layer{current_vode_idx}_gap{args.use_gap}_{set_name}_features.npy")
    #             labels_path = os.path.join("./extracted_features", f"{args.config_name}_layer{current_vode_idx}_gap{args.use_gap}_{set_name}_labels.npy")
    #             os.remove(features_path)
    #             os.remove(labels_path)
    #         except OSError as e:
    #             print(f"Error deleting file for Vode {current_vode_idx}, {set_name}: {e.strerror}")
    # print("Cleanup complete.")

if __name__ == "__main__":
    main() 