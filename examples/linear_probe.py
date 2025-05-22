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
    parser.add_argument("--feature_layer_vode_idx", type=int, default=None,
                        help="Index of the Vode from which to extract features. "
                             "0 for the top-most latent Vode. "
                             "Positive indexing for other layers. E.g., num_blocks + 1 for output of last block. "
                             "Negative indexing: -1 for sensory, -2 for pre-sensory/last block output. "
                             "If not provided, will sweep through all relevant Vodes.")
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
    print(f"Extracting features from Vode index: {args.feature_layer_vode_idx}")
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

    # Use hidden_lr_inference and hidden_momentum from the loaded config
    # These were part of ModelConfig, not TransformerConfig, so access via base_model_config_obj
    
    # Determine h_inference_lr for feature extraction
    if args.probe_h_lr is not None:
        h_inference_lr = args.probe_h_lr
        print(f"Using probe_h_lr from CLI for feature extraction: {h_inference_lr}")
    else:
        h_inference_lr = 0.055 # Default to 0.055 if not provided
        print(f"Using default h_inference_lr for feature extraction: {h_inference_lr}")
    
    h_momentum = base_model_config_obj.hidden_momentum
    h_grad_clip = base_model_config_obj.h_grad_clip_norm

    if base_model_config_obj.use_adamw_for_hidden_optimizer:
        print(f"Using AdamW for hidden state inference optimizer with LR: {h_inference_lr}")
        base_optim_h_feat_extract = optax.adamw(learning_rate=h_inference_lr, b1=0.9, b2=0.999, eps=1e-8)
    else:
        print(f"Using SGD for hidden state inference optimizer with LR: {h_inference_lr}, Momentum: {h_momentum}")
        base_optim_h_feat_extract = optax.sgd(h_inference_lr, momentum=h_momentum)

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

    # Conditional initial visualization
    if args.feature_layer_vode_idx is not None: # Only run initial viz if a specific Vode is targeted
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
    def extract_features_from_batch(x_batch_np, model_instance, optim_h, inference_steps_for_extraction, target_vode_idx, use_gap_flag):
        x_batch_jnp = jnp.array(x_batch_np)

        # A. Initialize optimizer state for the current model state (which includes loaded VodeParams)
        optim_h.clear()
        optim_h.init(pxu.M_hasnot(pxc.VodeParam, frozen=True)(model_instance)) # Target non-frozen VodeParams

        # B. Initial forward pass to set sensory Vode and allow initial propagation
        # Determine if STATUS.INIT should be used based on PRETRAINING config for UNMASKING (or training if unmasking not set)
        # For feature extraction, we usually want the model to adapt to the new input x_batch.
        # If the loaded VodeParams are a generic "prior", then STATUS.INIT might reset them too aggressively.
        # Let's assume for now we do NOT use STATUS.INIT, and instead let the loaded VodeParams be the starting point
        # and the inference steps will adapt them to x_batch.
        # The sensory Vode (vodes[-1]) will be set by the forward(x_batch_jnp, ...) call.
        
        with pxu.step(model_instance, clear_params=pxc.VodeParam.Cache): # No STATUS.INIT here
            model_instance.vodes[-1].h.frozen = False # Ensure sensory is not frozen for this initial setting
            forward(x_batch_jnp, model=model_instance) # This sets model.vodes[-1].h to x_batch_jnp
            model_instance.vodes[-1].h.frozen = True  # Re-freeze sensory after its set to the input

        # C. Run T_h inference steps to let latents converge for this batch
        # inference_steps_for_extraction is now passed directly
        
        # Define the energy function for gradient calculation (consistent with SSL pretraining)
        # This targets non-frozen VodeParams for gradient calculation.
        inference_step_fn_grad = pxf.value_and_grad(pxu.M_hasnot(pxc.VodeParam, frozen=True).to([False, True]), has_aux=True)(energy)

        for _ in range(inference_steps_for_extraction):
            with pxu.step(model_instance, clear_params=pxc.VodeParam.Cache):
                # Get energy and gradients for h (non-frozen VodeParams)
                (_, _), h_grad = inference_step_fn_grad(model=model_instance)
            
            model_grads_to_apply = h_grad["model"] # Grads for VodeParams

            # Apply scaling if it was used during pretraining
            if model_instance.config.use_inference_lr_scaling:
                model_grads_to_apply = apply_exponential_layer_scaling(
                    model_grads=model_grads_to_apply,
                    config=model_instance.config # Use the model's internal TransformerConfig
                )
            
            # Apply Vode h-gradient normalization if it was used during pretraining
            if model_instance.config.use_vode_grad_norm:
                model_grads_to_apply = normalize_vode_h_gradients(
                    model_grads=model_grads_to_apply,
                    config=model_instance.config # Use the model's internal TransformerConfig
                )
            
            optim_h.step(model_instance, model_grads_to_apply)

        # D. Extract features from the target Vode
        # Ensure target_vode_idx is valid
        num_vodes = len(model_instance.vodes)
        actual_idx = target_vode_idx
        if actual_idx < 0:
            actual_idx = num_vodes + actual_idx # Convert negative index to positive
        
        if not (0 <= actual_idx < num_vodes):
            raise ValueError(f"Invalid feature_layer_vode_idx: {target_vode_idx} (resolved to {actual_idx}). Model has {num_vodes} vodes.")

        # Get the h-state of the target Vode
        # We need to do a final forward pass with STATUS_FORWARD to ensure u states are current if the target Vode uses them
        # However, we are interested in the h state itself.
        # The h states are updated by optim_h.step().
        h_param = model_instance.vodes[actual_idx].h # Get the Param object
        extracted_h_features_array = h_param._value # Get the underlying JAX array

        # E. Apply Global Average Pooling if specified and features are per-patch
        # Features from vodes[0] are (batch, num_patches, patch_dim) or (batch, hidden_size) if top-level is single vector
        # Features from intermediate transformer blocks are (batch, num_patches, hidden_size)
        # Sensory vode vodes[-1] h is (batch, C, H, W)
        if use_gap_flag:
            if actual_idx == num_vodes -1 : # Sensory layer
                print(f"Warning: GAP requested for sensory layer (Vode {actual_idx}). Applying mean over spatial dims (H, W).")
                if extracted_h_features_array.ndim == 4: # (B, C, H, W)
                    extracted_h_features_array = jnp.mean(extracted_h_features_array, axis=(2,3)) # -> (B, C)
                elif extracted_h_features_array.ndim == 3: # (B, H, W) for grayscale
                    extracted_h_features_array = jnp.mean(extracted_h_features_array, axis=(1,2)) # -> (B,)
            elif extracted_h_features_array.ndim == 3: # Likely (batch, num_patches, feature_dim)
                extracted_h_features_array = jnp.mean(extracted_h_features_array, axis=1) # GAP over patches -> (batch, feature_dim)
            elif extracted_h_features_array.ndim == 2: # Already (batch, feature_dim), GAP not needed or already applied by Vode structure
                print(f"Note: Features from Vode {actual_idx} are already 2D. GAP not applied over patch dimension.")
            else:
                print(f"Warning: Cannot apply GAP to features from Vode {actual_idx} with shape {extracted_h_features_array.shape}. Returning as is.")
        
        return extracted_h_features_array # Return the JAX array

    # 7. Extract features for train and test sets
    print("\n--- Extracting Features ---")
    
    def extract_and_save_features(loader, set_name, model_instance_local, optim_h_local, inference_steps_local, target_vode_idx_local, use_gap_local):
        all_features_list = []
        all_labels_list = []
        print(f"Extracting features for {set_name} set...")
        for x_batch_pt, y_labels_pt in tqdm(loader, desc=f"Extracting {set_name} features"):
            batch_features_jnp = extract_features_from_batch(
                x_batch_pt.numpy(), # Convert PyTorch tensor to NumPy array
                model_instance_local, 
                optim_h_local, 
                inference_steps_local, 
                target_vode_idx_local, 
                use_gap_local
            )
            all_features_list.append(np.array(batch_features_jnp))
            all_labels_list.append(y_labels_pt.numpy()) # Convert PyTorch tensor to NumPy array
        
        # Concatenate all features and labels from the list of batches
        all_features_np = np.concatenate(all_features_list, axis=0)
        all_labels_np = np.concatenate(all_labels_list, axis=0)

        # Save features and labels
        features_path = os.path.join("./extracted_features", f"{args.config_name}_layer{target_vode_idx_local}_gap{use_gap_local}_{set_name}_features.npy")
        labels_path = os.path.join("./extracted_features", f"{args.config_name}_layer{target_vode_idx_local}_gap{use_gap_local}_{set_name}_labels.npy")
        np.save(features_path, all_features_np)
        np.save(labels_path, all_labels_np)
        print(f"Saved {set_name} features to: {features_path} (Shape: {all_features_np.shape})")
        print(f"Saved {set_name} labels to: {labels_path} (Shape: {all_labels_np.shape})")

    # Determine which Vode indices to run the probe for
    if args.feature_layer_vode_idx is not None:
        vode_indices_to_probe = [args.feature_layer_vode_idx]
        print(f"Probing specified Vode index: {vode_indices_to_probe}")
    else:
        # Sweep from Vode 0 (top-most) to Vode num_blocks + 1 (output of last transformer block)
        # Vode indices:
        # 0: Top-most latent
        # 1: Output of patch projection
        # 2 to num_blocks + 1: Output of each transformer block
        # num_blocks + 2: Sensory Vode (usually not used for features)
        num_total_vodes_in_model = base_model_config_obj.num_blocks + 2 # Vode 0, proj_vode, N block vodes
        vode_indices_to_probe = list(range(num_total_vodes_in_model))
        print(f"Sweeping through Vode indices: {vode_indices_to_probe}")

    # --- Define Linear Classifier Components ---
    # These are defined once here, to be in scope for the loop below.
    def linear_classifier_predict(params_local, x_local):
        return jnp.dot(x_local, params_local['W']) + params_local['b']

    def cross_entropy_loss(params_local, x_local, y_true_one_hot_local):
        logits_local = linear_classifier_predict(params_local, x_local)
        log_probs_local = jax.nn.log_softmax(logits_local)
        return -jnp.sum(log_probs_local * y_true_one_hot_local, axis=1).mean()

    def accuracy(params_local, x_local, y_true_one_hot_local):
        predictions_local = linear_classifier_predict(params_local, x_local)
        predicted_classes_local = jnp.argmax(predictions_local, axis=1)
        true_classes_local = jnp.argmax(y_true_one_hot_local, axis=1)
        return jnp.mean(predicted_classes_local == true_classes_local)
    # --- End Linear Classifier Components ---

    all_probe_results = [] # To store (vode_idx, test_accuracy)

    # --- Loop over Vode indices ---
    for current_vode_idx in vode_indices_to_probe:
        print(f"\n===== PROBING VODE INDEX: {current_vode_idx} =====")

        # Ensure ./extracted_features directory exists for this Vode index if needed
        # File paths inside extract_and_save_features and loading will use current_vode_idx
        # The `extract_and_save_features` function will now need current_vode_idx

        # 7. Extract features for train and test sets for the current_vode_idx
        print(f"\n--- Extracting Features for Vode {current_vode_idx} ---")
        extract_and_save_features(
            train_loader, "train", model, optim_h_extractor, 
            inference_steps_for_extraction, 
            current_vode_idx, # Pass current Vode index
            args.use_gap
        )
        extract_and_save_features(
            test_loader, "test", model, optim_h_extractor, 
            inference_steps_for_extraction, 
            current_vode_idx, # Pass current Vode index
            args.use_gap
        )
        print(f"\nFeature extraction for Vode {current_vode_idx} complete.")

        # 8. Linear Classifier Training and Evaluation for current_vode_idx
        print(f"\n--- Linear Classifier Training and Evaluation for Vode {current_vode_idx} ---")
        
        # Load the extracted features and labels for the current_vode_idx
        train_features_path = os.path.join("./extracted_features", f"{args.config_name}_layer{current_vode_idx}_gap{args.use_gap}_train_features.npy")
        train_labels_path = os.path.join("./extracted_features", f"{args.config_name}_layer{current_vode_idx}_gap{args.use_gap}_train_labels.npy")
        test_features_path = os.path.join("./extracted_features", f"{args.config_name}_layer{current_vode_idx}_gap{args.use_gap}_test_features.npy")
        test_labels_path = os.path.join("./extracted_features", f"{args.config_name}_layer{current_vode_idx}_gap{args.use_gap}_test_labels.npy")

        try:
            X_train_probe = np.load(train_features_path)
            y_train_probe = np.load(train_labels_path)
            X_test_probe = np.load(test_features_path)
            y_test_probe = np.load(test_labels_path)
        except FileNotFoundError:
            print(f"Error: Feature files not found for Vode {current_vode_idx}. Please ensure feature extraction ran successfully.")
            all_probe_results.append((current_vode_idx, "Error - Features not found"))
            continue # Skip to next Vode index

        print(f"Loaded training features for Vode {current_vode_idx}: {X_train_probe.shape}, labels: {y_train_probe.shape}")
        print(f"Loaded test features for Vode {current_vode_idx}: {X_test_probe.shape}, labels: {y_test_probe.shape}")

        # Initialize linear probe parameters (W, b)
        num_features = X_train_probe.shape[1]
        num_classes = 10 # For CIFAR-10
        key_probe, W_key, b_key = jax.random.split(jax.random.PRNGKey(args.seed + current_vode_idx), 3) # Vary seed per Vode

        probe_params = {
            'W': jax.random.normal(W_key, (num_features, num_classes)) * 0.01,
            'b': jnp.zeros(num_classes)
        }

        # One-hot encode labels for the probe
        y_train_probe_one_hot = jax.nn.one_hot(y_train_probe, num_classes)
        y_test_probe_one_hot = jax.nn.one_hot(y_test_probe, num_classes)

        # Setup optimizer for the probe
        probe_optimizer = optax.adamw(learning_rate=args.probe_lr, weight_decay=args.probe_wd)
        opt_state_probe = probe_optimizer.init(probe_params)

        print(f"Starting linear probe training for Vode {current_vode_idx}...")
        probe_train_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_train_probe).float(), torch.from_numpy(np.asarray(y_train_probe_one_hot)).float()),
            batch_size=args.probe_batch_size,
            shuffle=True, # Shuffle training data for the probe
            drop_last=True # Drop last if not a full batch
        )

        for epoch in range(args.probe_epochs):
            epoch_train_loss = 0.0
            num_batches_probe = 0
            for X_batch_pt, y_batch_pt in probe_train_loader: # Use the probe_train_loader
                X_batch_jax = jnp.array(X_batch_pt.numpy())
                y_batch_jax = jnp.array(y_batch_pt.numpy())

                loss_val, grads = jax.value_and_grad(cross_entropy_loss)(probe_params, X_batch_jax, y_batch_jax)
                
                updates, opt_state_probe = probe_optimizer.update(grads, opt_state_probe, probe_params)
                probe_params = optax.apply_updates(probe_params, updates)

                epoch_train_loss += loss_val
                num_batches_probe += 1

            avg_epoch_train_loss = epoch_train_loss / num_batches_probe
            
            # Calculate accuracy on the full training and test sets for the probe
            # Ensure to use the original (non-batched, non-shuffled) NumPy versions for X and one-hot JAX arrays for y
            epoch_train_acc = accuracy(probe_params, X_train_probe, np.asarray(y_train_probe_one_hot))
            epoch_test_acc = accuracy(probe_params, X_test_probe, np.asarray(y_test_probe_one_hot))
            
            print(f"Vode {current_vode_idx} - Probe Epoch {epoch+1}/{args.probe_epochs} - Train Loss: {avg_epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, Test Acc: {epoch_test_acc:.4f}")
        
        # Evaluate the probe on the test set (final accuracy after all epochs)
        final_test_acc = accuracy(probe_params, X_test_probe, np.asarray(y_test_probe_one_hot))
        print(f"===== Vode {current_vode_idx} - Final Test Accuracy: {final_test_acc:.4f} =====")
        all_probe_results.append((current_vode_idx, float(final_test_acc)))

    # --- End Loop --- 

    # Conditional post-extraction visualization
    if args.feature_layer_vode_idx is not None: # Only run post-extraction viz if a specific Vode was targeted
        # --- Add Post-Feature-Extraction Reconstruction Visualization ---
        print("\n--- Generating Post-Feature-Extraction Reconstruction Visualization ---")
        # We use the same vis_num_images, vis_fps, vis_image_shape as before.
        # vis_target_T_values will be a list with a single element: the final step number.
        vis_target_T_values = [inference_steps_for_extraction]
        print(f"Visualization will use T_values: {vis_target_T_values} (derived from probe_inference_steps or its default)")

        # Get a batch from the test_loader for visualization
        try:
            vis_batch_pt_full, vis_labels_pt_full = next(iter(test_loader))
        except StopIteration:
            print("Error: Could not get a batch from test_loader for visualization.")
            vis_batch_pt_full = None

        if vis_batch_pt_full is not None:
            model.eval() # Ensure model is in eval mode
            
            post_all_images_all_recons_list = [] # To store [[img1_recons_steps], [img2_recons_steps]]
            post_all_images_orig_for_video = []
            post_all_images_labels_for_video = []
            post_all_images_recon_loss = []

            for i in range(vis_num_images):
                print(f"Processing image {i+1}/{vis_num_images} for reconstruction video...")
                single_image_np = vis_batch_pt_full[i:i+1].numpy() # Batch of 1 image
                single_label_np = vis_labels_pt_full[i:i+1].numpy()

                post_all_images_orig_for_video.append(single_image_np[0]) # Store unbatched image
                post_all_images_labels_for_video.append(single_label_np[0])

                recon_loss_single, single_img_all_recons_list, _, _ = unmask_on_batch_enhanced(
                    use_corruption=False,
                    corrupt_ratio=0.0,
                    target_T_values=vis_target_T_values,
                    x=jnp.array(single_image_np), # Pass single image as JAX array
                    model=model,
                    optim_h=optim_h_extractor,
                    optim_w=None
                )
                post_all_images_all_recons_list.append(single_img_all_recons_list)
                post_all_images_recon_loss.append(recon_loss_single)
                print(f"  Reconstruction loss for image {i+1}: {recon_loss_single:.4f}")

            if post_all_images_all_recons_list:
                # create_reconstruction_video expects: List (num_images) of lists (T_steps) of reconstruction tensors.
                # Each inner list (T_steps) should contain tensors of shape (1, C, H, W) or (C,H,W) if squeezed later.
                # single_img_all_recons_list already contains [step0_recon_batch_of_1, step1_recon_batch_of_1, ...]
                # So, post_all_images_all_recons_list is already in the correct format.
                create_reconstruction_video(
                    all_reconstruction_frames=post_all_images_all_recons_list,
                    orig_images=post_all_images_orig_for_video,
                    masked_images=post_all_images_orig_for_video, # Same as original for no corruption
                    labels_list=post_all_images_labels_for_video,
                    num_images=vis_num_images,
                    image_shape=vis_image_shape,
                    wandb_run=None, 
                    epoch=None,     
                    fps=vis_fps,
                    reconstruction_mses=post_all_images_recon_loss 
                )
        # --- End Post-Feature-Extraction Reconstruction Visualization ---
    else:
        print("Skipping post-extraction reconstruction visualization because running in Vode sweep mode.")

    # 9. Log all results
    print("\n===== Linear Probe Sweep Results =====")
    results_log_path = os.path.join("./results", f"linear_probe_sweep_{args.config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    os.makedirs("./results", exist_ok=True)

    with open(results_log_path, 'w') as f:
        f.write(f"Linear Probe Sweep Results - Model: {args.model_path}, Config: {args.config_name}\n")
        f.write(f"Probe LR: {args.probe_lr}, Probe WD: {args.probe_wd}, Probe Epochs: {args.probe_epochs}\n")
        f.write(f"Feature Extraction H LR: {h_inference_lr}, Feature Extraction Steps: {inference_steps_for_extraction}\n")
        f.write(f"Global Average Pooling: {args.use_gap}\n\n")
        f.write("Vode Index | Test Accuracy\n")
        f.write("--------------------------\n")
        for vode_idx, acc in sorted(all_probe_results, key=lambda item: item[1], reverse=True):
            if isinstance(acc, str): # Handle error cases
                f.write(f"{vode_idx:<10} | {acc}\n")
                print(f"Vode {vode_idx}: {acc}")
            else:
                f.write(f"{vode_idx:<10} | {acc:.4f}\n")
                print(f"Vode {vode_idx}: {acc:.4f}")
    
    print(f"\nResults saved to: {results_log_path}")

    # Clean up: Remove feature files if desired (optional)
    # print("\nCleaning up extracted feature files...")
    # for current_vode_idx in vode_indices_to_probe:
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