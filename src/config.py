from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Callable, Dict, Any
import jax

from src.decoder_transformer import (
    TransformerConfig,
)


@dataclass
class ModelConfig:
    """Configuration for transformer models with all hyperparameters in one place."""
    name: str
    # Dataset settings
    dataset: str = "cifar10"
    dataset_img_shape: Optional[Tuple[int, ...]] = None # New: (C, H, W) or (F, C, H, W)
    data_dir: str = "../datasets/"
    train_subset: int = 50000
    test_subset: int = 200
    validation_subset: Optional[int] = 2000
    target_class: Optional[int] = None
    reconstruction_every_n_epochs: int = 25 # Adjusted for shorter runs
    validation_every_n_epochs: int = 5 # Adjusted for shorter runs

    use_corruption: bool = False
    corrupt_ratio: float = 0.25

    use_lower_half_mask: bool = False #If False it uses random masking
    inference_clamp_alpha: float = 1.0     # Blending factor for soft clamping

    # Visualization settings
    num_images: int = 1
    
    # Model architecture
    hidden_size: int = 48
    num_heads: int = 6
    num_blocks: int = 1
    mlp_ratio: float = 4.0
    patch_size: int = 4
    axes_dim: List[int] = field(default_factory=lambda: [16, 16])
    theta: int = 100
    act_fn: Callable = jax.nn.swish

    # Status init settings for training and unmasking
    use_status_init_in_training: bool = False
    use_status_init_in_unmasking: bool = False
    
    # Training settings
    use_noise: bool = True
    batch_size: int = 200
    epochs: int = 75
    inference_steps: int = 20
    eval_inference_steps: List[int] = field(default_factory=lambda: [20])
    reconstruction_steps: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 8, 12, 16, 20])

    # Settings without status.init: epochs=50, hidden_size=64, num_blocks=5, inference_steps=20, update_weights_every_inference_step=False, use_inference_lr_scaling=True, inference_lr_scale_base=1.2, h_grad_clip_norm=1000.0, w_grad_clip_norm=500.0, mse=0.005 => possible to hit but erratically
    peak_lr_weights: float = 0.001
    peak_lr_hidden: float = 0.1
    hidden_momentum: float = 0.1 # New: Momentum for hidden state optimizer


    update_weights_during_unmasking: bool = False

    hidden_lr_inference: float = peak_lr_hidden * 1
    weight_decay: float = 2e-4
    warmup_steps: int = 0 # New: Number of steps for warmup
    use_lr_schedule_w: bool = True # New: Use LR schedule for weights
    use_lr_schedule_h: bool = True  # New: Use LR schedule for hidden states
    seed: int = 42
    
    # Layer-specific inference LR scaling
    use_inference_lr_scaling: bool = True
    inference_lr_scale_base: Optional[float] = 1.25

    h_grad_clip_norm: Optional[float] = 2000.0 # Max norm for H-gradient clipping
    w_grad_clip_norm: Optional[float] = 500.0  # Max norm for W-gradient clipping

    # iPC or classic PC
    update_weights_every_inference_step: bool = False
    
    # Early stopping settings
    use_early_stopping: bool = True
    early_stopping_patience: int = 7 # Adjusted for longer runs
    early_stopping_min_delta: float = 0.001
    save_reconstruction_images: bool = True # Option to save static image grid
    save_reconstruction_video: bool = True # Option to save video
    video_fps: int = 60 # Frames per second for the reconstruction video
    reinitialize_model_for_each_epoch: bool = False # WARNING: setting this to True will get around 0.24 val loss with 100 images vs 0.15 without.
    
    # New normalization options
    use_vode_state_layernorm: bool = False # Apply LayerNorm to Vode hidden states (h)
    use_vode_grad_norm: bool = False       # Normalize Vode h-gradients before optimizer step
    vode_grad_norm_target: float = 1.0     # Target norm for h-gradient normalization
    use_adamw_for_hidden_optimizer: bool = False # New: Use AdamW for hidden state optimizer
    lr_schedule_min_lr_factor: float = 0.5 # New: Factor to determine min_lr in schedule (min_lr = base_lr * factor)
    use_ssl_augmentations: bool = True # New: Use stronger augmentations for SSL pretraining
    early_stopping_metric: str = "val_mse" # New: Metric for early stopping ("val_mse" or "train_mse")
    use_cifar10_norm: bool = True # New: Use CIFAR-10 specific normalization vs (0.5, 0.5, 0.5)
    save_model_train_mse_threshold: Optional[float] = 0.006 # New: Train MSE threshold to enable model saving
    model_saving_metric: str = "train_mse" # New: Metric to base model saving on ("train_mse" or "val_mse")
    
    # Linear Probing settings
    linear_probe_every_n_epochs: int = 0 # 0 to disable, N > 0 to run every N epochs
    linear_probe_vode_indices: str = "0" # Comma-separated string of Vode indices
    linear_probe_concatenate_features: bool = False
    linear_probe_use_gap: bool = True
    linear_probe_lr: float = 1e-3
    linear_probe_wd: float = 1e-4
    linear_probe_epochs: int = 100
    linear_probe_batch_size: int = 256
    linear_probe_h_lr: Optional[float] = None # If None, uses hidden_lr_inference from main config
    linear_probe_inference_steps: Optional[int] = None # If None, uses inference_steps from main config
    linear_probe_seed: int = 123 # Separate seed for linear probe for consistency

    # Regularization coefficients for intermediate Vodes
    intermediate_l1_coeff: float = 0.0
    intermediate_l2_coeff: float = 0.0


MODEL_CONFIGS = {
    "debug_tiny": ModelConfig(
        name="debug_tiny",
        hidden_size=64,
        num_heads=1,
        num_blocks=5,
        dataset_img_shape=(3,32,32), # Added for cifar10
        validation_subset=2000,
        # Target experiment settings for next run:
        peak_lr_hidden=0.07,
        inference_lr_scale_base=1.1,
        h_grad_clip_norm=1000.0,
        w_grad_clip_norm=500.0,
        epochs=25,
        use_inference_lr_scaling=True,
        validation_every_n_epochs=5, # Keep frequent validation
        reconstruction_every_n_epochs=25, # And reconstruction
    ),
    "0block": ModelConfig(
        name="0block",
        # Dataset settings
        dataset="cifar10",
        dataset_img_shape=(3,32,32), # Added for cifar10
        data_dir="../datasets/",
        train_subset=50000,
        test_subset=200,
        validation_subset=2000,
        target_class=None,
        reconstruction_every_n_epochs=10,
        validation_every_n_epochs=5,
        use_corruption=False,
        corrupt_ratio=0.25,
        use_lower_half_mask=False,
        inference_clamp_alpha=1.0,
        # Visualization settings
        num_images=2,
        # Model architecture
        hidden_size=64,
        num_heads=1,
        num_blocks=0,
        mlp_ratio=4.0,
        patch_size=4,
        axes_dim=[16, 16],
        theta=100,
        act_fn=jax.nn.swish,
        # Status init settings
        use_status_init_in_training=False,
        use_status_init_in_unmasking=False,
        # Training settings
        use_noise=True,
        batch_size=200,
        epochs=75,
        inference_steps=20,
        eval_inference_steps=[20],
        reconstruction_steps=[20],
        peak_lr_weights=0.001,
        peak_lr_hidden=0.095,
        update_weights_during_unmasking=False,
        hidden_lr_inference=0.095,
        weight_decay=2e-4,
        warmup_steps=0, # Explicitly set for 6block
        seed=42,
        # Layer-specific inference LR scaling
        use_inference_lr_scaling=True,
        inference_lr_scale_base=1.25,
        # Grad clipping
        h_grad_clip_norm=2000.0,
        w_grad_clip_norm=500.0,
        # iPC or classic PC
        update_weights_every_inference_step=False,
        # Early stopping settings
        use_early_stopping=True,
        early_stopping_patience=3,
        early_stopping_min_delta=0.001,
        save_reconstruction_images=True,
        save_reconstruction_video=True,
        video_fps=60,
        reinitialize_model_for_each_epoch=False,
        # New norm options
        use_vode_state_layernorm=False,
        use_vode_grad_norm=False,
        vode_grad_norm_target=1.0,
        hidden_momentum=0.4,
        use_adamw_for_hidden_optimizer=False, # Default to SGD
        lr_schedule_min_lr_factor=0.5, # Default factor
        # use_lr_schedule was True, translating to:
        use_lr_schedule_w=False, # Weights fixed
        use_lr_schedule_h=True,   # Hidden scheduled
        use_ssl_augmentations=True, # Enable SSL augmentations
        early_stopping_metric="val_mse", # Default early stopping metric
        use_cifar10_norm=True, # Default to CIFAR-10 specific norm
        save_model_train_mse_threshold=0.006,
        model_saving_metric="train_mse",
    ),
    "1block": ModelConfig(
        name="1block",
        # Dataset settings
        dataset="cifar10",
        dataset_img_shape=(3,32,32), # Added for cifar10
        data_dir="../datasets/",
        train_subset=50000,
        test_subset=200,
        validation_subset=2000,
        target_class=None,
        reconstruction_every_n_epochs=10,
        validation_every_n_epochs=5,
        use_corruption=False,
        corrupt_ratio=0.25,
        use_lower_half_mask=False,
        inference_clamp_alpha=1.0,
        # Visualization settings
        num_images=2,
        # Model architecture
        hidden_size=64,
        num_heads=1,
        num_blocks=1,
        mlp_ratio=4.0,
        patch_size=4,
        axes_dim=[16, 16],
        theta=100,
        act_fn=jax.nn.swish,
        # Status init settings
        use_status_init_in_training=False,
        use_status_init_in_unmasking=False,
        # Training settings
        use_noise=True,
        batch_size=200,
        epochs=75,
        inference_steps=20,
        eval_inference_steps=[20],
        reconstruction_steps=[20],
        peak_lr_weights=0.001,
        peak_lr_hidden=0.095,
        update_weights_during_unmasking=False,
        hidden_lr_inference=0.095,
        weight_decay=2e-4,
        warmup_steps=0, # Explicitly set for 6block
        seed=42,
        # Layer-specific inference LR scaling
        use_inference_lr_scaling=True,
        inference_lr_scale_base=1.25,
        # Grad clipping
        h_grad_clip_norm=2000.0,
        w_grad_clip_norm=500.0,
        # iPC or classic PC
        update_weights_every_inference_step=False,
        # Early stopping settings
        use_early_stopping=True,
        early_stopping_patience=3,
        early_stopping_min_delta=0.001,
        save_reconstruction_images=True,
        save_reconstruction_video=True,
        video_fps=60,
        reinitialize_model_for_each_epoch=False,
        # New norm options
        use_vode_state_layernorm=False,
        use_vode_grad_norm=False,
        vode_grad_norm_target=1.0,
        hidden_momentum=0.4,
        use_adamw_for_hidden_optimizer=False, # Default to SGD
        lr_schedule_min_lr_factor=0.5, # Default factor
        # use_lr_schedule was True, translating to:
        use_lr_schedule_w=False, # Weights fixed
        use_lr_schedule_h=True,   # Hidden scheduled
        use_ssl_augmentations=True, # Enable SSL augmentations
        early_stopping_metric="val_mse", # Default early stopping metric
        use_cifar10_norm=True, # Default to CIFAR-10 specific norm
        save_model_train_mse_threshold=0.006,
        model_saving_metric="train_mse",
    ),
    "2block": ModelConfig(
        name="2block",
        # Dataset settings
        dataset="cifar10",
        dataset_img_shape=(3,32,32), # Added for cifar10
        data_dir="../datasets/",
        train_subset=50000,
        test_subset=200,
        validation_subset=2000,
        target_class=None,
        reconstruction_every_n_epochs=10,
        validation_every_n_epochs=5,
        use_corruption=False,
        corrupt_ratio=0.25,
        use_lower_half_mask=False,
        inference_clamp_alpha=1.0,
        # Visualization settings
        num_images=2,
        # Model architecture
        hidden_size=64,
        num_heads=1,
        num_blocks=2,
        mlp_ratio=4.0,
        patch_size=4,
        axes_dim=[16, 16],
        theta=100,
        act_fn=jax.nn.swish,
        # Status init settings
        use_status_init_in_training=False,
        use_status_init_in_unmasking=False,
        # Training settings
        use_noise=True,
        batch_size=200,
        epochs=75,
        inference_steps=20,
        eval_inference_steps=[20],
        reconstruction_steps=[20],
        peak_lr_weights=0.001,
        peak_lr_hidden=0.095,
        update_weights_during_unmasking=False,
        hidden_lr_inference=0.095,
        weight_decay=2e-4,
        warmup_steps=0, # Explicitly set for 6block
        seed=42,
        # Layer-specific inference LR scaling
        use_inference_lr_scaling=True,
        inference_lr_scale_base=1.25,
        # Grad clipping
        h_grad_clip_norm=2000.0,
        w_grad_clip_norm=500.0,
        # iPC or classic PC
        update_weights_every_inference_step=False,
        # Early stopping settings
        use_early_stopping=True,
        early_stopping_patience=3,
        early_stopping_min_delta=0.001,
        save_reconstruction_images=True,
        save_reconstruction_video=True,
        video_fps=60,
        reinitialize_model_for_each_epoch=False,
        # New norm options
        use_vode_state_layernorm=False,
        use_vode_grad_norm=False,
        vode_grad_norm_target=1.0,
        hidden_momentum=0.4,
        use_adamw_for_hidden_optimizer=False, # Default to SGD
        lr_schedule_min_lr_factor=0.5, # Default factor
        # use_lr_schedule was True, translating to:
        use_lr_schedule_w=False, # Weights fixed
        use_lr_schedule_h=True,   # Hidden scheduled
        use_ssl_augmentations=True, # Enable SSL augmentations
        early_stopping_metric="val_mse", # Default early stopping metric
        use_cifar10_norm=True, # Default to CIFAR-10 specific norm
        save_model_train_mse_threshold=0.006,
        model_saving_metric="train_mse",
    ),
    "5block": ModelConfig(
        name="5block",
        # Dataset settings
        dataset="cifar10",
        dataset_img_shape=(3,32,32), # Added for cifar10
        data_dir="../datasets/",
        train_subset=50000,
        test_subset=200,
        validation_subset=2000,
        target_class=None,
        reconstruction_every_n_epochs=25,
        validation_every_n_epochs=10,
        use_corruption=False,
        corrupt_ratio=0.25,
        use_lower_half_mask=False,
        inference_clamp_alpha=1.0,
        # Visualization settings
        num_images=2,
        # Model architecture
        hidden_size=64,
        num_heads=1,
        num_blocks=5,
        mlp_ratio=4.0,
        patch_size=4,
        axes_dim=[16, 16],
        theta=100,
        act_fn=jax.nn.swish,
        # Status init settings
        use_status_init_in_training=False,
        use_status_init_in_unmasking=False,
        # Training settings
        use_noise=True,
        batch_size=200,
        epochs=75,
        inference_steps=20,
        eval_inference_steps=[20],
        reconstruction_steps=[20],
        peak_lr_weights=0.001,
        peak_lr_hidden=0.095,
        update_weights_during_unmasking=False,
        hidden_lr_inference=0.1,
        weight_decay=2e-4,
        warmup_steps=0, # Explicitly set for 5block
        seed=42,
        # Layer-specific inference LR scaling
        use_inference_lr_scaling=True,
        inference_lr_scale_base=1.25,
        # Grad clipping
        h_grad_clip_norm=5000.0,
        w_grad_clip_norm=500.0,
        # iPC or classic PC
        update_weights_every_inference_step=False,
        # Early stopping settings
        use_early_stopping=True,
        early_stopping_patience=3,
        early_stopping_min_delta=0.001,
        save_reconstruction_images=True,
        save_reconstruction_video=True,
        video_fps=60,
        reinitialize_model_for_each_epoch=False,
        # New norm options
        use_vode_state_layernorm=False,
        use_vode_grad_norm=False,
        vode_grad_norm_target=1.0,
        hidden_momentum=0.3,
        use_adamw_for_hidden_optimizer=False, # Default to SGD
        lr_schedule_min_lr_factor=0.5, # Default factor
        # use_lr_schedule was True, translating to:
        use_lr_schedule_w=False, # Weights fixed (as per recent changes)
        use_lr_schedule_h=True,   # Hidden scheduled
        use_ssl_augmentations=True, # Enable SSL augmentations
        early_stopping_metric="val_mse", # Default early stopping metric
        use_cifar10_norm=True, # Default to CIFAR-10 specific norm
        save_model_train_mse_threshold=0.006,
        model_saving_metric="train_mse",
    ),
    "6block": ModelConfig(
        name="6block",
        # Dataset settings
        dataset="cifar10",
        dataset_img_shape=(3,32,32), # Added for cifar10
        data_dir="../datasets/",
        train_subset=50000,
        test_subset=200,
        validation_subset=2000,
        target_class=None,
        reconstruction_every_n_epochs=10,
        validation_every_n_epochs=5,
        use_corruption=False,
        corrupt_ratio=0.25,
        use_lower_half_mask=False,
        inference_clamp_alpha=1.0,
        # Visualization settings
        num_images=2,
        # Model architecture
        hidden_size=64,
        num_heads=1,
        num_blocks=6,
        mlp_ratio=4.0,
        patch_size=4,
        axes_dim=[16, 16],
        theta=10_000,
        act_fn=jax.nn.swish,
        # Status init settings
        use_status_init_in_training=False,
        use_status_init_in_unmasking=False,
        # Training settings
        use_noise=True,
        batch_size=200,
        epochs=75,
        inference_steps=20,
        eval_inference_steps=[20],
        reconstruction_steps=[20],
        peak_lr_weights=0.001,
        peak_lr_hidden=0.095,
        update_weights_during_unmasking=False,
        hidden_lr_inference=0.095,
        weight_decay=2e-4,
        warmup_steps=0, # Explicitly set for 6block
        seed=42,
        # Layer-specific inference LR scaling
        use_inference_lr_scaling=True,
        inference_lr_scale_base=1.25,
        # Grad clipping
        h_grad_clip_norm=2000.0,
        w_grad_clip_norm=500.0,
        # iPC or classic PC
        update_weights_every_inference_step=False,
        # Early stopping settings
        use_early_stopping=True,
        early_stopping_patience=3,
        early_stopping_min_delta=0.001,
        save_reconstruction_images=True,
        save_reconstruction_video=True,
        video_fps=60,
        reinitialize_model_for_each_epoch=False,
        # New norm options
        use_vode_state_layernorm=False,
        use_vode_grad_norm=False,
        vode_grad_norm_target=1.0,
        hidden_momentum=0.4,
        use_adamw_for_hidden_optimizer=False, # Default to SGD
        lr_schedule_min_lr_factor=0.5, # Default factor
        # use_lr_schedule was True, translating to:
        use_lr_schedule_w=False, # Weights fixed
        use_lr_schedule_h=False,   # Hidden scheduled
        use_ssl_augmentations=False, # Enable SSL augmentations
        early_stopping_metric="val_mse", # Default early stopping metric
        use_cifar10_norm=False, # Default to CIFAR-10 specific norm
        save_model_train_mse_threshold=0.006,
        model_saving_metric="train_mse",
    )
}

# Default configuration to use
DEFAULT_CONFIG = "debug_tiny" # Changed to run the new specific settings


def create_config(dataset="cifar10", hidden_size=48, num_blocks=1, num_heads=6,
                 mlp_ratio=4.0, patch_size=4, axes_dim=None, theta=10_000, use_noise=True, use_lower_half_mask=False,
                 use_inference_lr_scaling=False, 
                 inference_lr_scale_base=1.1,
                 inference_clamp_alpha=1.0, update_weights_during_unmasking=False,
                 use_status_init_in_training: bool = True, use_status_init_in_unmasking: bool = True,
                 update_weights_every_inference_step=False,
                 use_vode_state_layernorm: bool = False, # New
                 use_vode_grad_norm: bool = False,       # New
                 vode_grad_norm_target: float = 1.0,      # New
                 hidden_momentum: float = 0.1, # New parameter for create_config, not directly used by TransformerConfig
                 intermediate_l1_coeff: float = 0.0, # ADDED
                 intermediate_l2_coeff: float = 0.0  # ADDED
                 ):
    """Create a TransformerConfig based on the dataset name and parameters."""
    axes_dim = axes_dim or [16, 16]
    
    if dataset == "cifar10":
        return TransformerConfig(
            image_shape=(3, 32, 32),
            num_frames=16,
            is_video=False,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_blocks=num_blocks,
            mlp_ratio=mlp_ratio,
            patch_size=patch_size,
            axes_dim=axes_dim,
            theta=theta,
            use_noise=use_noise,
            use_lower_half_mask=use_lower_half_mask,
            use_inference_lr_scaling=use_inference_lr_scaling,
            inference_lr_scale_base=inference_lr_scale_base,
            inference_clamp_alpha=inference_clamp_alpha,
            update_weights_during_unmasking=update_weights_during_unmasking,
            use_status_init_in_training=use_status_init_in_training,
            use_status_init_in_unmasking=use_status_init_in_unmasking,
            update_weights_every_inference_step=update_weights_every_inference_step,
            use_vode_state_layernorm=use_vode_state_layernorm,
            use_vode_grad_norm=use_vode_grad_norm,
            vode_grad_norm_target=vode_grad_norm_target,
            intermediate_l1_coeff=intermediate_l1_coeff,
            intermediate_l2_coeff=intermediate_l2_coeff
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")