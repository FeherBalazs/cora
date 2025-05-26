from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Callable # Removed Dict, Any as they are not directly used by ModelConfig
import jax # For jax.nn.swish

# Original MODEL_CONFIGS and ModelConfig were moved here from debug_transformer_wandb.py

@dataclass
class ModelConfig:
    """Configuration for transformer models with all hyperparameters in one place."""
    name: str
    # Dataset settings
    dataset: str = "cifar10"
    dataset_img_shape: Optional[Tuple[int, ...]] = None # New: (C, H, W) or (F, C, H, W)
    data_dir: str = "data/"
    train_subset: int = 50000
    test_subset: int = 200
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

    peak_lr_weights: float = 0.001
    peak_lr_hidden: float = 0.1
    hidden_momentum: float = 0.1 # New: Momentum for hidden state optimizer

    update_weights_during_unmasking: bool = False

    hidden_lr_inference: float = peak_lr_hidden * 1 # Note: This will use the class-level peak_lr_hidden
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
    run_linear_probe_during_hyperparam_search: bool = True # New: Control if probe runs
    probe_train_subset_n: int = 1000
    probe_test_subset_n: int = 200
    probe_feature_layer_vode_indices: str = "0,1" # Default: input, first block. Will be dynamically adjusted based on num_blocks.
    probe_epochs: int = 10 # Default epochs for the probe classifier
    probe_lr: float = 1e-3 # Default LR for the probe classifier
    probe_wd: float = 1e-4 # Default WD for the probe classifier

    def __post_init__(self):
        # Ensure hidden_lr_inference is correctly derived if peak_lr_hidden is changed post-init
        # This is important if ModelConfig instances are created and then peak_lr_hidden is modified.
        if hasattr(self, 'peak_lr_hidden'): # Check if peak_lr_hidden is an attribute
             self.hidden_lr_inference = self.peak_lr_hidden * 1


# Predefined configurations for easy experimentation
MODEL_CONFIGS = {
    "debug_tiny": ModelConfig(
        name="debug_tiny",
        hidden_size=64,
        num_heads=1,
        num_blocks=5,
        dataset_img_shape=(3,32,32), # Added for cifar10
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
        dataset_img_shape=(3,32,32),
        hidden_size=64,
        num_heads=1,
        num_blocks=0,
        theta=100, # Overriding default
        peak_lr_hidden=0.095,
        inference_lr_scale_base=1.25,
        h_grad_clip_norm=2000.0,
        w_grad_clip_norm=500.0,
        hidden_momentum=0.4,
        use_lr_schedule_w=False,
        use_lr_schedule_h=True,
        use_ssl_augmentations=True,
        use_cifar10_norm=True,
        reconstruction_every_n_epochs=10,
        validation_every_n_epochs=5,
        early_stopping_patience=3,
    ),
    "1block": ModelConfig(
        name="1block",
        dataset_img_shape=(3,32,32),
        hidden_size=64,
        num_heads=1,
        num_blocks=1,
        theta=100, # Overriding default
        peak_lr_hidden=0.095,
        inference_lr_scale_base=1.25,
        h_grad_clip_norm=2000.0,
        w_grad_clip_norm=500.0,
        hidden_momentum=0.4,
        use_lr_schedule_w=False,
        use_lr_schedule_h=True,
        use_ssl_augmentations=True,
        use_cifar10_norm=True,
        reconstruction_every_n_epochs=10,
        validation_every_n_epochs=5,
        early_stopping_patience=3,
    ),
    "2block": ModelConfig(
        name="2block",
        dataset_img_shape=(3,32,32),
        hidden_size=64,
        num_heads=1,
        num_blocks=2,
        theta=100, # Overriding default
        peak_lr_hidden=0.095,
        inference_lr_scale_base=1.25,
        h_grad_clip_norm=2000.0,
        w_grad_clip_norm=500.0,
        hidden_momentum=0.4,
        use_lr_schedule_w=False,
        use_lr_schedule_h=True,
        use_ssl_augmentations=True,
        use_cifar10_norm=True,
        reconstruction_every_n_epochs=10,
        validation_every_n_epochs=5,
        early_stopping_patience=3,
    ),
    "5block": ModelConfig(
        name="5block",
        dataset_img_shape=(3,32,32),
        hidden_size=64,
        num_heads=1,
        num_blocks=5,
        theta=100, # Overriding default
        peak_lr_hidden=0.095, # Keeping consistent high LR for hidden
        hidden_lr_inference=0.1, # Explicitly stated, but covered by __post_init__
        inference_lr_scale_base=1.25,
        h_grad_clip_norm=5000.0,
        w_grad_clip_norm=500.0,
        hidden_momentum=0.3,
        use_lr_schedule_w=False, # Weights fixed
        use_lr_schedule_h=True,   # Hidden scheduled
        use_ssl_augmentations=True,
        use_cifar10_norm=True,
        reconstruction_every_n_epochs=25,
        validation_every_n_epochs=10, # Less frequent for potentially longer runs
        early_stopping_patience=3,
    ),
    "6block": ModelConfig(
        name="6block",
        dataset_img_shape=(3,32,32),
        hidden_size=64,
        num_heads=1,
        num_blocks=6,
        theta=10_000, # Original theta for this config
        peak_lr_hidden=0.095,
        inference_lr_scale_base=1.25,
        h_grad_clip_norm=2000.0,
        w_grad_clip_norm=500.0,
        hidden_momentum=0.4,
        use_lr_schedule_w=False, 
        use_lr_schedule_h=False, # << Original was False for 6block test
        use_ssl_augmentations=False, # << Original
        use_cifar10_norm=False, # << Original
        reconstruction_every_n_epochs=10,
        validation_every_n_epochs=5,
        early_stopping_patience=3,
    )
}

# Default configuration to use
DEFAULT_CONFIG = "debug_tiny" 