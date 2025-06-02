# 12-Block Comprehensive Hyperparameter Search

## Overview
This sweep configuration (`sweep_12block_comprehensive.yaml`) is designed to extensively explore the hyperparameter space for 12-block PC-ViT models. Since we're doubling the depth from 6 to 12 blocks, we expect significantly different optimization dynamics and stability challenges.

## Key Changes from 6-Block Configuration

### Architecture
- **num_blocks**: Fixed to 12
- **hidden_size**: Expanded to [64, 96, 128] - larger models may need more capacity
- **num_heads**: Expanded to [1, 2, 4] - multi-head attention might help deeper models
- **batch_size**: [128, 200, 256] - smaller batches might improve stability

### Learning Rates (Critical for Deep Models)
- **peak_lr_hidden**: [0.05 - 0.11] - broader range around known good values
- **peak_lr_weights**: [0.0005 - 0.003] - slightly higher for deeper models
- **inference_lr_scale_base**: [1.15 - 1.35] - more aggressive scaling needed for 12 blocks

### Stability & Optimization
- **h_grad_clip_norm**: [1000 - 5000] - higher clipping values for deeper models
- **w_grad_clip_norm**: [300 - 1500] - expanded range
- **hidden_momentum**: [0.3 - 0.5] - higher momentum for stability
- **warmup_steps**: [0 - 200] - warmup becomes important for deep models
- **use_vode_state_layernorm**: [false, true] - layer norm might be critical

### Training Duration & Early Stopping
- **epochs**: 100 (vs 75 for 6-block)
- **early_stopping_patience**: [15, 20, 25] - more patience needed
- **early_stopping_min_delta**: [0.0005 - 0.002] - more sensitive stopping
- **validation_every_n_epochs**: [5, 10, 15] - more frequent validation

### Regularization (Very Important for Deep Models)
- **intermediate_l1_coeff**: [0.0 - 0.01] - extended range with finer granularity
- **intermediate_l2_coeff**: [0.0 - 0.01] - extended range with finer granularity

## Expected Challenges for 12-Block Models

1. **Training Instability**: Deeper models are more prone to gradient explosion/vanishing
2. **Slower Convergence**: May need more epochs and patience
3. **Memory Usage**: Larger models with more parameters
4. **Gradient Flow**: Higher layers may receive insufficient gradients
5. **Representation Quality**: Need to balance reconstruction vs feature learning

## Sweep Strategy

- **Method**: Random sampling (500 runs) to efficiently explore the large space
- **Metric**: Maximize `best_probe_accuracy` - focusing on representation quality
- **Computational Efficiency**: 
  - Disabled video reconstruction to save compute
  - Linear probing every 10 epochs (vs more frequent for smaller sweeps)
  - Quick 10-epoch linear probes

## Key Hypotheses to Test

1. **Layer Normalization**: Will `use_vode_state_layernorm=true` help with deeper models?
2. **Learning Rate Scaling**: Do we need more aggressive `inference_lr_scale_base` for 12 blocks?
3. **Regularization**: Will stronger L1/L2 regularization improve feature quality?
4. **Architecture**: Do more heads or larger hidden size help with depth?
5. **Training Stability**: What gradient clipping values work best for 12 blocks?

## Running the Sweep

```bash
# Create the sweep
wandb sweep sweeps/sweep_12block_comprehensive.yaml

# Run agents (adjust count based on available compute)
wandb agent <sweep_id>
```

## Expected Timeline

With 500 runs and approximately 100 epochs each:
- Estimated time per run: 2-4 hours (depending on early stopping)
- Total compute: ~1000-2000 GPU hours
- Recommended: Use multiple GPUs in parallel

## Success Metrics

- **Primary**: Linear probe accuracy > 35% (target improvement over 6-block best of ~30%)
- **Secondary**: Training stability (fewer diverged runs)
- **Tertiary**: Reconstruction MSE < 0.005 (maintain good reconstruction quality)

## Notes

- Monitor for frequent divergence - may need to adjust gradient clipping ranges
- Watch memory usage - 12-block models will be significantly larger
- Consider early stopping if many runs fail to converge
- May need to create a smaller focused sweep based on initial results 