# W&B Hyperparameter Sweeps Guide

This guide explains how to use the new W&B sweep system for hyperparameter optimization, replacing the custom grid search approach.

## Files Overview

- `sweep.yaml` - Sweep configuration defining the parameter search space
- `examples/run_sweep.py` - Script to create and run sweeps
- `examples/debug_transformer_wandb.py` - Modified to work with sweeps (backward compatible)

## Quick Start

### 1. Basic Sweep Run

```bash
# Create and run a sweep with 10 experiments
python examples/run_sweep.py \
    --sweep-config sweep.yaml \
    --project "pc-arch-search-sweeps" \
    --entity "neural-machines" \
    --count 10
```

### 2. Create Sweep Only (for multiple agents)

```bash
# Create sweep and get ID
python examples/run_sweep.py \
    --sweep-config sweep.yaml \
    --project "pc-arch-search-sweeps" \
    --entity "neural-machines" \
    --create-only
```

This will output a sweep ID like `abc123def`. You can then run multiple agents:

```bash
# Run multiple agents in parallel (different terminals/machines)
python examples/run_sweep.py \
    --sweep-id abc123def \
    --project "pc-arch-search-sweeps" \
    --entity "neural-machines" \
    --count 5

# Each agent will pick up different parameter combinations
```

### 3. Alternative: Direct sweep command

```bash
# Using wandb CLI directly
wandb sweep sweep.yaml
# This returns a sweep ID, then run:
wandb agent <sweep-id>
```

## Sweep Configuration

The `sweep.yaml` file defines your hyperparameter search space. Key sections:

### Search Method
```yaml
method: grid  # Options: grid, random, bayes
```

### Metric to Optimize
```yaml
metric:
  goal: minimize
  name: best_val_mse
```

### Parameter Types

**Grid Search Values:**
```yaml
num_blocks:
  values: [0, 1, 2, 3, 4, 5, 6]
```

**Random/Bayesian Ranges:**
```yaml
learning_rate:
  min: 0.001
  max: 0.1
  distribution: log_uniform_values
```

**Fixed Values:**
```yaml
batch_size:
  value: 200
```

## Modifying Search Space

### Adding New Parameters

1. Add to `sweep.yaml`:
```yaml
parameters:
  new_parameter:
    values: [option1, option2, option3]
```

2. Ensure the parameter exists in your `ModelConfig` class in `src/config.py`

### Changing Search Method

For large search spaces, consider switching from `grid` to `random` or `bayes`:

```yaml
method: random  # Much faster for large spaces
parameters:
  # ... your parameters
```

### Advanced: Bayesian Optimization

```yaml
method: bayes
metric:
  goal: minimize
  name: best_val_mse
early_terminate:
  type: hyperband
  min_iter: 5
```

## Monitoring and Analysis

### W&B Dashboard

1. Go to your W&B project dashboard
2. Click on "Sweeps" tab
3. View real-time progress, parallel coordinates plots, and parameter importance

### Key Metrics Tracked

- `best_val_mse` - Primary optimization target
- `best_train_mse` - Best training MSE achieved
- `final_train_mse` - Final training MSE
- `best_probe_accuracy` - Linear probe accuracy
- `early_stop_reason` - Why training stopped

## Scaling Up

### Multiple Machines

Run agents on different machines:

```bash
# Machine 1
python examples/run_sweep.py --sweep-id <ID> --project <PROJECT> --count 10

# Machine 2
python examples/run_sweep.py --sweep-id <ID> --project <PROJECT> --count 10
```

### Resource Management

Limit concurrent runs in sweep config:
```yaml
run_cap: 50  # Maximum total runs
```

Or use early termination:
```yaml
early_terminate:
  type: hyperband
  min_iter: 3
  eta: 2
```

## Migration from Old System

The old `hyperparam_search.py` searched these parameter combinations:
- Total combinations: 7 × 1 × 1 × 1 × 1 × 1 × 1 × 1 × 2 × 1 × 1 × 1 × 1 × 5 × 5 × 2 = **350 runs**

The new sweep system provides:
- ✅ Better visualization and monitoring
- ✅ Parallel execution across machines
- ✅ Early stopping of poor runs
- ✅ Bayesian optimization for smarter search
- ✅ Resume capability if interrupted

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure you're in the right directory and all dependencies are installed

2. **W&B authentication**: Run `wandb login` first

3. **YAML syntax**: Validate your `sweep.yaml` with an online YAML validator

4. **Memory issues**: Reduce `batch_size` or `hidden_size` in sweep config

### Debugging Individual Runs

Run a single experiment without sweep:
```bash
python examples/debug_transformer_wandb.py \
    --config 6block \
    --num_blocks 3 \
    --seed 42
```

## Best Practices

1. **Start small**: Test with a few parameter combinations first
2. **Use random search**: For >50 combinations, random often beats grid
3. **Monitor early**: Set up W&B alerts for completed sweeps
4. **Save good models**: The system automatically saves best models
5. **Clean up**: Delete failed/incomplete runs to keep dashboard clean

## Example Sweep Workflows

### Development (Small Search)
```yaml
method: grid
parameters:
  num_blocks: {values: [1, 2, 3]}
  seed: {values: [10, 20]}
# Total: 6 runs
```

### Production (Large Search)
```yaml
method: random
parameters:
  num_blocks: {values: [0, 1, 2, 3, 4, 5, 6]}
  intermediate_l1_coeff: {min: 0.0001, max: 0.1, distribution: log_uniform_values}
  intermediate_l2_coeff: {min: 0.0001, max: 0.1, distribution: log_uniform_values}
  seed: {values: [10, 20, 30, 40, 50]}
run_cap: 100  # Stop after 100 runs
``` 