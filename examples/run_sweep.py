#!/usr/bin/env python3
"""
W&B Sweep Runner for Hyperparameter Search

This script initializes and runs W&B sweeps for hyperparameter optimization.

Usage:
    python run_sweep.py --sweep-config sweep.yaml --project my-project --entity my-entity
    
Or to create and run a sweep in one go:
    python run_sweep.py --sweep-config sweep.yaml --project my-project --entity my-entity --count 10
"""

import argparse
import wandb
import yaml
from debug_transformer_wandb import run_sweep


def load_sweep_config(config_path):
    """Load sweep configuration from YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def create_sweep(sweep_config, project, entity=None):
    """Create a new W&B sweep."""
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project=project,
        entity=entity
    )
    return sweep_id


def run_sweep_agent(sweep_id, project, entity=None, count=None):
    """Run a sweep agent."""
    wandb.agent(
        sweep_id=sweep_id,
        function=run_sweep,
        project=project,
        entity=entity,
        count=count
    )


def main():
    parser = argparse.ArgumentParser(description='Run W&B hyperparameter sweeps')
    parser.add_argument('--sweep-config', type=str, required=True,
                        help='Path to sweep configuration YAML file')
    parser.add_argument('--project', type=str, required=True,
                        help='W&B project name')
    parser.add_argument('--entity', type=str, default=None,
                        help='W&B entity name (optional)')
    parser.add_argument('--sweep-id', type=str, default=None,
                        help='Existing sweep ID to join (if not provided, creates new sweep)')
    parser.add_argument('--count', type=int, default=None,
                        help='Number of runs to execute (if not provided, runs indefinitely)')
    parser.add_argument('--create-only', action='store_true',
                        help='Only create the sweep, don\'t run agent')
    
    args = parser.parse_args()
    
    # Load sweep configuration
    sweep_config = load_sweep_config(args.sweep_config)
    print(f"Loaded sweep config from {args.sweep_config}")
    
    # Create or use existing sweep
    if args.sweep_id:
        sweep_id = args.sweep_id
        print(f"Using existing sweep: {sweep_id}")
    else:
        sweep_id = create_sweep(sweep_config, args.project, args.entity)
        print(f"Created new sweep: {sweep_id}")
        print(f"W&B URL: https://wandb.ai/{args.entity or 'your-entity'}/{args.project}/sweeps/{sweep_id}")
    
    if args.create_only:
        print("Sweep created. Exiting (--create-only specified).")
        return
    
    # Run sweep agent
    print(f"Starting sweep agent...")
    if args.count:
        print(f"Will run {args.count} experiments")
    else:
        print("Will run indefinitely (use Ctrl+C to stop)")
    
    try:
        run_sweep_agent(sweep_id, args.project, args.entity, args.count)
    except KeyboardInterrupt:
        print("\nSweep agent stopped by user.")
    
    print("Sweep agent finished.")


if __name__ == "__main__":
    main() 