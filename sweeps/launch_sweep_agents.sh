#!/bin/bash

# --- Configuration ---
SWEEP_CONFIG_FILE="../sweeps/sweep_kornia.yaml"
PROJECT_NAME="6-blocks-kornia-reproducibility"
ENTITY_NAME="neural-machines"
# Generate a unique name for the sweep to avoid conflicts
SWEEP_NAME="kornia-repro-$(date +%Y%m%d-%H%M%S)"

NUM_AGENTS=10
COUNT_PER_AGENT=1  # Each agent will try to run 1 config from the grid
LOG_DIR="sweep_logs"
# --- End Configuration ---

# Create log directory
mkdir -p $LOG_DIR

echo "--- Step 1: Creating new W&B Sweep ---"
echo "Config: $SWEEP_CONFIG_FILE"
echo "Project: $PROJECT_NAME"
echo "Sweep Name: $SWEEP_NAME"

# Create the sweep and capture the full output
# The command now points to the correct location of run_sweep.py from the sweeps directory
CREATION_OUTPUT=$(python ../examples/run_sweep.py --sweep-config "$SWEEP_CONFIG_FILE" --project "$PROJECT_NAME" --entity "$ENTITY_NAME" --sweep-name "$SWEEP_NAME" --create-only)

# Extract the sweep ID by looking for the specific "Created new sweep:" line.
# This is more robust than parsing the URL.
SWEEP_ID=$(echo "$CREATION_OUTPUT" | grep "Created new sweep:" | awk '{print $4}')

# Extract the sweep ID from the URL. The ID is the last part of the URL path.
if [ -n "$SWEEP_ID" ]; then
    echo "Successfully created sweep with ID: $SWEEP_ID"
    # Also print the URL for easy access
    echo "$CREATION_OUTPUT" | grep "https://wandb.ai/"
else
    echo "Error: Failed to create sweep or capture the W&B sweep ID."
    echo "--- Full output from creation command ---"
    echo "$CREATION_OUTPUT"
    echo "-----------------------------------------"
    exit 1
fi

echo "Waiting 5 seconds for sweep to register on W&B servers..."
sleep 5

echo -e "\n--- Step 2: Launching $NUM_AGENTS sweep agents ---"
echo "Logs will be saved to $LOG_DIR/"

# Launch agents in background
for i in $(seq 1 $NUM_AGENTS); do
    echo "Starting agent $i..."
    # The command now points to the correct location of run_sweep.py from the sweeps directory
    nohup python ../examples/run_sweep.py \
        --sweep-id "$SWEEP_ID" \
        --project "$PROJECT_NAME" \
        --entity "$ENTITY_NAME" \
        --count "$COUNT_PER_AGENT" \
        > "$LOG_DIR/agent_${i}.log" 2>&1 &
    
    # Store PID for later management
    echo $! > "$LOG_DIR/agent_${i}.pid"
    
    # Staggered delay to prevent all agents from hitting the W&B API at once
    sleep 5
    
    # Progress indicator
    if [ $((i % 5)) -eq 0 ]; then
        echo "  ... $i/$NUM_AGENTS agents started"
    fi
done

echo -e "\n--- Launch Complete ---"
echo "All $NUM_AGENTS agents launched!"
echo "Monitor progress with: tail -f $LOG_DIR/agent_*.log"
echo "Check running processes: ps aux | grep run_sweep"
# Assuming you have a script to kill them based on the PIDs
echo "To kill all agents, run your kill script (e.g., ./kill_sweep_agents.sh)" 