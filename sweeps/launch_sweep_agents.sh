#!/bin/bash

# Configuration
SWEEP_ID="c1pmzydl"
PROJECT="12-blocks"  # Updated to match your actual project
NUM_AGENTS=3          # Reduced from 50 to fit in memory (35 × 2.5GB = 87.5GB)
COUNT_PER_AGENT=100     # Increased from 10 to maintain total work (35×15 = 525 vs 50×10 = 500)
LOG_DIR="sweep_logs"

# Create log directory
mkdir -p $LOG_DIR

echo "Launching $NUM_AGENTS sweep agents for sweep $SWEEP_ID"
echo "Logs will be saved to $LOG_DIR/"

# Launch agents in background
for i in $(seq 1 $NUM_AGENTS); do
    echo "Starting agent $i..."
    nohup python ../examples/run_sweep.py \
        --sweep-id $SWEEP_ID \
        --project $PROJECT \
        --count $COUNT_PER_AGENT \
        > $LOG_DIR/agent_${i}.log 2>&1 &
    
    # Store PID for later management
    echo $! > $LOG_DIR/agent_${i}.pid
    
    # Staggered delays: longer for first batch, shorter later
    if [ $i -le 10 ]; then
        sleep 30  # First 10 agents: 8 seconds (CUDA/JAX initialization)
    elif [ $i -le 20 ]; then
        sleep 30  # Next 10 agents: 5 seconds (W&B connections)
    else
        sleep 30  # Remaining agents: 3 seconds (basic startup)
    fi
    
    # Progress indicator
    if [ $((i % 5)) -eq 0 ]; then
        echo "  ... $i/$NUM_AGENTS agents started"
    fi
done

echo "All $NUM_AGENTS agents launched!"
echo "Monitor progress with: tail -f $LOG_DIR/agent_*.log"
echo "Check running processes: ps aux | grep run_sweep"
echo "Kill all agents: ./kill_sweep_agents.sh" 