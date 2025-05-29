#!/bin/bash

LOG_DIR="sweep_logs"

echo "Stopping all sweep agents..."

# Kill processes using stored PIDs
for pid_file in $LOG_DIR/agent_*.pid; do
    if [ -f "$pid_file" ]; then
        PID=$(cat "$pid_file")
        if kill -0 "$PID" 2>/dev/null; then
            echo "Killing agent with PID $PID"
            kill "$PID"
        fi
        rm "$pid_file"
    fi
done

# Fallback: kill any remaining processes
pkill -f "run_sweep.py.*0w4wlvej"

echo "All agents stopped." 