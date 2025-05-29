#!/bin/bash

LOG_DIR="sweep_logs"

echo "=== Sweep Agent Status ==="
echo "Active processes: $(ps aux | grep -c 'run_sweep.py.*0w4wlvej')"
echo "Log files: $(ls -1 $LOG_DIR/agent_*.log 2>/dev/null | wc -l)"
echo ""

echo "=== Recent Activity (last 10 lines from each agent) ==="
for log_file in $LOG_DIR/agent_*.log; do
    if [ -f "$log_file" ]; then
        agent_num=$(basename "$log_file" .log | sed 's/agent_//')
        echo "--- Agent $agent_num ---"
        tail -n 3 "$log_file" | grep -E "(Epoch|Loss|best|ERROR|WARNING)" || echo "No recent activity"
        echo ""
    fi
done

echo "=== GPU Memory Usage ==="
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits

echo "=== System Load ==="
uptime 