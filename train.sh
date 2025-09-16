#!/bin/bash

# Run training in background, survives session disconnect
nohup python main.py > training.log 2>&1 &

# Get the process ID
PID=$!
echo "🚀 Training started in background"
echo "📊 Process ID: $PID"
echo "📝 Logs: tail -f training.log"
echo "🛑 Stop: kill $PID"

# Save PID to file for easy killing later
echo $PID > training.pid
echo "💾 PID saved to training.pid"
