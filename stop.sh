#!/bin/bash

if [ -f training.pid ]; then
    PID=$(cat training.pid)
    if ps -p $PID > /dev/null; then
        kill $PID
        echo "🛑 Training stopped (PID: $PID)"
        rm -f training.pid
    else
        echo "❌ Training not running"
        rm -f training.pid
    fi
else
    echo "❌ No training process found"
fi
