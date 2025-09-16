#!/bin/bash

if [ -f training.pid ]; then
    PID=$(cat training.pid)
    if ps -p $PID > /dev/null; then
        echo "✅ Training is running (PID: $PID)"
        echo "📊 MLflow UI: http://localhost:5000"
        echo "📝 View logs: tail -f training.log"
        echo "🛑 Stop: kill $PID"
    else
        echo "❌ Training not running"
        rm -f training.pid
    fi
else
    echo "❌ No training process found"
fi
