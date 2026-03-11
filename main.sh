#!/bin/bash

# Default parameters
GENERATIONS=100
POP_SIZE=12
TRAINING_BUDGET=15
PORT=8501
USE_LLM_MUTATIONS="--use-llm-mutations"

print_usage() {
    echo "Usage: ./main.sh [start|stop|status] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  start     Start the evolution system and dashboard in the background"
    echo "  stop      Stop the evolution system and dashboard"
    echo "  status    Check if the processes are running"
    echo ""
    echo "Options (for 'start'):"
    echo "  --generations N       Number of generations (default: 100)"
    echo "  --pop-size N          Population size (default: 12)"
    echo "  --training-budget N   Seconds of training per model (default: 15)"
    echo "  --port N              Port for Streamlit dashboard (default: 8501)"
    echo "  --no-llm              Disable LLM mutations (fallback to random)"
    echo ""
}

stop_system() {
    echo "Stopping evolution system and dashboard..."
    pkill -f "python main.py"
    pkill -f "streamlit run"
    echo "Processes stopped."
}

status_system() {
    echo "--- System Status ---"
    if pgrep -f "python main.py" > /dev/null; then
        echo "Evolution: RUNNING"
    else
        echo "Evolution: STOPPED"
    fi
    if pgrep -f "streamlit run" > /dev/null; then
        echo "Dashboard: RUNNING"
    else
        echo "Dashboard: STOPPED"
    fi
}

setup_system() {
    echo "Checking submodules..."
    if [ -d ".git" ]; then
        git submodule update --init --recursive
    fi

    echo "Applying core patches to autoresearch..."
    cp core_patches/prepare_simple.py autoresearch/
    cp core_patches/model.py autoresearch/
    
    echo "Cleaning up any stale temp files..."
    rm -f autoresearch/train_tmp_*.py
}

start_system() {
    # Parse args
    while [[ "$#" -gt 0 ]]; do
        case $1 in
            --generations) GENERATIONS="$2"; shift ;;
            --pop-size) POP_SIZE="$2"; shift ;;
            --training-budget) TRAINING_BUDGET="$2"; shift ;;
            --port) PORT="$2"; shift ;;
            --no-llm) USE_LLM_MUTATIONS="" ;;
            *) echo "Unknown parameter passed: $1"; print_usage; exit 1 ;;
        esac
        shift
    done

    setup_system

    echo "Starting Streamlit Dashboard on port $PORT..."
    nohup streamlit run dashboard.py --server.port $PORT --server.headless true > dashboard.log 2>&1 &
    
    echo "Waiting for dashboard to initialize..."
    sleep 5
    
    echo "Starting Evolution Process with:"
    echo "  Generations: $GENERATIONS"
    echo "  Population Size: $POP_SIZE"
    echo "  Training Budget: ${TRAINING_BUDGET}s"
    if [ -n "$USE_LLM_MUTATIONS" ]; then echo "  LLM Mutations: Enabled"; else echo "  LLM Mutations: Disabled"; fi
    
    nohup python main.py --generations $GENERATIONS --pop-size $POP_SIZE --training-budget $TRAINING_BUDGET $USE_LLM_MUTATIONS --port $PORT > evolution.log 2>&1 &
    
    echo ""
    echo "System started successfully in the background."
    echo "Logs are being written to dashboard.log and evolution.log."
    echo "To view live evolution log: tail -f evolution.log"
}

if [ $# -eq 0 ]; then
    print_usage
    exit 1
fi

COMMAND=$1
shift

case $COMMAND in
    start)
        start_system "$@"
        ;;
    stop)
        stop_system
        ;;
    status)
        status_system
        ;;
    *)
        print_usage
        exit 1
        ;;
esac
