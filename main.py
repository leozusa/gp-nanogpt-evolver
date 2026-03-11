import argparse
import subprocess
import os
import sys
import config
from evolver import evolve

def main():
    parser = argparse.ArgumentParser(description="nanoGPT Evolver v1.0")
    parser.add_argument("--generations", type=int, default=20, help="Number of generations")
    parser.add_argument("--pop-size", type=int, default=12, help="Population size")
    parser.add_argument("--training-budget", type=int, default=10, help="Seconds of training per model")
    parser.add_argument("--use-llm-mutations", action="store_true", help="Enable LLM mutation logic")
    parser.add_argument("--dashboard-only", action="store_true", help="Only run the streamlit dashboard")
    parser.add_argument("--port", type=int, default=8501, help="Port for streamlit dashboard")
    
    args = parser.parse_args()

    # Update config values
    config.GENERATIONS = args.generations
    config.POPULATION_SIZE = args.pop_size
    config.TRAINING_BUDGET = args.training_budget
    if args.use_llm_mutations:
        config.USE_LLM_MUTATIONS = True

    if args.dashboard_only:
        print(f"Launching dashboard on port {args.port}...")
        subprocess.run(["streamlit", "run", "dashboard.py", "--server.port", str(args.port)])
        return

    print(f"Starting Evolution Process...")
    try:
        # Run evolution
        evolve()
    except KeyboardInterrupt:
        print("\nShutting down...")

if __name__ == "__main__":
    main()
