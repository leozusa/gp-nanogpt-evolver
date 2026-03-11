# nanoGPT Evolver v1.0

A neuroevolution system that automatically discovers and optimizes GPT-style transformer architectures using an LLM-guided evolutionary algorithm. Designed specifically for Apple Silicon (M2/MPS acceleration), it continuously searches for the most efficient model configuration within strict time and parameter constraints.

## Features

*   **Steady-State Evolution:** A continuous, asynchronous evolutionary loop that ensures 100% GPU saturation on Apple Silicon by eliminating generational downtime.
*   **LLM-Guided Mutations:** Uses the Gemini CLI to act as an "expert neuroevolution engineer," analyzing performance history and proposing intelligent architectural mutations rather than relying purely on random chance.
*   **Real-Time Dashboard:** A Streamlit-based web interface to monitor the evolution trajectory, population health, and parameter importance correlations in real-time.
*   **Resume Capability:** Automatically saves progress to a CSV and resumes from the best-known individuals if the process is interrupted.
*   **Automated Checkpointing:** The "Global Best" individual is automatically saved to disk, allowing you to sample and interact with the model immediately.
*   **M2 Optimized:** Hard limits on parameter counts (<50M) and custom data loading tailored for Metal Performance Shaders (MPS).

## Project Structure

*   `evolver.py`: The core asynchronous evolutionary loop utilizing multiprocessing.
*   `llm_mutator.py`: The mutation engine interfacing with the Gemini CLI for smart parameter adjustments.
*   `fitness.py`: The evaluation script that surgically patches `nanoGPT` training code and extracts performance metrics.
*   `genome.py`: Defines the architectural search space and enforces divisibility/hardware constraints.
*   `dashboard.py`: The Streamlit application for real-time visualization.
*   `sample.py`: A script to chat with the best evolved model.
*   `install.sh`: Script to initialize submodules, install Python dependencies, and prepare the dataset.
*   `main.sh`: Shell script for starting, stopping, and checking the status of the background processes.
*   `autoresearch/`: Contains the base training scripts derived from nanoGPT.
*   `nanoGPT/`: Contains the base model definitions and data preparation scripts.

## Requirements

*   macOS with Apple Silicon (M1/M2/M3)
*   Python 3.10+
*   Gemini CLI (for LLM mutations)

## Quick Start

1.  **Clone the Repository:**

    ```bash
    git clone --recursive https://github.com/leozusa/gp-nanogpt-evolver.git
    cd gp-nanogpt-evolver
    ```

2.  **Install Dependencies and Prepare Data:**

    ```bash
    ./install.sh
    ```

3.  **Start the Evolution System:**

    ```bash
    ./main.sh start
    ```

    By default, this runs for 100 generations with a population of 12, allowing 15 seconds of training per model.

4.  **Monitor Progress:**

    Open your browser and navigate to the dashboard at `http://localhost:8501`. 
    You can also tail the evolution log to see live terminal output:

    ```bash
    tail -f evolution.log
    ```

5.  **Chat with the Best Model:**

    Once a "Global Best" is found, you can sample from it using:

    ```bash
    python sample.py --prompt "ROMEO:" --len 200
    ```

6.  **Stop the System:**

    ```bash
    ./main.sh stop
    ```
    Your progress is safely stored in `results/results.csv` and will resume automatically on the next start.

## Configuration

You can customize the run by passing flags to `main.sh`:

```bash
./main.sh start --generations 500 --pop-size 24 --training-budget 300 --port 8502
```

To run a purely random search (disabling the LLM mutator):

```bash
./main.sh start --no-llm
```

## Data

The current configuration uses the Shakespeare character-level dataset. Before running the evolution, ensure the data is prepared by running the preparation script inside the `nanoGPT/data/shakespeare_char` directory.
