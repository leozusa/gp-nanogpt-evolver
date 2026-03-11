# Project Structure v1.0

- `autoresearch/`: Core training and model files (from nanoGPT/MPS fork).
  - `train.py`: Main training script (patched by fitness.py).
  - `model.py`: Model definition.
  - `prepare.py`: Data preparation.
- `results/`: Directory for storing evolution results and logs.
  - `results.csv`: Log of all individuals and their fitness.
  - `best_model.pt`: Weights of the best individual found.
- `config.py`: Centralized configuration for the evolution and LLM.
- `genome.py`: Definition of the hybrid genome and search space.
- `fitness.py`: Evaluator that patches `train.py`, runs 5-min training, and parses metrics.
- `llm_mutator.py`: Intelligent mutation logic using LiteLLM (Groq/Grok).
- `evolver.py`: Evolutionary loop (DEAP + Multiprocessing).
- `dashboard.py`: Streamlit-based live visualization and analysis tool.
- `main.py`: Entry point for starting the evolution and/or dashboard.
- `instrucions.md`: Original requirements document.
