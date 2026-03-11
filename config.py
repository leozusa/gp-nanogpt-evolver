import os

# Evolution Parameters
POPULATION_SIZE = 12
GENERATIONS = 20
MAX_WORKERS = 6  
TRAINING_BUDGET = 10  # Default 10s for testing, can be 300 for "real" runs
RESULTS_CSV = os.path.join("results", "results.csv")

# Mutation Parameters
USE_LLM_MUTATIONS = True
HISTORY_SIZE = 10  

# Hardware
DEVICE = "mps"

# Genome Constraints
MAX_PARAMS = 50_000_000
MIN_SEQUENCE_LEN = 128
MAX_SEQUENCE_LEN = 2048
VOCAB_SIZE = 32768
