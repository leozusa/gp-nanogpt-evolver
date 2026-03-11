import random

# Fixed order for CSV columns to prevent misalignment
GENOME_KEYS = [
    "n_layer", "n_head", "n_kv_head", "n_embd", "sequence_len", 
    "window_pattern", "unembedding_lr", "embedding_lr", "matrix_lr", 
    "scalar_lr", "weight_decay", "generation", "parent", "mutation_reasoning"
]

METRIC_KEYS = [
    "val_bpb", "num_params_M", "mfu_percent", "num_steps", "fitness", 
    "individual_id", "timestamp"
]

def create_random_genome():
    """Returns a random genome within specified search space."""
    # Architectural Params
    n_embd = random.choice([256, 384, 512, 768])
    n_layer = random.randint(4, 16)
    n_head = random.choice([4, 6, 8, 12, 16])
    # Ensure n_embd is divisible by n_head
    while n_embd % n_head != 0:
         n_head = random.choice([4, 6, 8, 12, 16])

    # n_kv_head must be a divisor of n_head
    kv_options = [i for i in range(1, n_head + 1) if n_head % i == 0]
    n_kv_head = random.choice(kv_options)

    genome = {
        # Architecture
        "n_layer": n_layer,
        "n_head": n_head,
        "n_kv_head": n_kv_head,
        "n_embd": n_embd,
        "sequence_len": random.choice([128, 256, 512, 1024, 2048]),
        "window_pattern": "".join(random.choices(["S", "L"], k=4)), # Cyclic pattern of 4

        # Hyperparameters (Optimizers & Scalars)
        "unembedding_lr": random.uniform(1e-4, 1e-2),
        "embedding_lr": random.uniform(0.01, 0.5),
        "matrix_lr": random.uniform(0.005, 0.05),
        "scalar_lr": random.uniform(0.1, 1.0),
        "weight_decay": random.uniform(0.0, 0.1),

        # Metadata (Updated by evolver)
        "generation": 0,
        "parent": None,
        "mutation_reasoning": "Initial random population"
    }
    return genome

def validate_genome(genome):
    """Sanitize genome constraints."""
    # Enforce numeric types for architecture
    for k in ["n_layer", "n_head", "n_kv_head", "n_embd", "sequence_len", "generation"]:
        if k in genome:
            try:
                genome[k] = int(float(genome[k]))
            except:
                pass

    # Enforce divisibility
    if genome["n_embd"] % genome["n_head"] != 0:
        # Adjust n_embd to nearest multiple of n_head
        genome["n_embd"] = (genome["n_embd"] // genome["n_head"]) * genome["n_head"]
        if genome["n_embd"] == 0: genome["n_embd"] = genome["n_head"]

    # Enforce n_kv_head is divisor of n_head
    if genome["n_head"] % genome["n_kv_head"] != 0:
        kv_options = [i for i in range(1, genome["n_head"] + 1) if genome["n_head"] % i == 0]
        genome["n_kv_head"] = min(kv_options, key=lambda x: abs(x - genome["n_kv_head"]))

    # Enforce window pattern length
    if "window_pattern" not in genome or not isinstance(genome["window_pattern"], str) or len(genome["window_pattern"]) == 0:
        genome["window_pattern"] = "SSSL"

    return genome
