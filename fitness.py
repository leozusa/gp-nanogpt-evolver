import subprocess
import os
import re
import time
import csv
import json
import sys
import config
from config import RESULTS_CSV
from genome import GENOME_KEYS, METRIC_KEYS

def run_fitness(genome, individual_id):
    """
    Patches train.py with genome, runs it, and returns metrics.
    Fitness is -val_bpb (higher is better).
    """
    train_path = os.path.join("autoresearch", "train.py")
    if not os.path.exists(train_path):
        return {"fitness": -10.0, "val_bpb": 10.0}

    with open(train_path, "r") as f:
        content = f.read()

    budget = getattr(config, "TRAINING_BUDGET", 10)
    
    # Check if we should save weights (passed via internal flag during evolver's "best" check)
    # Actually, easier: every worker saves to a temp file, and evolver renames it if it's the best.
    checkpoint_path = os.path.join("results", f"ckpt_{individual_id}.pt")

    injection = f"""
# --- GENOME INJECTION ---
from prepare_simple import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb

MAX_SEQ_LEN = {genome['sequence_len']}
DEPTH = {genome['n_layer']}
HEAD_DIM = {genome['n_embd'] // genome['n_head']}
WINDOW_PATTERN = "{genome['window_pattern']}"
EMBEDDING_LR = {genome['embedding_lr']}
UNEMBEDDING_LR = {genome['unembedding_lr']}
MATRIX_LR = {genome['matrix_lr']}
SCALAR_LR = {genome['scalar_lr']}
WEIGHT_DECAY = {genome['weight_decay']}
TIME_BUDGET = {budget}

def build_model_config(depth):
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
        n_layer={genome['n_layer']}, n_head={genome['n_head']}, 
        n_kv_head={genome['n_kv_head']}, n_embd={genome['n_embd']},
        window_pattern=WINDOW_PATTERN,
    )
# ------------------------
"""
    # Patch train.py to save the model at the end
    content = content.replace("from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb", "# Import removed")
    
    # Add weight saving at the very end of train.py
    save_logic = f"\ntorch.save(model.state_dict(), '{checkpoint_path}')\nprint('SAVED_WEIGHTS')"
    content += save_logic

    marker = "# Hyperparameters (edit these directly, no CLI flags needed)"
    new_content = content.replace(marker, injection + marker)

    temp_train_path = os.path.join("autoresearch", f"train_tmp_{individual_id}.py")
    with open(temp_train_path, "w") as f:
        f.write(new_content)

    metrics = {"val_bpb": 10.0, "num_params_M": 0.0, "mfu_percent": 0.0, "num_steps": 0}

    try:
        result = subprocess.run(
            [sys.executable, temp_train_path],
            capture_output=True,
            text=True,
            timeout=budget + 180 
        )
        stdout = result.stdout
        
        # Parse metrics...
        match_bpb = re.search(r"val_bpb:\s+([\d.]+)", stdout)
        if match_bpb: metrics["val_bpb"] = float(match_bpb.group(1))
        
        match_params = re.search(r"num_params_M:\s+([\d.]+)", stdout)
        if match_params: metrics["num_params_M"] = float(match_params.group(1))
        
        match_mfu = re.search(r"mfu_percent:\s+([\d.]+)", stdout)
        if match_mfu: metrics["mfu_percent"] = float(match_mfu.group(1))
        
        match_steps = re.search(r"num_steps:\s+([\d.]+)", stdout)
        if match_steps: metrics["num_steps"] = int(float(match_steps.group(1)))

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if os.path.exists(temp_train_path):
            os.remove(temp_train_path)

    metrics["fitness"] = -metrics["val_bpb"]
    metrics["checkpoint"] = checkpoint_path # Pass the path back
    save_to_csv(genome, metrics, individual_id)
    return metrics

def save_to_csv(genome, metrics, individual_id):
    file_exists = os.path.isfile(RESULTS_CSV)
    row = {**genome, **metrics, "individual_id": individual_id, "timestamp": time.time()}
    for k, v in row.items():
        if isinstance(v, (dict, list)): row[k] = json.dumps(v)
    fieldnames = GENOME_KEYS + METRIC_KEYS + ["checkpoint"]
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        if not file_exists: writer.writeheader()
        writer.writerow(row)
