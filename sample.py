import torch
import os
import json
import sys
import importlib.util

# To avoid executing train.py on import, we surgically extract the classes
def get_train_classes():
    train_path = os.path.join(os.getcwd(), "autoresearch", "train.py")
    with open(train_path, "r") as f:
        lines = f.readlines()
    
    classes_code = ""
    in_class = False
    for line in lines:
        # Detect start of important classes
        if line.startswith("class ") or line.startswith("@dataclass"):
            in_class = True
        
        # Stop at the optimizer definition completely
        if line.startswith("class MuonAdamW"):
            in_class = False
            break
        
        if in_class:
            classes_code += line
            
    # Mocking required functions for the classes
    mock_header = """
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import math

def norm(x): return F.rms_norm(x, (x.size(-1),))
def has_ve(layer_idx, n_layer): return layer_idx % 2 == (n_layer - 1) % 2
def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1, y2 = x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)
"""
    full_code = mock_header + classes_code
    
    # Execute in a new module
    module_name = "train_classes"
    spec = importlib.util.spec_from_loader(module_name, loader=None)
    module = importlib.util.module_from_spec(spec)
    exec(full_code, module.__dict__)
    return module

train_classes = get_train_classes()
from autoresearch.prepare_simple import Tokenizer

def sample(prompt="JULIET:", max_len=100, temperature=0.8):
    config_path = os.path.join("results", "best_config.json")
    model_path = os.path.join("results", "best_model.pt")

    if not os.path.exists(config_path) or not os.path.exists(model_path):
        print("Error: Best model or config not found in results/. Run evolution first.")
        return

    with open(config_path, "r") as f:
        config_dict = json.load(f)

    tokenizer = Tokenizer.from_directory()
    vocab_size = tokenizer.get_vocab_size()

    conf = train_classes.GPTConfig(
        sequence_len=config_dict['sequence_len'],
        vocab_size=vocab_size,
        n_layer=config_dict['n_layer'],
        n_head=config_dict['n_head'],
        n_kv_head=config_dict['n_kv_head'],
        n_embd=config_dict['n_embd'],
        window_pattern=config_dict['window_pattern']
    )
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = train_classes.GPT(conf)
    
    print(f"Loading weights from {model_path}...")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.to(dtype=torch.bfloat16) # train.py uses bfloat16
    model.eval()

    stoi = tokenizer.stoi
    itos = tokenizer.itos
    x = torch.tensor([stoi.get(c, 0) for c in prompt], dtype=torch.long, device=device)[None, ...]

    print(f"\n--- GENERATED TEXT (Temp: {temperature}) ---\n")
    print(prompt, end="")
    
    with torch.no_grad():
        for _ in range(max_len):
            # Casting input to match model
            logits = model(x)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            probs = torch.softmax(logits.float(), dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, next_token), dim=1)
            char = itos[int(next_token[0, 0])]
            print(char, end="", flush=True)
    print("\n\n--- END ---")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="JULIET:")
    parser.add_argument("--len", type=int, default=200)
    parser.add_argument("--temp", type=float, default=0.8)
    args = parser.parse_args()
    sample(prompt=args.prompt, max_len=args.len, temperature=args.temp)
