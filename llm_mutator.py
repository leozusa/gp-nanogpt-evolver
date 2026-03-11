import json
import os
import subprocess
import sys
import pandas as pd
from config import HISTORY_SIZE, RESULTS_CSV
from genome import validate_genome

def get_mutation_prompt(parent_genome, history_df):
    """Constructs a prompt for the Gemini CLI to mutate the genome."""
    
    if not history_df.empty:
        sorted_df = history_df.sort_values(by="fitness", ascending=False)
        best = sorted_df.head(HISTORY_SIZE)
        worst = sorted_df.tail(HISTORY_SIZE)
        
        history_str = "\n### RECENT PERFORMANCE HISTORY\n"
        history_str += "Best Individuals:\n"
        for _, row in best.iterrows():
            # Exclude long text fields for prompt brevity
            r_dict = row.drop(labels=["mutation_reasoning", "checkpoint", "timestamp"], errors='ignore').to_dict()
            history_str += f"- Fitness: {row['fitness']:.4f} | Config: {r_dict}\n"
        
        history_str += "\nWorst Individuals:\n"
        for _, row in worst.iterrows():
            r_dict = row.drop(labels=["mutation_reasoning", "checkpoint", "timestamp"], errors='ignore').to_dict()
            history_str += f"- Fitness: {row['fitness']:.4f} | Config: {r_dict}\n"
    else:
        history_str = "\nNo history available yet. This is the first generation."

    prompt = f"""
You are an expert Neuroevolution Engineer. Your task is to mutate a GPT model configuration (genome) to improve its fitness.

### CURRENT PARENT GENOME:
{json.dumps(parent_genome, indent=2)}

{history_str}

### INSTRUCTIONS:
1. Analyze the parent genome and the performance history.
2. Propose a smart mutation:
   - Identify which parameters likely contribute to high fitness based on 'Best Individuals'.
   - Avoid configurations similar to 'Worst Individuals'.
   - Consider hardware constraints (M2 MacBook): stay efficient but powerful.
3. Return ONLY a JSON object representing the mutated genome.
4. Include a new field "mutation_reasoning" explaining WHY you made these changes (KEEP IT UNDER 20 WORDS, NO COMMAS, NO NEWLINES).
5. Ensure architectural constraints: 
   - n_embd must be divisible by n_head.
   - n_kv_head must be a divisor of n_head.
   - sequence_len should be between 128 and 2048.

### MUTATED GENOME (JSON ONLY):
"""
    return prompt

def mutate_with_llm(parent_genome):
    """Uses Gemini CLI to intelligently mutate a genome."""
    
    history_df = pd.DataFrame()
    if os.path.exists(RESULTS_CSV):
        try:
            # Handle bad lines gracefully
            history_df = pd.read_csv(RESULTS_CSV, on_bad_lines='skip')
        except Exception as e:
            pass

    prompt = get_mutation_prompt(parent_genome, history_df)
    
    try:
        result = subprocess.run(
            ["gemini", "--output-format", "json", prompt],
            capture_output=True,
            text=True,
            check=True
        )
        
        full_response = json.loads(result.stdout)
        raw_content = full_response.get("response", "")
        
        if "```json" in raw_content:
            raw_content = raw_content.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_content:
            raw_content = raw_content.split("```")[1].split("```")[0].strip()
            
        mutated_genome = json.loads(raw_content)
        
        # Enforce string constraints to protect CSV
        reasoning = str(mutated_genome.get("mutation_reasoning", "Standard mutation"))
        reasoning = reasoning.replace(",", ";").replace("\n", " ").replace("\r", " ")
        mutated_genome["mutation_reasoning"] = reasoning[:100] # Hard limit length
        
        mutated_genome["parent"] = parent_genome.get("individual_id", "unknown")
        # In steady state, children inherit their parent's generation number + 1
        mutated_genome["generation"] = int(parent_genome.get("generation", 0)) + 1
        
        return validate_genome(mutated_genome)
        
    except Exception as e:
        print(f"Gemini CLI Mutation failed: {e}. Falling back to random mutation.")
        return fallback_mutate(parent_genome)

def fallback_mutate(genome):
    """Simple random mutation as fallback."""
    import random
    new_genome = genome.copy()
    param_to_mutate = random.choice(["n_layer", "n_head", "n_embd", "matrix_lr", "weight_decay"])
    
    if param_to_mutate in ["n_layer", "n_head", "n_embd"]:
        if random.random() > 0.5:
            new_genome[param_to_mutate] += 1 if param_to_mutate == "n_layer" else (2 if param_to_mutate == "n_head" else 64)
        else:
            new_genome[param_to_mutate] -= 1 if param_to_mutate == "n_layer" else (2 if param_to_mutate == "n_head" else 64)
            new_genome[param_to_mutate] = max(1, new_genome[param_to_mutate])
    else:
        new_genome[param_to_mutate] *= random.uniform(0.8, 1.2)
        
    new_genome["mutation_reasoning"] = f"Fallback random mutation of {param_to_mutate}"
    new_genome["generation"] = int(genome.get("generation", 0)) + 1
    return validate_genome(new_genome)
