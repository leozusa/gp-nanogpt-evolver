import random
import multiprocessing as mp
import os
import time
import json
import shutil
import config
import pandas as pd
from config import POPULATION_SIZE, GENERATIONS, MAX_WORKERS, USE_LLM_MUTATIONS, RESULTS_CSV
from genome import create_random_genome, GENOME_KEYS, validate_genome
from fitness import run_fitness
from llm_mutator import mutate_with_llm, fallback_mutate

def evolve():
    """Steady-state evolutionary loop with Verbose Logging and Resume capability."""
    print(f"Starting Steady-State Evolution: {GENERATIONS} generations (approx), workers {MAX_WORKERS}", flush=True)
    
    os.makedirs("results", exist_ok=True)
    
    # --- RESUME LOGIC ---
    existing_results = pd.DataFrame()
    if os.path.exists(RESULTS_CSV):
        try:
            existing_results = pd.read_csv(RESULTS_CSV)
            print(f"Resuming from {len(existing_results)} existing results.", flush=True)
        except Exception as e:
            print(f"Error reading existing results: {e}", flush=True)

    population = []
    if not existing_results.empty:
        latest = existing_results.sort_values("timestamp", ascending=False)
        top_individuals = latest.sort_values("fitness", ascending=False).head(POPULATION_SIZE).to_dict('records')
        for ind in top_individuals:
            genome = {k: ind[k] for k in GENOME_KEYS if k in ind}
            population.append(genome)
        while len(population) < POPULATION_SIZE:
            population.append(create_random_genome())
    else:
        for i in range(POPULATION_SIZE):
            genome = create_random_genome()
            genome["generation"] = 0
            population.append(genome)

    pool = mp.Pool(processes=MAX_WORKERS)
    active_tasks = {}
    
    def launch_individual(genome, ind_idx):
        gen = genome.get("generation", 0)
        individual_id = f"gen{gen}_ind{ind_idx}_{int(time.time() * 1000)}"
        res = pool.apply_async(run_fitness, (genome, individual_id))
        active_tasks[res] = (genome, individual_id)
        # Explicit log for launches
        print(f"[{time.strftime('%H:%M:%S')}] Launched: {individual_id}", flush=True)

    # Initial launch
    for i in range(min(MAX_WORKERS, len(population))):
        launch_individual(population[i], i)
        
    completed_count = len(existing_results)
    total_to_run = (GENERATIONS * POPULATION_SIZE) + len(existing_results)
    
    best_fitness = -float("inf")
    if not existing_results.empty:
        best_fitness = existing_results["fitness"].max()

    try:
        while completed_count < total_to_run:
            finished_res = [res for res in active_tasks if res.ready()]
            
            for res in finished_res:
                old_genome, individual_id = active_tasks.pop(res)
                try:
                    metrics = res.get()
                    completed_count += 1
                    
                    ckpt_path = metrics.get("checkpoint")
                    fitness = metrics.get("fitness", -10.0)
                    
                    print(f"[{time.strftime('%H:%M:%S')}] Finished: {individual_id} | Fitness: {fitness:.4f} | Total: {completed_count}", flush=True)
                    
                    # Update best
                    if fitness > best_fitness:
                        best_fitness = fitness
                        print(f"\n*** NEW GLOBAL BEST: {best_fitness:.4f} (ID: {individual_id}) ***", flush=True)
                        
                        config_to_save = {k: old_genome[k] for k in GENOME_KEYS if k in old_genome}
                        with open(os.path.join("results", "best_config.json"), "w") as f:
                            json.dump(config_to_save, f, indent=2)
                        
                        if ckpt_path and os.path.exists(ckpt_path):
                            shutil.copy(ckpt_path, os.path.join("results", "best_model.pt"))
                    
                    if ckpt_path and os.path.exists(ckpt_path):
                        os.remove(ckpt_path)
                    
                    # Selection & Mutation
                    if os.path.exists(RESULTS_CSV):
                        try:
                            df = pd.read_csv(RESULTS_CSV).sort_values("fitness", ascending=False)
                            sample_size = max(5, int(len(df) * 0.2))
                            top_parents = df.head(sample_size).to_dict('records')
                            parent_data = random.choice(top_parents)
                            parent = {k: parent_data[k] for k in GENOME_KEYS if k in parent_data}
                        except:
                            parent = old_genome 
                    else:
                        parent = old_genome

                    if USE_LLM_MUTATIONS:
                        child = mutate_with_llm(parent)
                    else:
                        child = fallback_mutate(parent)
                    
                    launch_individual(child, completed_count % POPULATION_SIZE)
                    
                except Exception as e:
                    print(f"\n[{time.strftime('%H:%M:%S')}] Worker error: {e}", flush=True)
                    launch_individual(create_random_genome(), completed_count % POPULATION_SIZE)

            time.sleep(1) 
            
    except KeyboardInterrupt:
        print("\nEvolution halted by user.", flush=True)
    finally:
        pool.terminate()
        pool.join()

if __name__ == "__main__":
    evolve()
