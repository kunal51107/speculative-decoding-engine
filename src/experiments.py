import time
import torch
import matplotlib.pyplot as plt
from engine import SpeculativeEngine

def run_gamma_experiment():
    print(f"\n{'='*60}")
    print(f" EXPERIMENT: Finding the Perfect Gamma")
    print(f"{'='*60}")

    engine = SpeculativeEngine()
    
    # We use a consistent prompt so the comparison is fair
    prompt = "The rapid development of artificial intelligence has led to"
    max_tokens = 40
    
    # Test different draft lengths
    gammas = [1, 2, 3, 4, 5, 6]
    
    speeds = []
    acceptance_rates = []

    print("\nStarting stress test...")
    print("-" * 50)
    print(f"{'Gamma':<10} | {'Speed (t/s)':<15} | {'Acceptance %':<15}")
    print("-" * 50)

    for gamma in gammas:
        # Run the engine
        start_time = time.time()
        
        # We don't print the text output here, just the stats
        # We suppress the internal prints of the engine by not capturing them, 
        # but the engine prints to stdout. That's fine for now.
        _ = engine.generate(prompt, max_new_tokens=max_tokens, gamma=gamma)
        
        end_time = time.time()
        
        # Calculate Speed
        total_time = end_time - start_time
        speed = max_tokens / total_time
        
        # Get Acceptance Rate from the metrics we added
        acc_rate = engine.metrics['acceptance_rate'] * 100
        
        speeds.append(speed)
        acceptance_rates.append(acc_rate)
        
        print(f"{gamma:<10} | {speed:<15.2f} | {acc_rate:<15.2f}")

    print("-" * 50)
    print("✅ Experiment Complete. Generating Graph...")

    # --- PLOTTING THE DATA ---
    plt.figure(figsize=(10, 5))

    # Subplot 1: Speed vs Gamma
    plt.subplot(1, 2, 1)
    plt.plot(gammas, speeds, marker='o', color='b', linewidth=2)
    plt.title("Speed vs. Draft Length (Gamma)")
    plt.xlabel("Gamma (Draft Tokens)")
    plt.ylabel("Tokens / Second")
    plt.grid(True)

    # Subplot 2: Acceptance Rate vs Gamma
    plt.subplot(1, 2, 2)
    plt.plot(gammas, acceptance_rates, marker='o', color='g', linewidth=2)
    plt.title("Acceptance Rate vs. Gamma")
    plt.xlabel("Gamma (Draft Tokens)")
    plt.ylabel("Acceptance Rate (%)")
    plt.grid(True)

    plt.tight_layout()

    import os
    os.makedirs("results", exist_ok=True)  
    plt.savefig("results/gamma_analysis.png") 
    print(f"\n📊 Graph saved to: results/gamma_analysis.png")

if __name__ == "__main__":
    run_gamma_experiment()