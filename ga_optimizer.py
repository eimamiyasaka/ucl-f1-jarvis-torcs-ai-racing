"""
Stage 2: Genetic Algorithm Optimizer
Population-based evolutionary search for optimal driving parameters.
"""

import sys
import os
import csv
import random
import numpy as np
from datetime import datetime

# Import baseline runner
from baseline_runner import run_episode, get_default_params, ResultLogger

# ================= PARAMETER BOUNDS =================
# [min, max] for each gene - 15 total chromosomes
PARAM_BOUNDS = {
    # Core driving parameters (4)
    'TARGET_SPEED': (50, 300),       # Target speed in km/h
    'STEER_GAIN': (1, 100),          # Steering sensitivity
    'CENTERING_GAIN': (0, 2),        # Track centering strength
    'BRAKE_THRESHOLD': (0.02, 1.5),  # Angle threshold for braking (radians)
    # Individual gear speed thresholds (5)
    'GEAR_2_SPEED': (30, 80),        # Speed to shift to gear 2
    'GEAR_3_SPEED': (60, 130),       # Speed to shift to gear 3
    'GEAR_4_SPEED': (100, 180),      # Speed to shift to gear 4
    'GEAR_5_SPEED': (140, 220),      # Speed to shift to gear 5
    'GEAR_6_SPEED': (180, 280),      # Speed to shift to gear 6
    # Throttle control parameters (3)
    'THROTTLE_INCREASE': (0.2, 0.8), # Acceleration rate when below target
    'THROTTLE_DECREASE': (0.1, 0.5), # Deceleration rate when above target
    'SPEED_STEER_FACTOR': (1.0, 5.0),# How steering affects target speed reduction
    # Brake control parameters (1)
    'BRAKE_INTENSITY': (0.1, 0.8),   # Braking force when threshold exceeded
    # Traction control parameters (2)
    'TC_THRESHOLD': (1, 10),         # Wheel spin differential to trigger TC
    'TC_REDUCTION': (0.05, 0.3),     # Throttle reduction when TC activates
}

# Gene indices for chromosome (15 genes)
GENE_NAMES = [
    # Core (4)
    'TARGET_SPEED', 'STEER_GAIN', 'CENTERING_GAIN', 'BRAKE_THRESHOLD',
    # Gears (5)
    'GEAR_2_SPEED', 'GEAR_3_SPEED', 'GEAR_4_SPEED', 'GEAR_5_SPEED', 'GEAR_6_SPEED',
    # Throttle (3)
    'THROTTLE_INCREASE', 'THROTTLE_DECREASE', 'SPEED_STEER_FACTOR',
    # Brake (1)
    'BRAKE_INTENSITY',
    # Traction control (2)
    'TC_THRESHOLD', 'TC_REDUCTION',
]

DNF_PENALTY = 999  # Penalty time for crashes/incomplete laps


# ================= CHROMOSOME CONVERSION =================
def params_to_chromosome(params):
    """Convert parameter dict to chromosome (15-element list)."""
    # Extract gear speeds from GEAR_SPEEDS list if present
    gear_speeds = params.get('GEAR_SPEEDS', [0, 50, 80, 120, 150, 200])

    return [
        # Core (4)
        params.get('TARGET_SPEED', 70),
        params.get('STEER_GAIN', 18),
        params.get('CENTERING_GAIN', 0.6),
        params.get('BRAKE_THRESHOLD', 0.2),
        # Gears (5) - extract from GEAR_SPEEDS list (indices 1-5)
        gear_speeds[1] if len(gear_speeds) > 1 else 50,
        gear_speeds[2] if len(gear_speeds) > 2 else 80,
        gear_speeds[3] if len(gear_speeds) > 3 else 120,
        gear_speeds[4] if len(gear_speeds) > 4 else 150,
        gear_speeds[5] if len(gear_speeds) > 5 else 200,
        # Throttle (3)
        params.get('THROTTLE_INCREASE', 0.4),
        params.get('THROTTLE_DECREASE', 0.2),
        params.get('SPEED_STEER_FACTOR', 2.5),
        # Brake (1)
        params.get('BRAKE_INTENSITY', 0.3),
        # Traction control (2)
        params.get('TC_THRESHOLD', 2.0),
        params.get('TC_REDUCTION', 0.1),
    ]


def chromosome_to_params(chromosome):
    """Convert chromosome (15-element list) to parameter dict."""
    return {
        # Core (4)
        'TARGET_SPEED': chromosome[0],
        'STEER_GAIN': chromosome[1],
        'CENTERING_GAIN': chromosome[2],
        'BRAKE_THRESHOLD': chromosome[3],
        # Gear speeds as list (gear 0 is always 0)
        'GEAR_SPEEDS': [0, chromosome[4], chromosome[5], chromosome[6],
                        chromosome[7], chromosome[8]],
        # Throttle (3)
        'THROTTLE_INCREASE': chromosome[9],
        'THROTTLE_DECREASE': chromosome[10],
        'SPEED_STEER_FACTOR': chromosome[11],
        # Brake (1)
        'BRAKE_INTENSITY': chromosome[12],
        # Traction control (2)
        'TC_THRESHOLD': chromosome[13],
        'TC_REDUCTION': chromosome[14],
        # Fixed parameter
        'ENABLE_TRACTION_CONTROL': True
    }


# ================= GENETIC OPERATORS =================
def random_chromosome():
    """Generate a random chromosome within bounds."""
    chromosome = []
    for gene_name in GENE_NAMES:
        low, high = PARAM_BOUNDS[gene_name]
        chromosome.append(random.uniform(low, high))
    # Repair gear speeds to be sorted
    return repair_chromosome(chromosome)


def repair_chromosome(chromosome):
    """
    Repair chromosome to satisfy constraints:
    - Clip all values to bounds
    - Ensure gear speeds are in ascending order
    """
    repaired = list(chromosome)

    # Clip to bounds
    for i, gene_name in enumerate(GENE_NAMES):
        low, high = PARAM_BOUNDS[gene_name]
        repaired[i] = max(low, min(high, repaired[i]))

    # Ensure gear speeds are in ascending order (indices 4-8)
    # GEAR_2 < GEAR_3 < GEAR_4 < GEAR_5 < GEAR_6
    gear_indices = [4, 5, 6, 7, 8]  # Indices for gear speeds in chromosome
    for i in range(1, len(gear_indices)):
        curr_idx = gear_indices[i]
        prev_idx = gear_indices[i - 1]
        # If current gear speed is not greater than previous, adjust it
        if repaired[curr_idx] <= repaired[prev_idx]:
            # Set to previous + small increment, but stay within bounds
            low, high = PARAM_BOUNDS[GENE_NAMES[curr_idx]]
            repaired[curr_idx] = min(high, repaired[prev_idx] + 10)

    return repaired


def tournament_select(population, fitnesses, tournament_size=3):
    """Select an individual using tournament selection."""
    indices = random.sample(range(len(population)), tournament_size)
    best_idx = max(indices, key=lambda i: fitnesses[i])
    return population[best_idx].copy()


def uniform_crossover(parent1, parent2, crossover_rate=0.8):
    """
    Perform uniform crossover between two parents.
    Each gene is randomly selected from one parent.
    """
    if random.random() > crossover_rate:
        # No crossover - return copies of parents
        return parent1.copy(), parent2.copy()

    child1 = []
    child2 = []

    for i in range(len(parent1)):
        if random.random() < 0.5:
            child1.append(parent1[i])
            child2.append(parent2[i])
        else:
            child1.append(parent2[i])
            child2.append(parent1[i])

    return child1, child2


def gaussian_mutate(chromosome, mutation_rate=0.15):
    """
    Apply Gaussian mutation to chromosome.
    Sigma is ~10% of parameter range.
    """
    mutated = chromosome.copy()

    for i, gene_name in enumerate(GENE_NAMES):
        if random.random() < mutation_rate:
            low, high = PARAM_BOUNDS[gene_name]
            # Gaussian mutation with sigma = 10% of range
            sigma = (high - low) * 0.1
            mutated[i] += random.gauss(0, sigma)

    return repair_chromosome(mutated)


# ================= FITNESS EVALUATION =================
def evaluate_fitness(chromosome, verbose=False, restart=True):
    """
    Evaluate fitness of a chromosome.
    Returns negative lap time (we maximize fitness, lower lap time is better).
    DNF returns -999.

    Args:
        chromosome: List of gene values
        verbose: Print progress info
        restart: If True, request race restart after evaluation. If False, just shutdown.
    """
    params = chromosome_to_params(chromosome)

    try:
        result = run_episode(params, target_laps=1, verbose=verbose, restart=restart)

        if result.get('dnf'):
            if verbose:
                print(f"  DNF: {result.get('dnf_reason', 'unknown')}")
            return -DNF_PENALTY

        if result['lap_time'] is None:
            return -DNF_PENALTY

        return -result['lap_time']  # Negative because we maximize
    except Exception as e:
        print(f"  Error during evaluation: {e}")
        return -DNF_PENALTY


# ================= GA LOGGING =================
class GALogger:
    """Logs generation-by-generation GA results to CSV."""

    def __init__(self, filepath='results/ga_log.csv'):
        self.filepath = filepath
        self._ensure_dir()
        self._write_header()
        self.generations_data = []

    def _ensure_dir(self):
        dir_path = os.path.dirname(self.filepath)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)

    def _write_header(self):
        with open(self.filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['generation', 'best_fitness', 'avg_fitness', 'worst_fitness']
            header += [f'best_{name}' for name in GENE_NAMES]
            writer.writerow(header)

    def log_generation(self, generation, fitnesses, best_chromosome):
        """Log a single generation's stats."""
        best_fit = max(fitnesses)
        avg_fit = sum(fitnesses) / len(fitnesses)
        worst_fit = min(fitnesses)

        row = [generation, best_fit, avg_fit, worst_fit]
        row += list(best_chromosome)

        with open(self.filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

        # Store for plotting
        self.generations_data.append({
            'generation': generation,
            'best_fitness': best_fit,
            'avg_fitness': avg_fit,
            'worst_fitness': worst_fit
        })

    def plot_convergence(self, output_path='results/ga_convergence.png'):
        """Generate convergence plot."""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt

            generations = [d['generation'] for d in self.generations_data]
            best = [d['best_fitness'] for d in self.generations_data]
            avg = [d['avg_fitness'] for d in self.generations_data]
            worst = [d['worst_fitness'] for d in self.generations_data]

            # Convert negative fitness back to lap times for display
            best_times = [-f if f > -DNF_PENALTY else None for f in best]
            avg_times = [-f for f in avg]

            plt.figure(figsize=(10, 6))
            plt.plot(generations, best_times, 'g-', label='Best Lap Time', linewidth=2)
            plt.plot(generations, avg_times, 'b--', label='Avg Fitness (neg)', alpha=0.7)

            plt.xlabel('Generation')
            plt.ylabel('Lap Time (seconds)')
            plt.title('GA Convergence - Lap Time Optimization')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Add best result annotation
            if best_times and best_times[-1] is not None:
                plt.annotate(f'Best: {best_times[-1]:.2f}s',
                           xy=(generations[-1], best_times[-1]),
                           xytext=(10, 10), textcoords='offset points')

            plt.tight_layout()
            plt.savefig(output_path, dpi=150)
            plt.close()

            print(f"Convergence plot saved to: {output_path}")
            return True

        except ImportError:
            print("Warning: matplotlib not available, skipping plot generation")
            return False


# ================= MAIN GA =================
def run_ga(
    population_size=30,
    generations=50,
    mutation_rate=0.15,
    crossover_rate=0.8,
    elite_count=3,
    tournament_size=3,
    seed=None,
    verbose=True,
    stagnation_limit=10
):
    """
    Run the Genetic Algorithm optimization.

    Args:
        population_size: Number of individuals (default 30)
        generations: Number of generations (default 50)
        mutation_rate: Per-gene mutation probability (default 0.15)
        crossover_rate: Probability of crossover (default 0.8)
        elite_count: Number of top individuals preserved (default 3)
        tournament_size: Tournament selection size (default 3)
        seed: Random seed for reproducibility
        verbose: Print progress (default True)
        stagnation_limit: Stop early if no improvement for this many generations (default 10, 0 to disable)

    Returns:
        dict with:
            - best_chromosome: Best solution found
            - best_fitness: Best fitness value
            - best_params: Best parameters as dict
            - best_lap_time: Best lap time in seconds
            - generations_run: Number of generations completed
            - converged: True if stopped due to convergence
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    logger = GALogger()

    if verbose:
        print("=" * 60)
        print("GENETIC ALGORITHM OPTIMIZER")
        print("=" * 60)
        print(f"Population size: {population_size}")
        print(f"Generations: {generations} (max)")
        print(f"Mutation rate: {mutation_rate}")
        print(f"Crossover rate: {crossover_rate}")
        print(f"Elite count: {elite_count}")
        print(f"Tournament size: {tournament_size}")
        print(f"Stagnation limit: {stagnation_limit} generations" if stagnation_limit > 0 else "Stagnation limit: disabled")
        print("=" * 60)

    # Initialize population
    if verbose:
        print("\nInitializing population...")

    population = [random_chromosome() for _ in range(population_size)]

    # Include default params as one individual (seeding with known good solution)
    default_chromosome = params_to_chromosome(get_default_params())
    population[0] = repair_chromosome(default_chromosome)

    best_ever_chromosome = None
    best_ever_fitness = float('-inf')
    generations_without_improvement = 0
    converged = False
    actual_generations = 0

    # Main GA loop
    for gen in range(generations):
        actual_generations = gen + 1
        if verbose:
            print(f"\n--- Generation {gen + 1}/{generations} ---")

        # Evaluate fitness for all individuals
        fitnesses = []
        is_final_generation = (gen == generations - 1)
        for i, chromosome in enumerate(population):
            if verbose:
                print(f"  Evaluating individual {i + 1}/{population_size}...", end=" ")

            # Only restart if there are more evaluations to come
            is_final_evaluation = is_final_generation and (i == population_size - 1)
            fitness = evaluate_fitness(
                chromosome,
                verbose=False,
                restart=not is_final_evaluation
            )
            fitnesses.append(fitness)

            if verbose:
                lap_time = -fitness if fitness > -DNF_PENALTY else "DNF"
                if isinstance(lap_time, float):
                    lap_time = f"{lap_time:.3f}s"
                print(f"Fitness: {fitness:.2f} (Lap: {lap_time})")

        # Find best in this generation
        gen_best_idx = np.argmax(fitnesses)
        gen_best_fitness = fitnesses[gen_best_idx]
        gen_best_chromosome = population[gen_best_idx]

        # Track best ever and convergence
        improved = False
        if gen_best_fitness > best_ever_fitness:
            best_ever_fitness = gen_best_fitness
            best_ever_chromosome = gen_best_chromosome.copy()
            generations_without_improvement = 0
            improved = True
        else:
            generations_without_improvement += 1

        # Log generation
        logger.log_generation(gen + 1, fitnesses, gen_best_chromosome)

        if verbose:
            best_time = -gen_best_fitness if gen_best_fitness > -DNF_PENALTY else "DNF"
            if isinstance(best_time, float):
                best_time = f"{best_time:.3f}s"
            avg_fitness = sum(fitnesses) / len(fitnesses)
            best_ever_time = -best_ever_fitness if best_ever_fitness > -DNF_PENALTY else "DNF"
            if isinstance(best_ever_time, float):
                best_ever_time = f"{best_ever_time:.3f}s"
            status = "NEW BEST!" if improved else f"(stagnant: {generations_without_improvement}/{stagnation_limit})"
            print(f"  Gen {gen + 1} Best: {best_time}, Best ever: {best_ever_time} {status}")

        # Check for convergence (early stopping)
        if stagnation_limit > 0 and generations_without_improvement >= stagnation_limit:
            converged = True
            if verbose:
                print(f"\n*** CONVERGED: No improvement for {stagnation_limit} generations ***")
            break

        # Check if we're at last generation
        if gen == generations - 1:
            break

        # Selection and reproduction
        # Sort by fitness (descending)
        sorted_indices = np.argsort(fitnesses)[::-1]

        # Elitism: keep top individuals
        new_population = []
        for i in range(elite_count):
            new_population.append(population[sorted_indices[i]].copy())

        # Fill rest with offspring
        while len(new_population) < population_size:
            # Tournament selection
            parent1 = tournament_select(population, fitnesses, tournament_size)
            parent2 = tournament_select(population, fitnesses, tournament_size)

            # Crossover
            child1, child2 = uniform_crossover(parent1, parent2, crossover_rate)

            # Mutation
            child1 = gaussian_mutate(child1, mutation_rate)
            child2 = gaussian_mutate(child2, mutation_rate)

            new_population.append(child1)
            if len(new_population) < population_size:
                new_population.append(child2)

        population = new_population

    # Generate convergence plot
    logger.plot_convergence()

    # Final results
    best_params = chromosome_to_params(best_ever_chromosome)
    best_lap_time = -best_ever_fitness if best_ever_fitness > -DNF_PENALTY else None

    if verbose:
        print("\n" + "=" * 60)
        print("OPTIMIZATION COMPLETE")
        print("=" * 60)
        if converged:
            print(f"Stopped early: converged after {actual_generations} generations")
        else:
            print(f"Completed all {actual_generations} generations")
        print(f"Best lap time: {best_lap_time:.3f}s" if best_lap_time else "Best: DNF")
        print(f"Best parameters:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
        print(f"\nResults logged to: {logger.filepath}")
        print("=" * 60)

    return {
        'best_chromosome': best_ever_chromosome,
        'best_fitness': best_ever_fitness,
        'best_params': best_params,
        'best_lap_time': best_lap_time,
        'generations_run': actual_generations,
        'converged': converged
    }


# ================= COMMAND LINE INTERFACE =================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='TORCS GA Optimizer')
    parser.add_argument('--population', type=int, default=30, help='Population size (default: 30)')
    parser.add_argument('--generations', type=int, default=50, help='Max generations (default: 50)')
    parser.add_argument('--mutation-rate', type=float, default=0.15, help='Mutation rate (default: 0.15)')
    parser.add_argument('--crossover-rate', type=float, default=0.8, help='Crossover rate (default: 0.8)')
    parser.add_argument('--elite', type=int, default=3, help='Elite count (default: 3)')
    parser.add_argument('--stagnation', type=int, default=10, help='Stop after N generations without improvement (default: 10, 0 to disable)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--quiet', action='store_true', help='Reduce output verbosity')
    args = parser.parse_args()

    # Clear sys.argv for torcs module
    sys.argv = [sys.argv[0]]

    result = run_ga(
        population_size=args.population,
        generations=args.generations,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        elite_count=args.elite,
        stagnation_limit=args.stagnation,
        seed=args.seed,
        verbose=not args.quiet
    )

    if result['converged']:
        print(f"\nConverged after {result['generations_run']} generations")
    print(f"Best lap time achieved: {result['best_lap_time']:.3f}s" if result['best_lap_time'] else "Best: DNF")
