"""
bfo_main.py

A simple / readable implementation of the Bacterial Foraging Optimization (BFO) algorithm.

Usage (example):
    python bfo_main.py --func rastrigin --dim 2 --pop 40 --chem_steps 30

Outputs:
    - Prints best solution and fitness
    - Saves plots via visualization module if available
"""
import argparse
import numpy as np
import time
from typing import Callable, Tuple

try:
    from objective_functions import FUNCTIONS, get_default_bounds
except Exception:
    raise

# Optional visualization import (if missing, program still runs)
try:
    from visualization import plot_fitness_history, plot_trajectories_2d, save_plots
    VIS_AVAILABLE = True
except Exception:
    VIS_AVAILABLE = False


def clamp(x: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(x, lb), ub)


class BFO:
    def __init__(
        self,
        func: Callable[[np.ndarray], float],
        dim: int,
        pop_size: int = 30,
        Nc: int = 20,  # chemotactic steps
        Ns: int = 4,  # swim length
        Nre: int = 4,  # reproduction steps
        Ned: int = 2,  # elimination-dispersal events
        Ped: float = 0.25,  # probability of elimination-dispersal
        step_size: float = 0.1,
        lb: np.ndarray = None,
        ub: np.ndarray = None,
        rng: np.random.Generator = None,
    ):
        self.func = func
        self.dim = dim
        self.pop_size = pop_size
        self.Nc = Nc
        self.Ns = Ns
        self.Nre = Nre
        self.Ned = Ned
        self.Ped = Ped
        self.step_size = step_size
        self.rng = rng or np.random.default_rng()
        # bounds
        if lb is None or ub is None:
            lb, ub = get_default_bounds("sphere", dim)
        self.lb = lb
        self.ub = ub

        # initialize population S x D
        self.positions = self.rng.uniform(lb, ub, size=(pop_size, dim))
        self.cost = np.array([self.func(p) for p in self.positions])
        self.best_idx = int(np.argmin(self.cost))
        self.best_pos = self.positions[self.best_idx].copy()
        self.best_cost = float(self.cost[self.best_idx])

        # history
        self.history_best = []

    def _attraction_repulsion(self, pos: np.ndarray) -> float:
        # Placeholder for interaction term; here we do NOT include inter-bacterial term
        return 0.0

    def run(self, verbose: bool = True):
        S = self.pop_size
        for ed in range(self.Ned):
            if verbose:
                print(f"[E&D {ed+1}/{self.Ned}] Starting elimination-dispersal cycle...")
            for re in range(self.Nre):
                if verbose:
                    print(f"  [Reprod {re+1}/{self.Nre}] Reproduction cycle...")
                # Chemotaxis loop
                for chem in range(self.Nc):
                    # For each bacterium
                    for i in range(S):
                        # tumble: generate random direction
                        delta = self.rng.normal(size=self.dim)
                        delta /= np.linalg.norm(delta) + 1e-12
                        # move a step
                        new_pos = clamp(self.positions[i] + self.step_size * delta, self.lb, self.ub)
                        new_cost = self.func(new_pos) + self._attraction_repulsion(new_pos)
                        # swim
                        swim_count = 0
                        if new_cost < self.cost[i]:
                            # accept first move
                            self.positions[i] = new_pos
                            self.cost[i] = new_cost
                            swim_count = 0
                            # continue swimming in same direction while improvement and Ns limit
                            while swim_count < self.Ns:
                                swim_count += 1
                                next_pos = clamp(self.positions[i] + self.step_size * delta, self.lb, self.ub)
                                next_cost = self.func(next_pos) + self._attraction_repulsion(next_pos)
                                if next_cost < self.cost[i]:
                                    self.positions[i] = next_pos
                                    self.cost[i] = next_cost
                                else:
                                    break
                        # update global best
                        cur_best_idx = int(np.argmin(self.cost))
                        if self.cost[cur_best_idx] < self.best_cost:
                            self.best_cost = float(self.cost[cur_best_idx])
                            self.best_pos = self.positions[cur_best_idx].copy()
                    # end for i
                    self.history_best.append(self.best_cost)
                # end chemotaxis
                # Reproduction: sort bacteria by health (sum of costs during chemotaxis)
                # Simple reproduction: keep best half and replicate
                order = np.argsort(self.cost)
                half = S // 2
                best_half = self.positions[order[:half]]
                # replicate
                self.positions[:half] = best_half
                self.positions[half:] = best_half.copy()
                # recompute costs
                self.cost = np.array([self.func(p) for p in self.positions])
            # end reproduction cycles
            # Elimination and dispersal
            for i in range(S):
                if self.rng.random() < self.Ped:
                    # relocate bacterium randomly
                    self.positions[i] = self.rng.uniform(self.lb, self.ub)
                    self.cost[i] = self.func(self.positions[i])
            # update best after E&D
            cur_best_idx = int(np.argmin(self.cost))
            if self.cost[cur_best_idx] < self.best_cost:
                self.best_cost = float(self.cost[cur_best_idx])
                self.best_pos = self.positions[cur_best_idx].copy()
        # end Ned
        if verbose:
            print("BFO finished.")
            print("Best cost:", self.best_cost)
            print("Best position:", self.best_pos)
        return self.best_pos, self.best_cost


def parse_args():
    p = argparse.ArgumentParser(description="Run a simple BFO optimization.")
    p.add_argument("--func", default="rastrigin", choices=list(FUNCTIONS.keys()))
    p.add_argument("--dim", type=int, default=2)
    p.add_argument("--pop", type=int, default=30)
    p.add_argument("--chem_steps", type=int, default=20)
    p.add_argument("--swim", type=int, default=4)
    p.add_argument("--reprod", type=int, default=4)
    p.add_argument("--ned", type=int, default=2)
    p.add_argument("--ped", type=float, default=0.25)
    p.add_argument("--step", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no-plot", action="store_true", help="Disable plotting even if visualization is available.")
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    f = FUNCTIONS[args.func]
    lb, ub = get_default_bounds(args.func, args.dim)
    bfo = BFO(
        func=f,
        dim=args.dim,
        pop_size=args.pop,
        Nc=args.chem_steps,
        Ns=args.swim,
        Nre=args.reprod,
        Ned=args.ned,
        Ped=args.ped,
        step_size=args.step,
        lb=lb,
        ub=ub,
        rng=rng,
    )
    t0 = time.time()
    best_pos, best_cost = bfo.run(verbose=True)
    dt = time.time() - t0
    print(f"Elapsed {dt:.2f}s — Best cost: {best_cost:.6f}")
    # plotting
    if VIS_AVAILABLE and not args.no_plot:
        try:
            plot_fitness_history(bfo.history_best, title=f"BFO fitness ({args.func})")
            if args.dim == 2:
                # show final positions and optionally trajectories if stored (here we only have positions)
                plot_trajectories_2d(bfo.positions, best_pos, title=f"Final positions ({args.func})")
            save_plots()
        except Exception as e:
            print("Visualization failed:", e)


if __name__ == "__main__":
    main()
