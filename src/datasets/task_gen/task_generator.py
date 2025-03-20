import functools
import random
from typing import Any, Callable, Optional

import networkx as nx
import numpy as np
import torch
from torch.utils.data import IterableDataset

from src.datasets.task_gen.utils import is_grid, run_with_timeout
from src.datasets.task_gen.re_arc_generators import GENERATORS_SRC_CODE, ARC_TASK_NAMES


class PatternTaskGenerator(IterableDataset):
    def __init__(
        self,
        num_pairs: int,
        seed: Optional[int] = None,
        num_rows: int = 10,
        num_cols: int = 10,
        pattern_size: int = 4,
        pattern_density: float = 1.0,
        **kwargs,
    ):
        self.num_pairs = num_pairs
        self.seed = seed
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.pattern_size = pattern_size
        self.pattern_density = pattern_density
        assert 1 <= self.pattern_size < min(self.num_rows, self.num_cols)
        assert 0.0 < self.pattern_density <= 1.0

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_seed = self.seed + worker_info.id if self.seed is not None else None
        else:
            worker_seed = self.seed
        if worker_seed is not None:
            random.seed(worker_seed)
        return self

    def __next__(self) -> tuple[list[dict[str, tuple]], dict[str, Any]]:
        task = []
        pattern = self.generate_pattern()
        for _ in range(self.num_pairs):
            pair = self.generate_pair(pattern)
            task.append(pair)
        info = {"num_attempts_generate_task": 1, "G": nx.MultiDiGraph()}
        return task, info

    def generate_pattern(self) -> np.ndarray:
        pattern = np.zeros((self.pattern_size, self.pattern_size), dtype=int)
        for i in range(self.pattern_size):
            for j in range(self.pattern_size):
                if random.random() < self.pattern_density:
                    pattern[i, j] = random.randint(1, 9)
        return pattern

    def generate_pair(self, pattern: np.ndarray) -> dict[str, np.ndarray]:
        input_grid = np.zeros((self.num_rows, self.num_cols), dtype=int)
        output_grid = np.zeros((self.num_rows, self.num_cols), dtype=int)
        pattern_loc_row = random.randint(0, self.num_rows - self.pattern_size)
        pattern_loc_col = random.randint(0, self.num_cols - self.pattern_size)
        input_grid[pattern_loc_row, pattern_loc_col] = 1
        output_grid[
            pattern_loc_row : pattern_loc_row + self.pattern_size,
            pattern_loc_col : pattern_loc_col + self.pattern_size,
        ] = pattern
        return {"input": input_grid, "output": output_grid}

class PatternTaskGeneratorHard(IterableDataset):
    def __init__(
        self,
        num_pairs: int,
        seed: Optional[int] = None,
        num_rows: int = 10,
        num_cols: int = 10,
        pattern_size: int = 4,
        pattern_density: float = 1.0,
        **kwargs,
    ):
        self.num_pairs = num_pairs
        self.seed = seed
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.pattern_size = pattern_size
        self.pattern_density = pattern_density
        assert 1 <= self.pattern_size < min(self.num_rows, self.num_cols)
        assert 0.0 < self.pattern_density <= 1.0

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_seed = self.seed + worker_info.id if self.seed is not None else None
        else:
            worker_seed = self.seed
        if worker_seed is not None:
            random.seed(worker_seed)
        return self

    def __next__(self) -> tuple[list[dict[str, tuple]], dict[str, Any]]:
        task = []
        colors = list(range(1, 9))
        random.shuffle(colors)
        color_function = self._get_color_function(colors)
        pointer_color, pattern_colors, background_colors = colors[0], colors[1:4], colors[4:]
        pattern = self.generate_pattern(pattern_colors)
        for _ in range(self.num_pairs):
            pair = self.generate_pair(pattern, background_colors, pointer_color, color_function)
            task.append(pair)
        info = {"num_attempts_generate_task": 1, "G": nx.MultiDiGraph()}
        return task, info

    def generate_pattern(self, colors: list[int]) -> np.ndarray:
        pattern = np.zeros((self.pattern_size, self.pattern_size), dtype=int)
        for i in range(self.pattern_size):
            for j in range(self.pattern_size):
                if random.random() < self.pattern_density:
                    pattern[i, j] = random.choice(colors)
        return pattern

    def generate_pair(self, pattern: np.ndarray, colors: list[int], pointer_color: int, color_function: Callable[[int], int]) -> dict[str, np.ndarray]:
        input_grid = np.random.choice(colors, size=(self.num_rows, self.num_cols), replace=True)
        output_grid = np.zeros((self.num_rows, self.num_cols), dtype=int)
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                output_grid[i, j] = color_function(input_grid[i, j])
        pattern_loc_row = random.randint(0, self.num_rows - self.pattern_size)
        pattern_loc_col = random.randint(0, self.num_cols - self.pattern_size)
        input_grid[pattern_loc_row, pattern_loc_col] = pointer_color
        output_grid[
            pattern_loc_row : pattern_loc_row + self.pattern_size,
            pattern_loc_col : pattern_loc_col + self.pattern_size,
        ] = pattern
        return {"input": input_grid, "output": output_grid}

    def _get_color_function(self, colors: list[int]) -> Callable[[int], int]:
        if random.random() < 0.25:
            def color_function(x: int) -> int:
                return x
        else:
            color_map = {c: random.choice(colors) for c in colors}
            def color_function(x: int) -> int:
                return color_map[x]
        return color_function

class CellularAutomataTaskGenerator(IterableDataset):
    def __init__(
        self,
        num_pairs: int,
        seed: Optional[int] = None,
        num_rows: int = 10,
        num_cols: int = 10,
        **kwargs,
    ):
        self.num_pairs = num_pairs
        self.seed = seed
        self.num_rows = num_rows
        self.num_cols = num_cols
        
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_seed = self.seed + worker_info.id if self.seed is not None else None
        else:
            worker_seed = self.seed
        if worker_seed is not None:
            random.seed(worker_seed)
        return self
    
    def __next__(self) -> tuple[list[dict[str, tuple]], dict[str, Any]]:
        task = []
        
        # Choose two colors to represent 0 and 1
        all_colors = list(range(1, 9))
        random.shuffle(all_colors)
        color_0, color_1 = all_colors[0], all_colors[1]
        
        # Choose a direction pattern (horizontal, vertical, diagonal1, diagonal2)
        direction = random.choice(["horizontal", "vertical", "diagonal1", "diagonal2"])
        
        # Create a random mapping for the 8 possible neighbor configurations
        # Each configuration maps to either 0 or 1 (represented by our chosen colors)
        rule = {i: random.choice([0, 1]) for i in range(8)}
        
        for _ in range(self.num_pairs):
            pair = self.generate_pair(color_0, color_1, direction, rule)
            task.append(pair)
            
        info = {
            "num_attempts_generate_task": 1, 
            "G": nx.MultiDiGraph(),
            "direction": direction,
            "color_0": color_0,
            "color_1": color_1
        }
        return task, info
    
    def generate_pair(self, color_0: int, color_1: int, direction: str, rule: dict) -> dict[str, np.ndarray]:
        """Generate a pair of input and output grids using the given rule."""
        # Generate a random binary input grid (using our two colors)
        binary_grid = np.random.choice([0, 1], size=(self.num_rows, self.num_cols))
        input_grid = np.where(binary_grid == 0, color_0, color_1)
        
        # Apply the cellular automata rule to generate the output grid
        output_binary = np.zeros((self.num_rows, self.num_cols), dtype=int)
        
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                # Get the relevant neighbors based on the direction
                neighbors = self._get_neighbors(binary_grid, i, j, direction)
                
                # Convert neighbors to an index (treating them as binary digits)
                index = 0
                for k, neighbor in enumerate(neighbors):
                    if neighbor == 1:
                        index |= (1 << k)
                
                # Apply the rule to determine the new state
                output_binary[i, j] = rule[index]
        
        # Convert binary output to colors
        output_grid = np.where(output_binary == 0, color_0, color_1)
                
        return {"input": input_grid, "output": output_grid}
    
    def _get_neighbors(self, grid: np.ndarray, row: int, col: int, direction: str) -> list:
        """Get the three relevant neighbors based on the direction."""
        neighbors = []
        
        if direction == "horizontal":
            # Left, center, right
            neighbors.append(grid[row, col-1] if col > 0 else 0)
            neighbors.append(grid[row, col])
            neighbors.append(grid[row, col+1] if col < self.num_cols-1 else 0)
        
        elif direction == "vertical":
            # Top, center, bottom
            neighbors.append(grid[row-1, col] if row > 0 else 0)
            neighbors.append(grid[row, col])
            neighbors.append(grid[row+1, col] if row < self.num_rows-1 else 0)
        
        elif direction == "diagonal1":
            # Top-left, center, bottom-right
            neighbors.append(grid[row-1, col-1] if row > 0 and col > 0 else 0)
            neighbors.append(grid[row, col])
            neighbors.append(grid[row+1, col+1] if row < self.num_rows-1 and col < self.num_cols-1 else 0)
        
        elif direction == "diagonal2":
            # Top-right, center, bottom-left
            neighbors.append(grid[row-1, col+1] if row > 0 and col < self.num_cols-1 else 0)
            neighbors.append(grid[row, col])
            neighbors.append(grid[row+1, col-1] if row < self.num_rows-1 and col > 0 else 0)
        
        return neighbors

class CombinedTaskGenerator(IterableDataset):
    """A task generator that combines multiple task generators and randomly selects one when __next__ is called."""
    
    def __init__(
        self,
        generators: list,
        probabilities: Optional[list[float]] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize a combined task generator.
        
        Args:
            generators: List of task generator instances to choose from
            probabilities: Optional list of probabilities for selecting each generator.
                           If None, generators will be selected with equal probability.
            seed: Random seed for reproducibility
        """
        self.generators = generators
        self.seed = seed
        
        if probabilities is None:
            # Equal probability for all generators
            self.probabilities = [1.0 / len(generators)] * len(generators)
        else:
            # Normalize probabilities to sum to 1
            total = sum(probabilities)
            self.probabilities = [p / total for p in probabilities]
            
        assert len(self.generators) == len(self.probabilities), "Number of generators must match number of probabilities"
        assert abs(sum(self.probabilities) - 1.0) < 1e-10, "Probabilities must sum to 1"
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_seed = self.seed + worker_info.id if self.seed is not None else None
        else:
            worker_seed = self.seed
            
        if worker_seed is not None:
            random.seed(worker_seed)
            
        # Initialize iterators for all generators
        self.iterators = [iter(gen) for gen in self.generators]
        return self
    
    def __next__(self) -> tuple[list[dict[str, tuple]], dict[str, Any]]:
        # Randomly select a generator based on probabilities
        generator_idx = random.choices(range(len(self.generators)), weights=self.probabilities, k=1)[0]
        
        # Get the next task from the selected generator
        task, info = next(self.iterators[generator_idx])
        
        # Add information about which generator was used
        info["generator_idx"] = generator_idx
        info["generator_type"] = type(self.generators[generator_idx]).__name__
        
        return task, info


class ArcTrainTaskGenerator(IterableDataset):
    def __init__(
        self,
        num_pairs: int,
        seed: Optional[int] = None,
        timeout_generate_pair: int = 5,
        overfit_task: Optional[str] = None,
        only_n_tasks: Optional[int] = None,
        max_rows: int = 30,
        max_cols: int = 30,
        **kwargs,
    ):
        self.num_pairs = num_pairs
        self.seed = seed
        self.timeout_generate_pair = timeout_generate_pair
        self.random_state = None
        self.generate_functions = []
        if overfit_task is not None and only_n_tasks is not None:
            raise ValueError("Cannot specify both overfit_task and only_n_tasks.")
        self.overfit_task = overfit_task
        self.only_n_tasks = only_n_tasks
        self.task_names = ARC_TASK_NAMES
        self.max_rows = max_rows
        self.max_cols = max_cols
        if only_n_tasks is not None:
            self.task_names = self.task_names[:only_n_tasks]

    def __iter__(self):
        exec(GENERATORS_SRC_CODE, globals())  # add the generate functions to the global namespace
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_seed = self.seed + worker_info.id if self.seed is not None else None
        else:
            worker_seed = self.seed
        if worker_seed is not None:
            random.seed(worker_seed)
        self.random_state = random.getstate()
        if self.overfit_task is not None:
            task_fn_name = f"generate_{self.overfit_task}"
            assert task_fn_name in globals(), f"Function {task_fn_name} not found."
            self.generate_functions = [functools.partial(globals()[task_fn_name], 0, 0)]
        else:
            self.generate_functions = [
                functools.partial(globals()[f"generate_{task_name}"], 0, 0) for task_name in self.task_names
            ]
        return self

    def __next__(self) -> tuple[list[dict[str, tuple]], dict[str, Any]]:
        stop = False
        num_attempts = 0
        while not stop:
            stop = True
            num_attempts += 1
            program_id = random.randint(0, len(self.generate_functions) - 1)
            generate_fn = self.generate_functions[program_id]
            task = []
            for _ in range(self.num_pairs):
                try:
                    if self.timeout_generate_pair:
                        # Use a signal to run the function with a timeout
                        pair, self.random_state, exception = run_with_timeout(
                            generate_fn, timeout=self.timeout_generate_pair
                        )(random_state=self.random_state)
                        if exception is not None:
                            raise exception
                    else:
                        # Run the function without a timeout
                        pair = generate_fn()
                except KeyboardInterrupt:
                    raise
                except Exception:
                    stop = False
                    break
                if not is_grid(pair["input"]) or not is_grid(pair["output"]):
                    stop = False
                    break
                if not (len(pair["input"]) <= self.max_rows and len(pair["input"][0]) <= self.max_cols):
                    stop = False
                    break
                task.append({key: np.array(value) for key, value in pair.items()})
        info = {"num_attempts_generate_task": num_attempts, "program_id": program_id}
        return task, info


if __name__ == "__main__":
    from src.datasets.task_gen.utils import plot_task

    task_gen = ArcTrainTaskGenerator(num_pairs=4, seed=None)
    task, info = next(iter(task_gen))
    print(f"Generated a valid task after {info['num_attempts_generate_task']} attempts.")
    plot_task(task, figsize_factor=2)
    if "program_id" in info:
        print(f"Program ID: {info['program_id']}")
