import os
import math
from typing import Optional

import chex
import jax
import jax.numpy as jnp
import numpy as np

from huggingface_hub import hf_hub_download


DATASETS_BASE_PATH = "src/datasets"


def load_datasets(dataset_dirs: list[str], use_hf: bool) -> list[tuple[chex.Array, chex.Array, chex.Array]]:
    """Load datasets from the given directories.

    Args:
        dataset_dirs: List of directories containing the datasets.
        use_hf: Whether to use the HF hub to download the datasets.

    Returns:
        List of tuples containing the grids and shapes of the datasets.
    """

    datasets = []

    if use_hf:
        for dataset_dir in dataset_dirs:
            grids = np.load(
                hf_hub_download(
                    repo_id="arcenv/arc_datasets",
                    filename=os.path.join(dataset_dir, "grids.npy"),
                    repo_type="dataset",
                )
            ).astype("uint8")
            shapes = np.load(
                hf_hub_download(
                    repo_id="arcenv/arc_datasets",
                    filename=os.path.join(dataset_dir, "shapes.npy"),
                    repo_type="dataset",
                )
            ).astype("uint8")
            try:
                program_ids = np.load(
                    hf_hub_download(
                        repo_id="arcenv/arc_datasets",
                        filename=os.path.join(dataset_dir, "program_ids.npy"),
                        repo_type="dataset",
                    )
                ).astype("uint32")
            except:
                program_ids = jnp.zeros(grids.shape[0], dtype=jnp.uint32)

            datasets.append((grids, shapes, program_ids))

    else:
        for dataset_dir in dataset_dirs:
            grids = np.load(os.path.join(DATASETS_BASE_PATH, dataset_dir, "grids.npy")).astype("uint8")
            shapes = np.load(os.path.join(DATASETS_BASE_PATH, dataset_dir, "shapes.npy")).astype("uint8")

            try:
                program_ids = np.load(
                    os.path.join(DATASETS_BASE_PATH, dataset_dir, "program_ids.npy")
                ).astype("uint32")
            except:
                program_ids = jnp.zeros(grids.shape[0], dtype=jnp.uint32)

            datasets.append((grids, shapes, program_ids))
    return datasets


def shuffle_dataset_into_batches(
    dataset_grids: chex.Array, dataset_shapes: chex.Array, batch_size: int, key: chex.PRNGKey
) -> tuple[chex.Array, chex.Array]:
    if dataset_grids.shape[0] != dataset_shapes.shape[0]:
        raise ValueError("Dataset grids and shapes must have the same length.")

    # Shuffle the dataset.
    shuffled_indices = jax.random.permutation(key, len(dataset_grids))
    shuffled_grids = dataset_grids[shuffled_indices]
    shuffled_shapes = dataset_shapes[shuffled_indices]

    # Determine the number of batches.
    num_batches = len(dataset_grids) // batch_size
    if num_batches < 1:
        raise ValueError(f"Got dataset size: {len(dataset_grids)} < batch size: {batch_size}.")

    # Reshape the dataset into batches and crop the last batch if necessary.
    batched_grids = shuffled_grids[: num_batches * batch_size].reshape(
        (num_batches, batch_size, *dataset_grids.shape[1:])
    )
    batched_shapes = shuffled_shapes[: num_batches * batch_size].reshape(
        (num_batches, batch_size, *dataset_shapes.shape[1:])
    )

    return batched_grids, batched_shapes


def _apply_rotation(grid: chex.Array, grid_shape: chex.Array, k: int) -> tuple[chex.Array, chex.Array]:
    assert grid.ndim == 2 and grid_shape.ndim == 1

    def rotate_once(_, carry: tuple[chex.Array, chex.Array]) -> tuple[chex.Array, chex.Array]:
        grid, grid_shape = carry
        grid = jnp.rot90(grid, k=-1)
        # Roll the columns to the left until the first non-padded column is at the first position.
        num_rows = grid_shape[0].astype(jnp.int32)
        grid = jax.lax.fori_loop(
            0, grid.shape[0] - num_rows, lambda _, x: jnp.roll(x, shift=-1, axis=-1), grid
        )
        # Swap the rows and cols.
        grid_shape = grid_shape[::-1]
        return grid, grid_shape

    grid, grid_shape = jax.lax.fori_loop(0, k, rotate_once, (grid, grid_shape))
    return grid, grid_shape


def _apply_color_permutation(grid: chex.Array, key: chex.PRNGKey) -> chex.Array:
    # Exempt black (0).
    non_exempt = jnp.array([i for i in range(10) if i not in [0]], dtype=grid.dtype)
    permutation = jax.random.permutation(key, np.arange(len(non_exempt), dtype=grid.dtype))
    permuted_non_exempt = jnp.array([non_exempt[i] for i in permutation], dtype=grid.dtype)
    color_map = jnp.arange(10, dtype=grid.dtype).at[non_exempt].set(permuted_non_exempt)
    return color_map[grid]


def data_augmentation_fn(
    grids: chex.Array, shapes: chex.Array, key: chex.PRNGKey
) -> tuple[chex.Array, chex.Array]:
    """Apply data augmentation to the grids and shapes.

    Args:
        grids: The input grids. Shape (*B, N, R, C, 2).
        shapes: The shapes of the grids. Shape (*B, N, 2, 2).
        key: The random key.

    Returns:
        The augmented grids and shapes.
    """
    rotation_key, color_key = jax.random.split(key, 2)

    rotation_indices = jax.random.randint(rotation_key, grids.shape[:-4], 0, 4)
    apply_rotation_fn = _apply_rotation
    # vmap over the input/output channel (use same rotation).
    apply_rotation_fn = jax.vmap(apply_rotation_fn, in_axes=(-1, -1, None), out_axes=-1)
    # vmap over the pairs from the same task (use same rotation).
    apply_rotation_fn = jax.vmap(apply_rotation_fn, in_axes=(0, 0, None))
    # vmap over the batch dims if any (use different rotations).
    for _ in range(rotation_indices.ndim):
        apply_rotation_fn = jax.vmap(apply_rotation_fn)

    grids, shapes = apply_rotation_fn(grids, shapes, rotation_indices)

    color_keys = jax.random.split(color_key, grids.shape[:-4])
    apply_color_permutation_fn = _apply_color_permutation
    # vmap over the batch dims if any (use different color permutations).
    for _ in range(color_keys.ndim - 1):
        apply_color_permutation_fn = jax.vmap(apply_color_permutation_fn)

    grids = apply_color_permutation_fn(grids, color_keys)

    return grids, shapes


def make_leave_one_out(array: chex.Array, axis: int) -> chex.Array:
    """
    Args:
        array: shape (*B, N, *H).
        axis: The axis where N appears.

    Returns:
        Array of shape (*B, N, N-1, *H).
    """
    axis = axis % array.ndim
    output = []
    for i in range(array.shape[axis]):
        array_before = jax.lax.slice_in_dim(array, 0, i, axis=axis)
        array_after = jax.lax.slice_in_dim(array, i + 1, array.shape[axis], axis=axis)
        output.append(jnp.concatenate([array_before, array_after], axis=axis))
    output = jnp.stack(output, axis=axis)
    return output


def make_all_pairs(array: jnp.ndarray, axis: int = 1) -> jnp.ndarray:
    """
    Enumerate all distinct pairs along `axis` in a fully vectorized manner.

    Given an array of shape (*B, N, *H) along `axis` (the dimension of size N),
    this returns a new array of shape (*B, C, 2, *H), where
      - C = binomial_coefficient(N, 2) = N*(N-1)//2
      - The old dimension N is replaced by two dimensions: (C, 2).

    Args:
      array: A JAX array of shape (*B, N, *H), where `axis` is the N dimension.
      axis:  The axis along which to enumerate 2-combinations.

    Returns:
      A JAX array of shape (*B, N*(N-1)//2, 2, *H).
    """
    # Normalize `axis` in case it's negative
    axis = axis % array.ndim
    N = array.shape[axis]

    # Get all (i, j) with i < j; shape of i and j will be (C,) where C = N*(N-1)//2
    i, j = jnp.triu_indices(N, k=1)

    # Gather all "first elements" of pairs along the chosen axis
    e1 = jnp.take(array, i, axis=axis)  # shape: (*B, C, *H)
    # Gather all "second elements" of pairs
    e2 = jnp.take(array, j, axis=axis)  # shape: (*B, C, *H)

    # Stack them to get shape (*B, C, 2, *H)
    # We insert the new "pair" dimension (of size 2) just after the C dimension.
    # The dimension C is at `axis`, so we use `axis + 1` for the stacking axis.
    pairs = jnp.stack([e1, e2], axis=axis + 1)

    return pairs

def make_combinations_of_2(array: chex.Array, axis: int) -> chex.Array:
    """
    Creates pairs of elements along the specified axis.
    
    Args:
        array: shape (*B, N, *H).
        axis: The axis where N appears.
    Returns:
        Array of shape (*B, BinomialCoefficient(N,2), 2, *H).
        Each pair is a combination of 2 elements from the original array.
    """
    n = array.shape[axis]
    
    # Create all pairs of indices (i,j) where i < j
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Create output
    pairs_outputs = []
    
    for i, j in pairs:
        # Extract elements at indices i and j
        elem_i = jnp.take(array, i, axis=axis)
        elem_j = jnp.take(array, j, axis=axis)
        
        # Stack to form a pair
        pair = jnp.stack([elem_i, elem_j], axis=-1)
        pairs_outputs.append(pair)
    
    # Stack all pairs
    result = jnp.stack(pairs_outputs, axis=0)
    
    # Rearrange dimensions
    # Current: (C(N,2), *B(without N), *H, 2)
    # Target: (*B(up to N), C(N,2), 2, *B(after N), *H)
    
    # Move the C(N,2) dimension to position 'axis'
    result = jnp.moveaxis(result, 0, axis)
    
    # Move the 2 dimension from the end to position 'axis + 1'
    result = jnp.moveaxis(result, -1, axis + 1)
    
    return result



def _filter_pairs_without_element(array: jnp.ndarray, axis: int = 1, original_dim: int = None) -> jnp.ndarray:
    """
    For each element k in the original dimension N, select all pairs from R that don't include k.
    
    Args:
        array: A JAX array of shape (*B, R, *H), where R = N*(N-1)//2 is the number of pairs
               created by make_all_pairs on an axis of size N.
        axis: The axis where R appears.
        original_dim: The original dimension size N. If None, it will be inferred from R.
        
    Returns:
        A JAX array of shape (*B, N, (N-1)*(N-2)//2, *H), where each slice along the N axis
        contains all pairs that don't include the corresponding element.
    """
    # Normalize axis if negative
    axis = axis % array.ndim
    
    # Get the number of pairs R
    R = array.shape[axis]
    
    # Infer the original dimension N if not provided
    if original_dim is None:
        # Solve for N: R = N*(N-1)//2
        # This is a quadratic equation: N^2 - N - 2*R = 0
        # Using the quadratic formula: N = (1 + sqrt(1 + 8*R)) / 2
        N = int((1 + math.sqrt(1 + 8 * R)) / 2)
    else:
        N = original_dim
    
    # Generate all original pairs (i, j) with i < j, just like in make_all_pairs
    all_i, all_j = jnp.triu_indices(N, k=1)
    
    # For each element k, we want to select all pairs that don't include k
    results = []
    
    for k in range(N):
        # Find the indices of pairs that don't include element k
        mask = (all_i != k) & (all_j != k)
        pair_indices = jnp.where(mask)[0]
        
        # Gather these pairs from the input array
        filtered_pairs = jnp.take(array, pair_indices, axis=axis)
        results.append(filtered_pairs)
    
    # Stack the results along a new dimension at the beginning
    result = jnp.stack(results, axis=0)
    
    # Move the N dimension to position `axis` in the output
    result = jnp.moveaxis(result, 0, axis)
    
    return result


def filter_pairs_without_element(array: jnp.ndarray, axis: int = -2, original_dim: int = None) -> jnp.ndarray:
    """
    For each element k in the original dimension N, select all pairs from R that don't include k.
    This function is compatible with JAX transformations like jit.
    
    Args:
        array: A JAX array of shape (*B, R, *H), where R = N*(N-1)//2 is the number of pairs
               created by make_all_pairs on an axis of size N.
        axis: The axis where R appears.
        original_dim: The original dimension size N. If None, it will be inferred from R.
        
    Returns:
        A JAX array of shape (*B, N, (N-1)*(N-2)//2, *H), where each slice along the N axis
        contains all pairs that don't include the corresponding element.
    """
    # Normalize axis if negative
    ndim = array.ndim
    axis = axis if axis >= 0 else ndim + axis
    
    # Get the number of pairs R
    R = array.shape[axis]
    
    # Infer the original dimension N if not provided
    if original_dim is None:
        # Solve for N: R = N*(N-1)//2
        # This is a quadratic equation: N^2 - N - 2*R = 0
        # Using the quadratic formula: N = (1 + sqrt(1 + 8*R)) / 2
        N = int((1 + math.sqrt(1 + 8 * R)) / 2)
    else:
        N = original_dim
    
    # For each N, we'll create a static selection map
    # This approach avoids dynamic indexing that would break jit
    
    # Here's a function that generates a selection map for a given N
    # Each row k contains the indices of pairs that don't include element k
    # We only need to generate this once, and it's independent of the input array
    def get_selection_map(N):
        # Generate all pairs (i,j) with i < j
        all_pairs = []
        for i in range(N):
            for j in range(i + 1, N):
                all_pairs.push([i, j])
        
        # For each element k, find indices of pairs that don't include k
        selection_map = []
        for k in range(N):
            indices = []
            for p_idx, (i, j) in enumerate(all_pairs):
                if i != k and j != k:
                    indices.append(p_idx)
            selection_map.append(indices)
        
        return selection_map
    
    # Static selection maps for common N values
    # Using hardcoded selection maps for common N values
    # This is JAX-friendly because it avoids dynamic computation
    selection_maps = {
        3: [[2], [1], [0]],  # For N=3, R=3
        4: [[3, 4, 5], [1, 2, 5], [0, 2, 4], [0, 1, 3]],  # For N=4, R=6
        5: [[6, 7, 8, 9], [3, 4, 5, 9], [1, 2, 5, 8], [0, 2, 4, 7], [0, 1, 3, 6]],  # For N=5, R=10
        # Add more maps as needed for your specific use case
    }
    
    if N not in selection_maps:
        raise ValueError(f"Selection map for N={N} not pre-computed. Add it to the selection_maps dictionary.")
    
    # Get the selection map for our N
    current_map = jnp.array(selection_maps[N])
    
    # Using vmap and gather for JAX-friendly selection
    def select_pairs(k):
        # Get the selection indices for element k
        indices = current_map[k]
        
        # Use gather to select pairs
        # This is a JAX-friendly way to index
        selected = jnp.take(array, indices, axis=axis)
        
        return selected
    
    # Apply the selection function to each element index using vmap
    result = jax.vmap(select_pairs)(jnp.arange(N))
    
    # Move the N dimension to the right position
    # Currently: (N, (N-1)*(N-2)//2, *H) for a specific axis case
    # We want: (*B, N, (N-1)*(N-2)//2, *H)
    result = jnp.moveaxis(result, 0, axis)
    
    return result

if __name__ == "__main__":
    import time

    # Basic shape tests
    grids = jnp.ones((6,4,2))
    leave_one_out = make_leave_one_out(grids, axis=1)
    all_pairs = make_all_pairs(grids, axis=1)
    print(f"all_pairs shape: {all_pairs.shape}")
    filtered = filter_pairs_without_element(all_pairs, axis=1)
    filtered2 = _filter_pairs_without_element(all_pairs, axis=1)
    print(f"grids shape: {grids.shape}")
    print(f"filtered shape: {filtered.shape}")
    print(f"filtered2 shape: {filtered2.shape}")
    print(f"filtered == filtered2: {jnp.all(filtered == filtered2)}")
    all_pairs2 = make_all_pairs(leave_one_out, axis=2)
    print(f"all_pairs2 shape: {all_pairs2.shape}")

    print(f"all_pairs2 equal to filtered: {jnp.all(all_pairs2 == filtered)}")