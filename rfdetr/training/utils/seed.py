"""
Seed management for reproducible experiments.

This module provides functions to set and manage random seeds across
Python, NumPy, and PyTorch for reproducible training experiments.
"""

import random
import os
from typing import Dict, Any, Optional

import numpy as np
import torch


def setup_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set random seed for reproducibility across all random number generators.
    
    Args:
        seed: The seed value to use.
        deterministic: If True, sets PyTorch to deterministic mode (may impact performance).
    
    Example:
        >>> setup_seed(42)
        >>> # Now all random operations will be reproducible
    """
    # Python random
    random.seed(seed)
    
    # Environment variable for some libraries
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch CPU
    torch.manual_seed(seed)
    
    # PyTorch CUDA (all GPUs)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Deterministic operations (may slow down training)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # Allow cuDNN to find optimal algorithms (faster but non-deterministic)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def get_random_state() -> Dict[str, Any]:
    """
    Get current random state from all random number generators.
    
    Returns:
        Dictionary containing random states for Python, NumPy, and PyTorch.
    
    Example:
        >>> state = get_random_state()
        >>> # ... do some random operations ...
        >>> set_random_state(state)  # Restore to previous state
    """
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch_cpu': torch.get_rng_state(),
    }
    
    if torch.cuda.is_available():
        state['torch_cuda'] = torch.cuda.get_rng_state_all()
    
    return state


def set_random_state(state: Dict[str, Any]) -> None:
    """
    Restore random state for all random number generators.
    
    Args:
        state: Dictionary containing random states (from get_random_state).
    
    Example:
        >>> state = get_random_state()
        >>> # ... do some random operations ...
        >>> set_random_state(state)  # Restore to previous state
    """
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'])
    
    if 'torch_cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['torch_cuda'])


def worker_init_fn(worker_id: int, base_seed: Optional[int] = None) -> None:
    """
    Initialize random seed for DataLoader workers.
    
    This function should be passed to DataLoader's worker_init_fn parameter
    to ensure each worker has a different but reproducible random seed.
    
    Args:
        worker_id: The worker ID (provided by DataLoader).
        base_seed: Base seed to use. If None, uses current NumPy random state.
    
    Example:
        >>> from functools import partial
        >>> loader = DataLoader(
        ...     dataset,
        ...     num_workers=4,
        ...     worker_init_fn=partial(worker_init_fn, base_seed=42)
        ... )
    """
    if base_seed is not None:
        seed = base_seed + worker_id
    else:
        # Use worker_id to differentiate workers
        seed = np.random.get_state()[1][0] + worker_id
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
