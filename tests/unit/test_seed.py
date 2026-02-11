"""
Tests for seed management functionality.
"""

import sys
import os
import random

import numpy as np
import pytest
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rfdetr.training.utils.seed import (
    setup_seed,
    get_random_state,
    set_random_state,
    worker_init_fn,
)


class TestSetupSeed:
    """Tests for setup_seed function."""
    
    def test_python_random_reproducibility(self):
        """Test that Python random is reproducible after setup_seed."""
        setup_seed(42)
        values1 = [random.random() for _ in range(10)]
        
        setup_seed(42)
        values2 = [random.random() for _ in range(10)]
        
        assert values1 == values2, "Python random should be reproducible"
    
    def test_numpy_reproducibility(self):
        """Test that NumPy random is reproducible after setup_seed."""
        setup_seed(42)
        values1 = np.random.rand(10).tolist()
        
        setup_seed(42)
        values2 = np.random.rand(10).tolist()
        
        assert values1 == values2, "NumPy random should be reproducible"
    
    def test_torch_cpu_reproducibility(self):
        """Test that PyTorch CPU random is reproducible after setup_seed."""
        setup_seed(42)
        values1 = torch.rand(10).tolist()
        
        setup_seed(42)
        values2 = torch.rand(10).tolist()
        
        assert values1 == values2, "PyTorch CPU random should be reproducible"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_torch_cuda_reproducibility(self):
        """Test that PyTorch CUDA random is reproducible after setup_seed."""
        setup_seed(42)
        values1 = torch.rand(10, device='cuda').cpu().tolist()
        
        setup_seed(42)
        values2 = torch.rand(10, device='cuda').cpu().tolist()
        
        assert values1 == values2, "PyTorch CUDA random should be reproducible"
    
    def test_different_seeds_produce_different_values(self):
        """Test that different seeds produce different random values."""
        setup_seed(42)
        values1 = [random.random() for _ in range(10)]
        
        setup_seed(123)
        values2 = [random.random() for _ in range(10)]
        
        assert values1 != values2, "Different seeds should produce different values"
    
    def test_deterministic_mode(self):
        """Test that deterministic mode is set correctly."""
        setup_seed(42, deterministic=True)
        assert torch.backends.cudnn.deterministic == True
        assert torch.backends.cudnn.benchmark == False
        
        setup_seed(42, deterministic=False)
        assert torch.backends.cudnn.deterministic == False
        assert torch.backends.cudnn.benchmark == True


class TestRandomState:
    """Tests for get_random_state and set_random_state functions."""
    
    def test_save_and_restore_python_state(self):
        """Test saving and restoring Python random state."""
        setup_seed(42)
        random.random()  # Advance state
        
        state = get_random_state()
        value1 = random.random()
        
        set_random_state(state)
        value2 = random.random()
        
        assert value1 == value2, "Python state should be restored correctly"
    
    def test_save_and_restore_numpy_state(self):
        """Test saving and restoring NumPy random state."""
        setup_seed(42)
        np.random.rand()  # Advance state
        
        state = get_random_state()
        value1 = np.random.rand()
        
        set_random_state(state)
        value2 = np.random.rand()
        
        assert value1 == value2, "NumPy state should be restored correctly"
    
    def test_save_and_restore_torch_state(self):
        """Test saving and restoring PyTorch random state."""
        setup_seed(42)
        torch.rand(1)  # Advance state
        
        state = get_random_state()
        value1 = torch.rand(1).item()
        
        set_random_state(state)
        value2 = torch.rand(1).item()
        
        assert value1 == value2, "PyTorch state should be restored correctly"
    
    def test_state_dict_contains_expected_keys(self):
        """Test that state dict contains expected keys."""
        setup_seed(42)
        state = get_random_state()
        
        assert 'python' in state
        assert 'numpy' in state
        assert 'torch_cpu' in state
        
        if torch.cuda.is_available():
            assert 'torch_cuda' in state


class TestWorkerInitFn:
    """Tests for worker_init_fn function."""
    
    def test_different_workers_get_different_seeds(self):
        """Test that different workers get different seeds."""
        # Simulate two workers
        worker_init_fn(0, base_seed=42)
        values_worker0 = [random.random() for _ in range(5)]
        
        worker_init_fn(1, base_seed=42)
        values_worker1 = [random.random() for _ in range(5)]
        
        assert values_worker0 != values_worker1, "Different workers should have different random sequences"
    
    def test_same_worker_same_seed_reproducible(self):
        """Test that same worker with same seed is reproducible."""
        worker_init_fn(0, base_seed=42)
        values1 = [random.random() for _ in range(5)]
        
        worker_init_fn(0, base_seed=42)
        values2 = [random.random() for _ in range(5)]
        
        assert values1 == values2, "Same worker same seed should be reproducible"
    
    def test_worker_init_without_base_seed(self):
        """Test worker_init_fn without base_seed (uses NumPy state)."""
        setup_seed(42)
        worker_init_fn(0)  # Should not raise
        
        # Just verify it runs without error
        random.random()
        np.random.rand()
        torch.rand(1)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
