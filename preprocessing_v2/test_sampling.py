import sys
from pathlib import Path
import numpy as np

# Import the production sampler from preprocessing_v2
sys.path.insert(0, str(Path(__file__).parent))
from step03_segment import uniform_sample

def test_sampling():
    T = 32
    
    cases = [1, 6, 16, 31, 32, 33, 45]
    for n in cases:
        coords = np.zeros((n, 1, 1), dtype=np.float32)
        coords[:, 0, 0] = np.arange(n)
        sampled = uniform_sample(coords, T)
        
        assert sampled.shape == (T, 1, 1), f"n={n}: Expected shape ({T}, 1, 1), got {sampled.shape}"
        sampled_indices = sampled[:, 0, 0].astype(int).tolist()
        unique_sampled = set(sampled_indices)
        
        if n < T:
            # All original frames should be preserved
            expected = set(range(n))
            assert unique_sampled == expected, f"n={n}: Missing original frames: {expected - unique_sampled}"
        else:
            # Subsampled should just have unique elements equal to T (if n >= T)
            assert len(unique_sampled) == T, f"n={n}: Did not subsample to T unique frames"
            
        print(f"Passed case n={n}. Unique frames preserved: {len(unique_sampled)}")

    # Empty input behavior
    coords = np.zeros((0, 1, 1), dtype=np.float32)
    try:
        uniform_sample(coords, T)
        assert False, "n=0: Failed to raise ValueError for empty clip"
    except ValueError as e:
        print(f"Passed case n=0. Raised expected ValueError: {e}")

if __name__ == "__main__":
    test_sampling()
    print("All tests passed!")
