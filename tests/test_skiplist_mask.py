#!/usr/bin/env python3
"""
Unit tests for skiplist_mask optimization.

These tests ensure that the optimized implementation maintains
backward compatibility and correctness.
"""

import pytest
import torch
from pylate.models.colbert import ColBERT


class TestSkiplistMask:
    """Test suite for skiplist_mask functionality."""
    
    def setup_method(self):
        """Clear cache before each test."""
        if hasattr(ColBERT, "_skiplist_lut_cache"):
            ColBERT._skiplist_lut_cache.clear()
    
    def test_empty_skiplist(self):
        """Test with empty skiplist - all tokens should be kept."""
        input_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
        skiplist = []
        
        mask = ColBERT.skiplist_mask(input_ids, skiplist)
        
        assert mask.shape == input_ids.shape
        assert mask.dtype == torch.bool
        assert mask.all()  # All tokens should be True (kept)
    
    def test_single_token_skip(self):
        """Test skipping a single token."""
        input_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
        skiplist = [3]
        
        mask = ColBERT.skiplist_mask(input_ids, skiplist)
        
        expected = torch.tensor([True, True, False, True, True])
        assert torch.equal(mask, expected)
    
    def test_multiple_tokens_skip(self):
        """Test skipping multiple tokens."""
        input_ids = torch.tensor([1, 2, 3, 4, 5, 2, 3], dtype=torch.long)
        skiplist = [2, 3]
        
        mask = ColBERT.skiplist_mask(input_ids, skiplist)
        
        expected = torch.tensor([True, False, False, True, True, False, False])
        assert torch.equal(mask, expected)
    
    def test_all_tokens_skip(self):
        """Test skipping all tokens."""
        input_ids = torch.tensor([1, 2, 3], dtype=torch.long)
        skiplist = [1, 2, 3]
        
        mask = ColBERT.skiplist_mask(input_ids, skiplist)
        
        expected = torch.tensor([False, False, False])
        assert torch.equal(mask, expected)
    
    def test_no_matches(self):
        """Test when no tokens match the skiplist."""
        input_ids = torch.tensor([1, 2, 3], dtype=torch.long)
        skiplist = [4, 5, 6]
        
        mask = ColBERT.skiplist_mask(input_ids, skiplist)
        
        expected = torch.tensor([True, True, True])
        assert torch.equal(mask, expected)
    
    def test_2d_input(self):
        """Test with 2D input (batch of sequences)."""
        input_ids = torch.tensor([
            [1, 2, 3, 4],
            [5, 3, 2, 1],
            [2, 2, 2, 2]
        ], dtype=torch.long)
        skiplist = [2]
        
        mask = ColBERT.skiplist_mask(input_ids, skiplist)
        
        expected = torch.tensor([
            [True, False, True, True],
            [True, True, False, True],
            [False, False, False, False]
        ])
        assert torch.equal(mask, expected)
    
    def test_large_skiplist(self):
        """Test with a large skiplist."""
        input_ids = torch.tensor([1, 50, 100, 150, 200], dtype=torch.long)
        skiplist = list(range(0, 1000, 2))  # Even numbers from 0 to 998
        
        mask = ColBERT.skiplist_mask(input_ids, skiplist)
        
        expected = torch.tensor([True, False, False, False, False])
        assert torch.equal(mask, expected)
    
    def test_duplicate_skiplist(self):
        """Test that duplicate values in skiplist are handled correctly."""
        input_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
        skiplist = [2, 2, 3, 3, 3]  # Duplicates
        
        mask = ColBERT.skiplist_mask(input_ids, skiplist)
        
        expected = torch.tensor([True, False, False, True, True])
        assert torch.equal(mask, expected)
    
    def test_cache_consistency(self):
        """Test that cached results are consistent."""
        input_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
        skiplist = [2, 4]
        
        # First call
        mask1 = ColBERT.skiplist_mask(input_ids, skiplist)
        
        # Second call (should use cache)
        mask2 = ColBERT.skiplist_mask(input_ids, skiplist)
        
        assert torch.equal(mask1, mask2)
    
    def test_different_devices(self):
        """Test behavior across different devices if available."""
        input_ids_cpu = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
        skiplist = [2, 4]
        
        mask_cpu = ColBERT.skiplist_mask(input_ids_cpu, skiplist)
        
        if torch.cuda.is_available():
            input_ids_gpu = input_ids_cpu.cuda()
            mask_gpu = ColBERT.skiplist_mask(input_ids_gpu, skiplist)
            
            # Results should be the same (after moving to same device)
            assert torch.equal(mask_cpu, mask_gpu.cpu())
    
    def test_edge_case_single_element(self):
        """Test with single element tensor."""
        input_ids = torch.tensor([5], dtype=torch.long)
        
        # Not in skiplist
        mask = ColBERT.skiplist_mask(input_ids, [1, 2, 3])
        assert mask.item() is True
        
        # In skiplist
        mask = ColBERT.skiplist_mask(input_ids, [5])
        assert mask.item() is False
    
    def test_high_token_ids(self):
        """Test with high token IDs (beyond typical vocab size)."""
        input_ids = torch.tensor([50000, 60000, 70000], dtype=torch.long)
        skiplist = [60000]
        
        mask = ColBERT.skiplist_mask(input_ids, skiplist)
        
        expected = torch.tensor([True, False, True])
        assert torch.equal(mask, expected)
    
    def test_skiplist_ordering_invariance(self):
        """Test that skiplist order doesn't affect results."""
        input_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
        
        # Different orderings of the same skiplist
        skiplist1 = [2, 4, 1]
        skiplist2 = [4, 1, 2]
        skiplist3 = [1, 4, 2]
        
        mask1 = ColBERT.skiplist_mask(input_ids, skiplist1)
        mask2 = ColBERT.skiplist_mask(input_ids, skiplist2)
        mask3 = ColBERT.skiplist_mask(input_ids, skiplist3)
        
        assert torch.equal(mask1, mask2)
        assert torch.equal(mask2, mask3)
    
    @pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
    def test_input_dtype_handling(self, dtype):
        """Test that only long dtype is accepted."""
        input_ids = torch.tensor([1, 2, 3], dtype=dtype)
        skiplist = [2]
        
        if dtype == torch.long:
            # Should work
            mask = ColBERT.skiplist_mask(input_ids, skiplist)
            assert mask.shape == input_ids.shape
        else:
            # Should raise assertion error
            with pytest.raises(AssertionError):
                ColBERT.skiplist_mask(input_ids, skiplist)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
