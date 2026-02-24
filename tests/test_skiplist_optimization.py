#!/usr/bin/env python3
"""
Benchmark test for skiplist_mask optimization in ColBERT.

This test demonstrates the performance improvement achieved by replacing
the O(n×m) skiplist masking algorithm with an O(n) lookup table approach.

The optimization is particularly important for ColBERT models which process
large batches of tokenized text and need to mask punctuation/special tokens
efficiently.
"""

import time
import torch
import numpy as np
from typing import Callable, Dict, List, Tuple
import matplotlib.pyplot as plt
from pylate.models.colbert import ColBERT


def measure_performance(
    fn: Callable,
    input_ids: torch.Tensor,
    skiplist: List[int],
    runs: int = 100,
    warmup: int = 10
) -> Dict[str, float]:
    """Measure performance of a skiplist_mask implementation.
    
    Args:
        fn: The skiplist_mask function to benchmark
        input_ids: Input tensor of token IDs
        skiplist: List of token IDs to skip
        runs: Number of benchmark runs
        warmup: Number of warmup runs
        
    Returns:
        Dictionary with performance metrics
    """
    device = input_ids.device
    
    # Warmup runs to ensure JIT compilation and cache warming
    for _ in range(warmup):
        _ = fn(input_ids, skiplist)
    
    # Synchronize GPU before timing
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    # Time the runs
    start = time.perf_counter()
    for _ in range(runs):
        mask = fn(input_ids, skiplist)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    elapsed = time.perf_counter() - start
    
    return {
        "total_time": elapsed,
        "time_per_call": elapsed / runs,
        "calls_per_sec": runs / elapsed,
        "ms_per_call": (elapsed / runs) * 1000
    }


def create_original_skiplist_mask():
    """Create the original O(n×m) implementation for comparison."""
    @staticmethod
    def skiplist_mask_original(input_ids: torch.Tensor, skiplist: list[int]) -> torch.Tensor:
        """Original skiplist_mask implementation with O(n×m) complexity."""
        skiplist_tensor = torch.tensor(
            data=skiplist, dtype=torch.long, device=input_ids.device
        )
        
        # Create a tensor of ones with the same shape as input_ids
        mask = torch.ones_like(input=input_ids, dtype=torch.bool)
        
        # Update the mask for each token in the skiplist
        for token_id in skiplist:
            mask = torch.where(
                condition=input_ids == token_id,
                input=torch.tensor(data=0, dtype=torch.bool, device=input_ids.device),
                other=mask,
            )
        
        return mask
    
    return skiplist_mask_original


def run_comprehensive_benchmark():
    """Run comprehensive benchmarks comparing original vs optimized implementations."""
    print("=" * 80)
    print("ColBERT skiplist_mask Optimization Benchmark")
    print("=" * 80)
    
    # Test parameters
    vocab_size = 30522  # BERT vocabulary size
    sequence_lengths = [128, 256, 512]  # Typical sequence lengths
    skiplist_sizes = [10, 50, 100, 200]  # Common punctuation/special token counts
    batch_sizes = [1, 8, 32]  # Typical batch sizes
    
    # Check device availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # Get implementations
    original_fn = create_original_skiplist_mask()
    optimized_fn = ColBERT.skiplist_mask
    
    # Store results for visualization
    results = []
    
    # Run benchmarks
    for batch_size in batch_sizes:
        for seq_len in sequence_lengths:
            for skip_size in skiplist_sizes:
                # Generate test data
                input_ids = torch.randint(
                    0, vocab_size, 
                    (batch_size, seq_len), 
                    dtype=torch.long, 
                    device=device
                )
                skiplist = list(range(skip_size))  # Common pattern: skip first N tokens
                
                # Measure original implementation
                orig_perf = measure_performance(original_fn, input_ids, skiplist)
                
                # Clear cache for fair comparison
                if hasattr(ColBERT, "_skiplist_lut_cache"):
                    ColBERT._skiplist_lut_cache.clear()
                
                # Measure optimized implementation
                opt_perf = measure_performance(optimized_fn, input_ids, skiplist)
                
                # Calculate speedup
                speedup = orig_perf["ms_per_call"] / opt_perf["ms_per_call"]
                
                # Store results
                result = {
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "skip_size": skip_size,
                    "orig_ms": orig_perf["ms_per_call"],
                    "opt_ms": opt_perf["ms_per_call"],
                    "speedup": speedup
                }
                results.append(result)
                
                # Print results
                print(f"Batch={batch_size:<3} SeqLen={seq_len:<4} SkipList={skip_size:<4} | "
                      f"Original: {orig_perf['ms_per_call']:>8.3f}ms | "
                      f"Optimized: {opt_perf['ms_per_call']:>8.3f}ms | "
                      f"Speedup: {speedup:>6.1f}x")
    
    # Calculate average speedup
    avg_speedup = np.mean([r["speedup"] for r in results])
    print(f"\nAverage Speedup: {avg_speedup:.1f}x")
    
    # Test cache effectiveness
    print("\n" + "=" * 80)
    print("Cache Effectiveness Test")
    print("=" * 80)
    
    test_ids = torch.randint(0, vocab_size, (32, 256), dtype=torch.long, device=device)
    test_skiplist = list(range(50))
    
    # First call (cache miss)
    start = time.perf_counter()
    _ = optimized_fn(test_ids, test_skiplist)
    if device.type == "cuda":
        torch.cuda.synchronize()
    first_call = (time.perf_counter() - start) * 1000
    
    # Second call (cache hit)
    start = time.perf_counter()
    _ = optimized_fn(test_ids, test_skiplist)
    if device.type == "cuda":
        torch.cuda.synchronize()
    second_call = (time.perf_counter() - start) * 1000
    
    cache_speedup = first_call / second_call
    print(f"First call (cache miss):  {first_call:.3f}ms")
    print(f"Second call (cache hit):  {second_call:.3f}ms")
    print(f"Cache speedup:            {cache_speedup:.1f}x")
    
    # Verify correctness
    print("\n" + "=" * 80)
    print("Correctness Verification")
    print("=" * 80)
    
    # Test various scenarios
    test_cases = [
        ("Empty skiplist", torch.tensor([1, 2, 3, 4, 5]), []),
        ("Single skip", torch.tensor([1, 2, 3, 4, 5]), [3]),
        ("Multiple skips", torch.tensor([1, 2, 3, 4, 5]), [2, 4]),
        ("All tokens", torch.tensor([1, 2, 3, 4, 5]), [1, 2, 3, 4, 5]),
    ]
    
    all_correct = True
    for name, test_input, test_skip in test_cases:
        orig_mask = original_fn(test_input, test_skip)
        opt_mask = optimized_fn(test_input, test_skip)
        correct = torch.equal(orig_mask, opt_mask)
        all_correct &= correct
        
        print(f"{name:<20} | Input: {test_input.tolist():<20} | "
              f"Skip: {test_skip:<15} | Match: {'✓' if correct else '✗'}")
    
    print(f"\nAll tests passed: {'✓' if all_correct else '✗'}")
    
    # Memory usage analysis
    print("\n" + "=" * 80)
    print("Memory Usage Analysis")
    print("=" * 80)
    
    if hasattr(ColBERT, "_skiplist_lut_cache"):
        cache = ColBERT._skiplist_lut_cache
        total_memory = 0
        
        for (device_id, skiplist), lut in cache.items():
            memory_bytes = lut.element_size() * lut.nelement()
            total_memory += memory_bytes
            print(f"Cache entry - Device: {device_id}, "
                  f"Skiplist size: {len(skiplist)}, "
                  f"LUT size: {lut.shape[0]}, "
                  f"Memory: {memory_bytes / 1024:.2f} KB")
        
        print(f"\nTotal cache memory: {total_memory / 1024:.2f} KB")
    
    return results


def plot_results(results: List[Dict]):
    """Create visualization of benchmark results."""
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(results)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('ColBERT skiplist_mask Optimization Performance', fontsize=16)
        
        # Plot 1: Speedup by sequence length
        ax1 = axes[0, 0]
        for batch_size in df['batch_size'].unique():
            data = df[df['batch_size'] == batch_size]
            grouped = data.groupby('seq_len')['speedup'].mean()
            ax1.plot(grouped.index, grouped.values, marker='o', label=f'Batch {batch_size}')
        ax1.set_xlabel('Sequence Length')
        ax1.set_ylabel('Speedup Factor')
        ax1.set_title('Speedup by Sequence Length')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Speedup by skiplist size
        ax2 = axes[0, 1]
        for seq_len in df['seq_len'].unique():
            data = df[df['seq_len'] == seq_len]
            grouped = data.groupby('skip_size')['speedup'].mean()
            ax2.plot(grouped.index, grouped.values, marker='s', label=f'SeqLen {seq_len}')
        ax2.set_xlabel('Skiplist Size')
        ax2.set_ylabel('Speedup Factor')
        ax2.set_title('Speedup by Skiplist Size')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Original vs Optimized time
        ax3 = axes[1, 0]
        ax3.scatter(df['orig_ms'], df['opt_ms'], alpha=0.6)
        max_time = max(df['orig_ms'].max(), df['opt_ms'].max())
        ax3.plot([0, max_time], [0, max_time], 'r--', alpha=0.5, label='Equal performance')
        ax3.set_xlabel('Original Implementation (ms)')
        ax3.set_ylabel('Optimized Implementation (ms)')
        ax3.set_title('Performance Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        
        # Plot 4: Speedup distribution
        ax4 = axes[1, 1]
        ax4.hist(df['speedup'], bins=20, edgecolor='black', alpha=0.7)
        ax4.axvline(df['speedup'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {df["speedup"].mean():.1f}x')
        ax4.set_xlabel('Speedup Factor')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Speedup Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('skiplist_optimization_results.png', dpi=150, bbox_inches='tight')
        print("\nVisualization saved to 'skiplist_optimization_results.png'")
        
    except ImportError:
        print("\nNote: Install matplotlib and pandas for visualization of results")


if __name__ == "__main__":
    results = run_comprehensive_benchmark()
    plot_results(results)
