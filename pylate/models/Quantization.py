from __future__ import annotations

import torch
from sentence_transformers.models.Module import Module


class Quantization(Module):
    """
    Advanced Quantizer module supporting trinary, 4-bit, and 8-bit quantization.
    Uses rolling statistics for adaptive quantization bounds.
    Works with dictionary input containing token embeddings.
    """

    def __init__(self, num_bits: int = 8, momentum: float = 0.1):
        super().__init__()
        self.num_bits = num_bits
        self.momentum = momentum

        # Register buffers for rolling statistics (not learnable parameters)
        self.register_buffer("rolling_mean", torch.tensor(0.0))
        self.register_buffer("rolling_std", torch.tensor(1.0))
        self.register_buffer("initialized", torch.tensor(False))

        # Set quantization ranges based on bit width
        if num_bits == 2:  # Trinary is special case
            self.qmin = -1
            self.qmax = 1
        else:
            # For signed integers: range is [-2^(n-1), 2^(n-1) - 1]
            self.qmin = -(2 ** (num_bits - 1))
            self.qmax = 2 ** (num_bits - 1) - 1

    def __repr__(self) -> str:
        return f"Quantization({self.get_config_dict()})"

    def update_rolling_stats(self, x: torch.Tensor):
        """Update rolling mean and std statistics."""
        with torch.no_grad():
            # Calculate batch statistics
            batch_mean = x.mean()
            batch_std = x.std()

            if not self.initialized:
                self.rolling_mean.copy_(batch_mean)
                self.rolling_std.copy_(batch_std)
                self.initialized.copy_(torch.tensor(True))
            else:
                # Update rolling averages
                self.rolling_mean = (
                    1 - self.momentum
                ) * self.rolling_mean + self.momentum * batch_mean
                self.rolling_std = (
                    1 - self.momentum
                ) * self.rolling_std + self.momentum * batch_std

    def get_bounds(self):
        """Calculate min/max bounds based on rolling statistics."""
        # Using 3-sigma rule for bounds (covers ~99.7% of values)
        # min_val = self.rolling_mean - 3 * self.rolling_std
        # max_val = self.rolling_mean + 3 * self.rolling_std
        min_val = self.rolling_mean - self.rolling_std
        max_val = self.rolling_mean + self.rolling_std
        return min_val, max_val

    def quantize_trinary(
        self, x: torch.Tensor, min_val: torch.Tensor, max_val: torch.Tensor
    ) -> torch.Tensor:
        """Trinary quantization: -1, 0, 1"""
        quantized = torch.zeros_like(x)
        quantized[x >= max_val] = 1
        quantized[x <= min_val] = -1
        # Values between min and max remain 0
        return quantized

    def quantize_int(
        self, x: torch.Tensor, min_val: torch.Tensor, max_val: torch.Tensor
    ) -> torch.Tensor:
        """Integer quantization for 4-bit or 8-bit"""
        with torch.no_grad():
            quantized = torch.zeros_like(x)

            # Handle values outside bounds
            quantized[x >= max_val] = self.qmax
            quantized[x <= min_val] = self.qmin

            # Handle values within bounds
            in_bounds_mask = (x > min_val) & (x < max_val)
            if in_bounds_mask.any():
                # Scale to quantization range
                scale_factor = (self.qmax - self.qmin) / (max_val - min_val)
                scaled = scale_factor * (x[in_bounds_mask] - min_val) + self.qmin
                quantized[in_bounds_mask] = torch.round(scaled)

            return quantized

    def dequantize_trinary(
        self, q: torch.Tensor, min_val: torch.Tensor, max_val: torch.Tensor
    ) -> torch.Tensor:
        """Dequantize trinary values back to continuous space."""
        # Map -1, 0, 1 back to min, mid, max
        mid_val = (min_val + max_val) / 2
        dequantized = torch.zeros_like(q)
        dequantized[q == -1] = min_val
        dequantized[q == 0] = mid_val
        dequantized[q == 1] = max_val
        return dequantized

    def dequantize_int(
        self, q: torch.Tensor, min_val: torch.Tensor, max_val: torch.Tensor
    ) -> torch.Tensor:
        """Dequantize integer values back to continuous space."""
        # Inverse of quantization formula
        scale_factor = (max_val - min_val) / (self.qmax - self.qmin)
        dequantized = scale_factor * (q - self.qmin) + min_val
        return dequantized

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass that quantizes token embeddings in the input dictionary.

        Args:
            features: Dictionary containing at least "token_embeddings" key with tensor value
                     Shape of token_embeddings: (batch_size, sequence_length, embedding_dim)

        Returns:
            Dictionary with quantized token_embeddings
        """
        # Extract token embeddings
        token_embeddings = features["token_embeddings"]

        # Update rolling statistics during training
        if self.training:
            self.update_rolling_stats(token_embeddings)

        # Get current bounds
        min_val, max_val = self.get_bounds()

        # Quantize based on bit width
        if self.num_bits == 2:  # Trinary
            quantized = self.quantize_trinary(token_embeddings, min_val, max_val)
            dequantized = self.dequantize_trinary(quantized, min_val, max_val)
        else:  # 4-bit or 8-bit
            quantized = self.quantize_int(token_embeddings, min_val, max_val)
            dequantized = self.dequantize_int(quantized, min_val, max_val)

        # print(quantized.dtype, dequantized.dtype)
        # Straight-through estimator: use dequantized in forward, but original for gradient
        quantized_embeddings = (
            token_embeddings + (dequantized - token_embeddings).detach()
        )
        del token_embeddings, dequantized, quantized  # Free memory
        # print(quantized_embeddings.dtype)

        # Create output dictionary with updated token embeddings
        # output_features = features.copy()

        # output_features["token_embeddings"] = quantized_embeddings
        features["token_embeddings"] = quantized_embeddings

        return features

    def extra_repr(self) -> str:
        return f"num_bits={self.num_bits}, momentum={self.momentum}"

    def save(
        self, output_path: str, *args, safe_serialization: bool = True, **kwargs
    ) -> None:
        self.save_config(output_path)
