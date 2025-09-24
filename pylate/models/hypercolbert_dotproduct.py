from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from .colbert import ColBERT

logger = logging.getLogger(__name__)


class TokenWiseQNet(nn.Module):
    """
    A simple MLP that processes tokens independently.
    This replaces the complex NoTorch* classes with standard PyTorch modules.
    """
    def __init__(
        self,
        vector_dimensions: List[int],
        activation_type: str = "relu",
        do_dropout: bool = False,
        dropout_prob: float = 0.1,
        do_layer_norm: bool = True,
        do_residual: bool = True,
        do_residual_on_last: bool = False,
        layer_norm_before_residual: bool = True,
    ):
        super().__init__()
        
        self.do_residual = do_residual
        self.do_residual_on_last = do_residual_on_last
        self.layer_norm_before_residual = layer_norm_before_residual
        
        # Build layers
        layers = []
        for i in range(len(vector_dimensions) - 1):
            in_dim = vector_dimensions[i]
            out_dim = vector_dimensions[i + 1]
            
            # Add linear layer
            layers.append(nn.Linear(in_dim, out_dim))
            
            # Add activation for all but last layer
            if i < len(vector_dimensions) - 2:
                if activation_type == "relu":
                    layers.append(nn.ReLU())
                elif activation_type == "gelu":
                    layers.append(nn.GELU())
                elif activation_type == "tanh":
                    layers.append(nn.Tanh())
                
                # Add dropout if needed
                if do_dropout:
                    layers.append(nn.Dropout(dropout_prob))
                
                # Add layer norm if needed
                if do_layer_norm:
                    layers.append(nn.LayerNorm(out_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Store dimensions for residual connections
        self.vector_dimensions = vector_dimensions
    
    def forward(self, features):
        # Pass through - not used in main pipeline
        return features


class HyperEncoderModule(nn.Module):
    """
    Simplified HyperEncoder that generates q-net parameters directly.
    Uses HyperNetworks approach: generate MLP weights from query embeddings.
    """
    
    def __init__(self, converter_kwargs: dict, base_encoder_output_dim: int):
        super().__init__()
        
        self.base_encoder_output_dim = base_encoder_output_dim
        # Change to simpler architecture: just 128 -> 1
        self.vector_dimensions = [128, 1]  # Simplified to single layer
        
        # Store converter kwargs for compatibility
        self.converter_kwargs = converter_kwargs
        
        # Calculate total number of parameters needed for q-net
        self.weight_shapes = []
        self.bias_shapes = []
        total_params = 0
        
        for i in range(len(self.vector_dimensions) - 1):
            in_dim = self.vector_dimensions[i]
            out_dim = self.vector_dimensions[i + 1]
            
            # Weight shape
            self.weight_shapes.append((in_dim, out_dim))
            total_params += in_dim * out_dim
            
            # Bias shape
            self.bias_shapes.append((out_dim,))
            total_params += out_dim
        
        # Single projection to generate all q-net parameters at once
        self.param_generator = nn.Linear(base_encoder_output_dim, total_params)
        
        # SPECIAL INITIALIZATION for dot product behavior
        # For a single layer [128, 1], we have 128 weights + 1 bias = 129 params
        with torch.no_grad():
            # Initialize to generate identity-like weights
            # We want the param_generator to output the query embedding itself as weights
            # and zero bias initially
            
            # Since param_generator maps from hidden_dim to 129 params,
            # we want the first 128 outputs to be the identity of the input
            # and the last output (bias) to be zero
            
            # Initialize weight matrix
            if total_params == 129 and base_encoder_output_dim == 128:
                # Set up identity mapping for the weight portion
                self.param_generator.weight[:128, :] = torch.eye(128)
                # Set bias generation to zero
                self.param_generator.weight[128, :] = 0.0
                
                # Initialize biases
                self.param_generator.bias[:128] = 0.0  # No offset for weights
                self.param_generator.bias[128] = 0.0    # Zero bias for q-net
            else:
                # Fallback to small random initialization
                nn.init.normal_(self.param_generator.weight, std=0.01)
                nn.init.zeros_(self.param_generator.bias)
    def forward(self, features):
        # Pass through - not used in main pipeline
        return features
    
    def generate_qnet_params(self, query_embedding):
        """
        Generate q-net parameters from a query embedding.
        
        Parameters
        ----------
        query_embedding : torch.Tensor
            Shape (hidden_dim,) - single token embedding
            
        Returns
        -------
        dict
            Dictionary of parameters for the q-net
        """
        # Generate all parameters at once
        params = self.param_generator(query_embedding)
        
        # Split into weights and biases
        param_dict = {}
        offset = 0
        
        for i, ((in_dim, out_dim), bias_dim) in enumerate(zip(self.weight_shapes, self.bias_shapes)):
            # Extract weight
            weight_size = in_dim * out_dim
            weight = params[offset:offset + weight_size].view(out_dim, in_dim)
            param_dict[f'mlp.{i*2}.weight'] = weight  # i*2 because of activation layers
            offset += weight_size
            
            # Extract bias
            bias_size = bias_dim[0]
            bias = params[offset:offset + bias_size]
            param_dict[f'mlp.{i*2}.bias'] = bias
            offset += bias_size
        
        return param_dict
    
    def forward(self, features):
        # Pass through - not used in main pipeline
        return features


class SimplifiedQNet(nn.Module):
    """
    A lightweight q-net that can have its parameters dynamically set.
    Much simpler than the NoTorch* classes.
    """
    def __init__(self, vector_dimensions, **kwargs):
        super().__init__()
        
        # Build a simple MLP
        layers = []
        for i in range(len(vector_dimensions) - 1):
            layers.append(nn.Linear(vector_dimensions[i], vector_dimensions[i + 1]))
            if i < len(vector_dimensions) - 2:  # No activation on last layer
                layers.append(nn.ReLU())
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, features):
        # Pass through - not used in main pipeline
        return features
    


class HyperColBERT(ColBERT):
    """
    Simplified HyperColBERT with cleaner gradient flow.
    
    Main simplifications:
    1. Use standard PyTorch modules instead of custom NoTorch* classes
    2. Generate q-net parameters more directly
    3. Simplified similarity computation
    """
    
    def __init__(
        self,
        converter_kwargs: dict = None,
        base_encoder_output_dim: int = 128,
        freeze_transformer: bool = False,
        debug_use_cosine: bool = False,  # DEBUG FLAG
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Default converter kwargs
        converter_kwargs = converter_kwargs or {"vector_dimensions": [128, 1]}
        self.vector_dimensions = [128, 1] #converter_kwargs.get("vector_dimensions", [128, 1])
        
        # Create hyperencoder
        self.hyperencoder = HyperEncoderModule(converter_kwargs, base_encoder_output_dim)
        
        # Create a single shared q-net structure that we'll modulate
        self.shared_qnet = SimplifiedQNet(self.vector_dimensions)
        
        # For backward compatibility
        self.weight_to_model_converter = None  # Not needed in simplified version
        
        # DEBUG: Flag to use cosine similarity instead of q-net
        self.debug_use_cosine = debug_use_cosine
        
        # Freeze transformer if requested
        if freeze_transformer:
            for module in self.modules():
                if isinstance(module, (AutoModel, type(self[0]))):
                    for param in module.parameters():
                        param.requires_grad = False
                    break
    
    def compute_similarity_with_hyperencoder(
        self,
        query_embeddings: torch.Tensor,
        document_embeddings: torch.Tensor,
        query_masks: torch.Tensor
    ) -> torch.Tensor:
        """
        Simplified similarity computation using hyperencoder.
        
        DEBUG MODE: If debug_use_cosine is True, uses standard MaxSim cosine similarity
        instead of q-net based similarity.
        
        Instead of creating separate q-net objects, we:
        1. Generate q-net parameters for each query token
        2. Apply them using functional operations
        3. Compute similarities in a vectorized way
        """
        batch_size, seq_len, hidden_dim = query_embeddings.shape
        num_docs, doc_len, doc_dim = document_embeddings.shape
        
        # Initialize similarity scores
        similarity_scores = torch.zeros(batch_size, num_docs, device=query_embeddings.device)
        
        # DEBUG: Use standard cosine similarity (MaxSim)
        if False:
            logger.info("DEBUG: Using cosine similarity instead of q-net")
            
            # Normalize embeddings for cosine similarity
            query_norm = F.normalize(query_embeddings, p=2, dim=-1)  # (batch_size, seq_len, hidden_dim)
            doc_norm = F.normalize(document_embeddings, p=2, dim=-1)  # (num_docs, doc_len, doc_dim)
            
            for b in range(batch_size):
                for t in range(seq_len):
                    # Skip padding tokens
                    if query_masks[b, t] == 0:
                        continue
                    
                    # Get query token embedding
                    q_token = query_norm[b, t:t+1, :]  # (1, hidden_dim)
                    
                    # Compute cosine similarity with all document tokens
                    # q_token: (1, hidden_dim)
                    # doc_norm: (num_docs, doc_len, doc_dim)
                    cosine_sim = torch.matmul(doc_norm, q_token.transpose(0, 1))  # (num_docs, doc_len, 1)
                    cosine_sim = cosine_sim.squeeze(-1)  # (num_docs, doc_len)
                    
                    # Take max over document tokens for this query token (MaxSim)
                    max_scores, _ = cosine_sim.max(dim=1)  # (num_docs,)
                    
                    # Add to total similarity
                    similarity_scores[b] += max_scores
            
            # Log some debug info
            logger.info(f"DEBUG MaxSim - Min score: {similarity_scores.min():.4f}, "
                       f"Max score: {similarity_scores.max():.4f}, "
                       f"Mean score: {similarity_scores.mean():.4f}")
            
        else:
            # Original q-net based similarity computation
            # Process each query in the batch
            for b in range(batch_size):
                # Process each query token
                for t in range(seq_len):
                    # Skip padding tokens
                    if query_masks[b, t] == 0:
                        continue
                    
                    # Get embedding for this token
                    token_emb = query_embeddings[b, t]  # (hidden_dim,)
                    
                    # Generate q-net parameters for this token
                    qnet_params = self.hyperencoder.generate_qnet_params(token_emb)
                    
                    # Apply q-net to all document tokens at once
                    # Reshape documents for batch processing
                    docs_flat = document_embeddings.view(-1, doc_dim)  # (num_docs * doc_len, doc_dim)
                    
                    # Manually apply the q-net transformation using generated parameters
                    x = docs_flat
                    if True:
                        layer_idx = 0
                        for i in range(len(self.vector_dimensions) - 1):
                            # Apply linear transformation
                            weight = qnet_params[f'mlp.{layer_idx}.weight']
                            bias = qnet_params[f'mlp.{layer_idx}.bias']
                            x = F.linear(x, weight, bias)
                            
                            # Apply activation (except for last layer)
                            if i < len(self.vector_dimensions) - 2:
                                x = F.relu(x)
                                layer_idx += 2  # Account for activation layer in indexing
                            else:
                                layer_idx += 1
                    else:
                        x = torch.matmul(token_emb, x.transpose(0,1))
                    # Reshape back to (num_docs, doc_len, output_dim)
                    # Since output_dim = 1, we reshape and squeeze
                    doc_scores = x.view(num_docs, doc_len, -1).squeeze(-1)
                    
                    # Take max over document tokens for this query token
                    max_scores, _ = doc_scores.max(dim=1)  # (num_docs,)
                    
                    # Add to total similarity
                    similarity_scores[b] += max_scores
        
        return similarity_scores
    
    def generate_q_nets_per_token(
        self, 
        query_embeddings: torch.Tensor, 
        query_masks: torch.Tensor
    ) -> Tuple[List, int, int]:
        """
        For API compatibility - but now returns parameter dictionaries instead of NoTorchSequential.
        """
        batch_size, seq_len = query_masks.shape
        
        # Generate parameters for each token
        qnet_params_list = []
        for b in range(batch_size):
            for t in range(seq_len):
                if query_masks[b, t] > 0:
                    token_emb = query_embeddings[b, t]
                    params = self.hyperencoder.generate_qnet_params(token_emb)
                else:
                    params = None
                qnet_params_list.append(params)
        
        return qnet_params_list, batch_size, seq_len
    
    def compute_similarity_with_per_token_qnets(
        self, 
        q_nets: List,  # Now these are parameter dictionaries
        document_embeddings: torch.Tensor,
        query_masks: torch.Tensor,
        batch_size: int,
        seq_len: int
    ) -> torch.Tensor:
        """
        Simplified similarity computation using parameter dictionaries.
        """
        num_docs, doc_len, doc_dim = document_embeddings.shape
        similarity_scores = torch.zeros(batch_size, num_docs, device=document_embeddings.device)
        
        for b in range(batch_size):
            for t in range(seq_len):
                flat_idx = b * seq_len + t
                
                # Skip padding or missing parameters
                if query_masks[b, t] == 0 or q_nets[flat_idx] is None:
                    continue
                
                qnet_params = q_nets[flat_idx]
                
                # Apply q-net to all documents
                docs_flat = document_embeddings.view(-1, doc_dim)
                x = docs_flat
                
                # Apply layers with generated parameters
                layer_idx = 0
                for i in range(len(self.vector_dimensions) - 1):
                    weight = qnet_params[f'mlp.{layer_idx}.weight']
                    bias = qnet_params[f'mlp.{layer_idx}.bias']
                    x = F.linear(x, weight, bias)
                    
                    if i < len(self.vector_dimensions) - 2:
                        x = F.relu(x)
                        layer_idx += 2
                    else:
                        layer_idx += 1
                
                # Compute max similarity per document
                # Reshape: (num_docs * doc_len, output_dim) -> (num_docs, doc_len, output_dim)
                # Since output_dim = 1, we reshape and squeeze
                doc_scores = x.view(num_docs, doc_len, -1).squeeze(-1)  # (num_docs, doc_len)
                max_scores, _ = doc_scores.max(dim=1)
                # print("doc_scores:", doc_scores)
                similarity_scores[b] += max_scores
        
        return similarity_scores
    
    def forward(self, input):
        """Use parent ColBERT forward for standard embedding generation."""
        return super().forward(input)