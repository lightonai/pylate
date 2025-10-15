# all_gather_with_gradients

Gathers a tensor from each distributed rank into a list. All the tensors will retain gradients. This is the same as `all_gather`, but all the tensors will retain gradients and is used to compute contrastive with local queries only to lower the memory usage, see https://github.com/mlfoundations/open_clip/issues/616

- If torch.distributed is available and initialized, gather all the tensors (with gradients) from each rank into a list 

- If torch.distributed is either unavailable, uninitialized, or   `world_size == 1`, it returns a list containing only the   original tensor and throws a warning to notify the user (helpful when using a single GPU setup).

## Parameters

- **tensor** (*'torch.Tensor'*)




