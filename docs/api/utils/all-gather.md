# all_gather

Gathers a tensor from each distributed rank into a list. The tensor for the local rank is the original one, with the gradients while the others have no gradients.

- If torch.distributed is available and initialized:   1. Creates a list of tensors (each sized like the input `tensor`).   2. Gathers tensors from each rank into that list.   3. Replaces the local tensor in the list with the original tensor that retains gradients. 

- If torch.distributed is either unavailable, uninitialized, or   `world_size == 1`, it returns a list containing only the   original tensor and throws a warning to notify the user (helpful when using a single GPU setup).

## Parameters

- **tensor** (*'torch.Tensor'*)




