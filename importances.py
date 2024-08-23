import torch


def attn_head_importance(outs) -> torch.Tensor:
    """calculates the multi-head-attention layer's importance per head"""
    # outs.shape = (B, T, E) where B: batch_size, T: num tokens, E: embedding size
    # the importance is calculated as summing the L2 norm of the attn outputs on B and T dimensions
    outs_flat = outs.view(-1, outs.shape[-1])  # (b,t,e) -> (b*t, e)
    importance = torch.linalg.vector_norm(outs_flat.detach().cpu(), ord=2, dim=-1).sum()

    # print(outs_flat.shape)
    # print("module:", module.__class__.__name__, end=" ")
    # print("importance:", importance)
    # print(f"{module.__class__.__name__} importance: {importance.shape}")
    return importance


def neuron_importance(module, outs) -> None:
    """calculates the neuron importance for the given layer"""

    # the ffwd linear weights should be in the shape of (out, in)
    # the paper sums up the values of (X * W_i^T) meaning (B, T, in) x (in, 1)= (B,T,1) -> (1, ) (summed up)

    # thus, in order to vectorize this operation, we'll need to hook this function to the first linear layer itself rather than the whole ffwd block.

    # for each neuron in the ffwd layer, we can simply sum up the output columns

    # as they're the activations of individual neurons
    # calculate the importances
    # importance = outs.detach().sum()
    importance = outs.detach().cpu().sum(dim=(0, 1))
    # print(f"{module.__class__.__name__} importance.shape: {importance.shape}")

    module.calculated_importance = importance


def embedding_importance(module, outs) -> None:
    # the first block's first processing layer will be the
    # layer norm
    # so we'll just sum up the layer norm outputs after getting them
    # calculate the importances

    importance = outs.detach().sum(dim=(0, 1))
    # print("importance.shape:", importance.shape)
    # print("n_embd: ", outs.size(-1))
    # print("module:", module.__class__.__name__)
    # print("outs.shape:", outs.shape) # probably (B, T, E)

    module.calculated_importance = importance

    # print(f"{module.__class__.__name__} importance.shape: {importance.shape}")


def block_importance(module, ins, outs) -> None:

    in_vectors = ins[0].detach()  # (B, T, E)
    out_vectors = outs.detach()  # (B, T, E)

    # Calculate cosine similarity for each sample and time step
    dot_product = torch.sum(in_vectors * out_vectors, dim=-1)  # (B, T)
    in_norm = torch.norm(in_vectors, p=2, dim=-1)  # (B, T)
    out_norm = torch.norm(out_vectors, p=2, dim=-1)  # (B, T)

    cosine_sim = dot_product / (in_norm * out_norm + 1e-8)  # (B, T)

    # Calculate BI by taking the expectation (mean) and subtracting from 1
    block_importance = 1 - torch.mean(cosine_sim)

    # print("Block Importance:", block_importance.item())
    # print("module:", module.__class__.__name__)
    # print("outs.shape:", outs.shape)  # (B, T, E)

    module.calculated_importance = block_importance

    # print(f"{module.__class__.__name__} importance.shape: {block_importance.shape}")
