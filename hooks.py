import torch
import torch.nn as nn

# set up the initial hooks for all the corresponding layers
from models import GPT, Block


def delete_importance_attr(layer: nn.Module):
    if hasattr(layer, "calculated_importance"):
        del layer.calculated_importance


def remove_all_forward_hooks(model: GPT):
    if not isinstance(model, GPT):
        raise NotImplementedError("Only GPT models are supported for now")

    for module in model.modules():
        if isinstance(module, Block):
            for head in module.sa.heads:
                head._forward_hooks.clear()

                head.key._forward_hooks.clear()
                head.value._forward_hooks.clear()
                head.query._forward_hooks.clear()

                delete_importance_attr(head)

                delete_importance_attr(head.key)
                delete_importance_attr(head.query)
                delete_importance_attr(head.value)

            module.ffwd.net[0]._forward_hooks.clear()
            module.ln1._forward_hooks.clear()
            module.sa._forward_hooks.clear()
            module.sa.proj._forward_hooks.clear()
            delete_importance_attr(module.ffwd.net[0])
            delete_importance_attr(module.ln1)
            delete_importance_attr(module.sa)
            delete_importance_attr(module.sa.proj)


def register_all_forward_hooks(model: GPT):
    if not isinstance(model, GPT):
        raise NotImplementedError("Only GPT models are supported for now")

    num_blocks = 0
    for module in model.modules():
        if isinstance(module, Block):
            num_blocks += 1
            for head in module.sa.heads:
                head.register_forward_hook(attn_head_importance_hook)

                head.key.register_forward_hook(neuron_importance_hook)
                head.value.register_forward_hook(neuron_importance_hook)
                head.query.register_forward_hook(neuron_importance_hook)

            module.ffwd.net[0].register_forward_hook(
                neuron_importance_hook
            )  # register the forward hook to the linear layer inside of the ffwd block
            module.sa.proj.register_forward_hook(neuron_importance_hook)
            module.ln1.register_forward_hook(embedding_importance_hook)
            module.register_forward_hook(block_importance_hook)


def attn_head_importance_hook(
    module, ins, outs
) -> None:  # TODO: does the importance calculation returns the correct values for each head?
    """calculates the multi-head-attention layer's importance per head"""
    # outs.shape = (B, T, E) where B: batch_size, T: num tokens, E: embedding size
    # the importance is calculated as summing the L2 norm of the attn outputs on B and T dimensions
    outs_flat = outs.view(-1, outs.shape[-1])  # (b,t,e) -> (b*t, e)
    importance = torch.linalg.vector_norm(outs_flat.detach().cpu(), ord=2, dim=-1).sum()

    module.calculated_importance = importance

    # print(outs_flat.shape)
    # print("module:", module.__class__.__name__, end=" ")
    # print("importance:", importance)
    # print(f"{module.__class__.__name__} importance: {importance.shape}")

def neuron_importance_hook(module, ins, outs) -> None:
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


def embedding_importance_hook(module, ins, outs) -> None:
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


def block_importance_hook(module, ins, outs) -> None:

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
