import torch.nn as nn

from models import Block


def prune_neurons(model, n: list[int] | float = 0.2) -> None:
    # goal: trim the MLP layer weights
    # 1 - argsort the importances of the `ffwd` layers defined in the model
    # 2 - remove the weights with respect to the given ratio

    constraints = None
    c = 0

    for module in model.modules():
        if isinstance(module, Block):
            importances = module.ffwd.net[0].calculated_importance

            if constraints is None:
                constraints = pruning_n_handler(n, importances.size(0), model.n_blocks)

            num_neurons = constraints[c]  # type: ignore
            c += 1

            idx = importances.argsort(descending=True)[:num_neurons]
            # reinitialize the weights along with the layer
            dense1 = module.ffwd.net[0]
            dense2 = module.ffwd.net[2]

            module.ffwd.net[0] = nn.Linear(dense1.in_features, num_neurons).to(
                model.device
            )  # weights.shape = (num_neurons, dense1.in_features)
            module.ffwd.net[2] = nn.Linear(num_neurons, dense2.out_features).to(
                model.device
            )  # weights.shape = (dense2.out_features = emb)
            # now we need to set the weights to the new layers.

            module.ffwd.net[0].weight.data = dense1.weight.data[idx, :]
            module.ffwd.net[0].bias.data = dense1.bias.data[idx]

            module.ffwd.net[2].weight.data = dense2.weight.data[:, idx]
            module.ffwd.net[2].bias.data = dense2.bias.data

            module.ffwd.net[0].calculated_importance = importances[idx]
            module.ffwd.net[2].calculated_importance = importances[idx]

    return model


def prune_heads(model, n: list[int] | float) -> None:
    # goal: trim the attention heads' layer weights using the same approach as the `prune_neurons`

    constraints = None
    c = 0

    for module in model.modules():
        if isinstance(module, Block):
            # now the multi-head attention
            for head in module.sa.heads:
                # key,value,query weight shape: (head_size, n_embd) # n_embd
                k, v, q = head.key, head.value, head.query

                key_importances = head.key.calculated_importance
                value_importances = head.value.calculated_importance
                query_importances = head.query.calculated_importance

                if constraints is None:
                    constraints = pruning_n_handler(
                        n, key_importances.size(0), model.n_blocks
                    )

                num_neurons = constraints[c]  # type: ignore

                k_idx = key_importances.argsort(descending=True)[:num_neurons]
                v_idx = value_importances.argsort(descending=True)[:num_neurons]
                q_idx = query_importances.argsort(descending=True)[:num_neurons]

                head.key = nn.Linear(k.in_features, num_neurons, bias=False).to(
                    model.device
                )
                head.value = nn.Linear(v.in_features, num_neurons, bias=False).to(
                    model.device
                )
                head.query = nn.Linear(q.in_features, num_neurons, bias=False).to(
                    model.device
                )

                head.key.weight.data = k.weight.data[
                    k_idx, :
                ]  # (head_size, num_dense_embd)
                head.value.weight.data = v.weight.data[
                    v_idx, :
                ]  # (head_size, num_dense_embd)
                head.query.weight.data = q.weight.data[
                    q_idx, :
                ]  # (head_size, num_dense_embd)

                head.key.calculated_importance = key_importances[k_idx]
                head.value.calculated_importance = value_importances[v_idx]
                head.query.calculated_importance = query_importances[q_idx]

                # TODO: only the weights in the embedding layers are prunned (1st strategy)
                # TODO: need to follow the correct implementation from the paper (pruning every linear layer?)
            proj = module.sa.proj
            proj_importances = module.sa.proj.calculated_importance

            num_neurons = constraints[c] * model.n_head  # type: ignore

            idx = proj_importances.argsort(descending=True)[:num_neurons]

            module.sa.proj = nn.Linear(num_neurons, proj.out_features).to(model.device)

            module.sa.proj.weight.data = proj.weight.data[:, idx]
            module.sa.proj.bias.data = proj.bias.data

            module.sa.proj.calculated_importance = proj_importances[idx]

            c += 1
            
            
def prune_embeddings(model, ratio=0.2) -> None:
    # goal: trim the embedding dimension of the weight matrices in MLP, MHA, and LayerNorm layers.
    importances = model.blocks[0].ln1.calculated_importance
    num_dense_embd = int((1 - ratio) * model.n_embd)
    idx = importances.argsort(descending=True)[:num_dense_embd]



    for module in model.modules():
        if isinstance(module, Block):
            # start with pruning the MLP layers
            importances = module.ln1.calculated_importance

            dense1 = module.ffwd.net[0]  # weights.shape = (emb, 4 * emb)
            dense2 = module.ffwd.net[2]  # weights.shape = (4 * emb, emb)

            module.ffwd.net[0] = nn.Linear(num_dense_embd, dense1.out_features).to(
                model.device
            )  # weights.shape = (num_dense_embd, dense1.in_features)
            module.ffwd.net[2] = nn.Linear(dense2.in_features, num_dense_embd).to(
                model.device
            )  # weights.shape = (dense2.out_features = emb)

            module.ffwd.net[0].weight.data = dense1.weight.data[:, idx]
            module.ffwd.net[0].bias.data = dense1.bias.data
            module.ffwd.net[2].weight.data = dense2.weight.data[idx, :]
            module.ffwd.net[2].bias.data = dense2.bias.data[idx]

            # now the multi-head attention
            for head in module.sa.heads:
                # key,value,query weight shape: (head_size, n_embd) # n_embd
                k, v, q = head.key, head.value, head.query

                head.key = nn.Linear(num_dense_embd, k.out_features, bias=False).to(
                    model.device
                )
                head.value = nn.Linear(num_dense_embd, v.out_features, bias=False).to(
                    model.device
                )
                head.query = nn.Linear(num_dense_embd, q.out_features, bias=False).to(
                    model.device
                )

                head.key.weight.data = k.weight.data[
                    :, idx
                ]  # (head_size, num_dense_embd)
                head.value.weight.data = v.weight.data[
                    :, idx
                ]  # (head_size, num_dense_embd)
                head.query.weight.data = q.weight.data[
                    :, idx
                ]  # (head_size, num_dense_embd)

                head.key.calculated_importance = k.calculated_importance
                head.value.calculated_importance = v.calculated_importance
                head.query.calculated_importance = q.calculated_importance

            ln1 = module.ln1
            ln2 = module.ln2

            module.ln1 = nn.LayerNorm(num_dense_embd).to(model.device)
            module.ln1.weight.data = ln1.weight.data[idx]
            module.ln1.bias.data = ln1.bias.data[idx]

            module.ln2 = nn.LayerNorm(num_dense_embd).to(model.device)
            module.ln2.weight.data = ln2.weight.data[idx]
            module.ln2.bias.data = ln2.bias.data[idx]

            proj = module.sa.proj
            module.sa.proj = nn.Linear(proj.in_features, num_dense_embd).to(
                model.device
            )
            module.sa.proj.weight.data = proj.weight.data[
                idx, :
            ]  # (num_dense_embd, n_embd)
            module.sa.proj.bias.data = proj.bias.data[idx]

            module.sa.proj.calculated_importance = proj.calculated_importance

    temb_table = model.token_embedding_table
    pemb_table = model.position_embedding_table

    model.token_embedding_table = nn.Embedding(model.vocab_size, num_dense_embd).to(
        model.device
        ) # type: ignore
    model.position_embedding_table = nn.Embedding(model.block_size, num_dense_embd).to(
        model.device
    )

    model.token_embedding_table.weight.data = temb_table.weight.data[:, idx]
    model.position_embedding_table.weight.data = pemb_table.weight.data[:, idx]

    lnf = model.ln_f
    ln_head = model.ln_head

    model.ln_f = nn.LayerNorm(num_dense_embd).to(model.device)
    model.ln_head = nn.Linear(num_dense_embd, ln_head.out_features).to(model.device)

    model.ln_f.weight.data = lnf.weight.data[idx]
    model.ln_f.bias.data = lnf.bias.data[idx]
    model.ln_head.weight.data = ln_head.weight.data[
        :, idx
    ]  # weight.shape = (vocab_size, embd)
    model.ln_head.bias.data = ln_head.bias.data


def pruning_n_handler(n, size, iters: int = 1):
    if isinstance(n, int):
        assert (
            n < size
        ), "`n` can't be higher than the calculated number of activation importances!"
        return [n] * iters

    elif isinstance(n, float) and 0 <= n < 1:  # if n is a ratio
        num = int((1 - n) * size)
        return [num] * iters

    elif isinstance(n, list):
        assert (
            len(n) == iters
        ), "the number of layers being pruned should be same with `iters`!"
        return n



AVAILABLE_PRUNING_STRATEGIES = {
    "width_head": prune_heads,
    "width_neuron": prune_neurons,
    "width_embedding": prune_embeddings,
}


