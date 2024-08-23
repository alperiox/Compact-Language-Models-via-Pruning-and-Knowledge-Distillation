import pytest
from models import GPT, Block
import torch


@pytest.fixture 
def hyperparameters() -> dict[str, str|float|int]:
    block_size = 16  # maximum context length for the preds
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    n_embd = 32
    n_head = 2
    n_blocks = 2
    dropout = 0.2

    return {
            "block_size": 16,
            "device": device,
            "n_embd": n_embd,
            "n_head": n_head,
            "n_blocks": n_blocks,
            "dropout": dropout,
        }

@pytest.fixture
def example_model(hyperparameters) -> GPT:
    hyperparameters['vocab_size'] = 64
    model = GPT(**hyperparameters)
    model.to(hyperparameters['device'])

    return model

@pytest.fixture 
def example_dataset() -> torch.Tensor | Tokenizer:
    from tokenizers import Tokenizer

    data = open("dataset/calibration.txt", "r").read()
    chars = sorted(list(set(data)))
    vocab_size = len(chars)

    tokenizer = Tokenizer(chars)

    tokenized_data = torch.tensor(tokenizer.encode(data), dtype=torch.long)

    return tokenized_data, tokenizer



@pytest.fixture 
def hooked_model(hyperparameters) -> GPT:
    from hooks import register_all_forward_hooks 

    data = open("dataset/calibration.txt", "r").read()
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    hyperparameters['vocab_size'] = vocab_size 
    model = GPT(**hyperparameters)

    return model



def test_registering_forward_hooks(example_model):
    from hooks import register_all_forward_hooks
    from hooks import (
        attn_head_importance_hook,
        neuron_importance_hook,
        embedding_importance_hook,
        block_importance_hook,
    )

    register_all_forward_hooks(example_model)

    for module in example_model.modules():
        if isinstance(module, Block):
            for head in module.sa.heads:
                assert attn_head_importance_hook in head._forward_hooks.values()

                assert neuron_importance_hook in head.key._forward_hooks.values()
                assert neuron_importance_hook in head.value._forward_hooks.values()
                assert neuron_importance_hook in head.query._forward_hooks.values()

            assert neuron_importance_hook in module.ffwd.net[0]._forward_hooks.values()
            assert neuron_importance_hook in module.sa.proj._forward_hooks.values()
            assert embedding_importance_hook in module.ln1._forward_hooks.values()
            assert block_importance_hook in module._forward_hooks.values()


def test_removing_forward_hooks():
    pass


def test_neuron_importance_hook(example_model: GPT):
    from hooks import register_all_forward_hooks

    register_all_forward_hooks(example_model)

    










