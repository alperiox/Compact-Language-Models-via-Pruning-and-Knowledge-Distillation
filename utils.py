import itertools
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim.adamw import AdamW
from tqdm import tqdm

from hooks import register_all_forward_hooks, remove_all_forward_hooks
from models import GPT


class BatchLoader:
    def __init__(self, data, block_size, batch_size, device, name="batch_loader"):
        self.data = data
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device
        self.name = name

    def get_batch(self):
        ix = torch.randint(len(self.data) - self.block_size, (self.batch_size,))
        x = torch.stack([self.data[i : i + self.block_size] for i in ix])
        y = torch.stack([self.data[i + 1 : i + self.block_size + 1] for i in ix])
        x, y = x.to(self.device), y.to(self.device)
        return x, y


def get_num_params(model):
    t = 0
    for k in model.parameters():
        if k.requires_grad:
            t += k.numel()

    return t


def get_config_combinations(start: float = 0.1, end: float = 0.5, step: float = 0.15):
    # Define the range and step
    # Create the list of values for widths
    values = np.arange(start, end + step, step)

    # Initialize the experiment config list
    experiment_config = []

    for s in ["width_head", "width_neuron", "width_embedding"]:
        config = [[(s, round(v, 2))] for v in values]
        experiment_config.extend(config)

    # Setup 1: Vary width_head and width_neuron
    config1 = [
        [("width_head", round(wh, 2)), ("width_neuron", round(wn, 2))]
        for wh, wn in itertools.product(values, values)
    ]
    experiment_config.extend(config1)

    # Setup 2: Vary width_head and width_embedding
    config2 = [
        [("width_head", round(wh, 2)), ("width_embedding", round(we, 2))]
        for wh, we in itertools.product(values, values)
    ]
    experiment_config.extend(config2)

    # Setup 3: Vary width_neuron and width_embedding
    config3 = [
        [("width_neuron", round(wn, 2)), ("width_embedding", round(we, 2))]
        for wn, we in itertools.product(values, values)
    ]
    experiment_config.extend(config3)

    # Setup 4: Vary all three - width_head, width_neuron, and width_embedding
    config4 = [
        [
            ("width_head", round(wh, 2)),
            ("width_neuron", round(wn, 2)),
            ("width_embedding", round(we, 2)),
        ]
        for wh, wn, we in itertools.product(values, values, values)
    ]
    experiment_config.extend(config4)

    # Show an example of what the experiment_config looks like
    print(f"Total configurations: {len(experiment_config)}")

    return experiment_config


def get_model_with_importances(
    device, model_path, calibration_loader, batch_size, block_size
):
    model, _ = init_models(device, model_path)
    num_params = get_num_params(model)

    sample_batch = calibration_loader.data[: batch_size * block_size]
    sample_batch = sample_batch.view(batch_size, block_size)
    sample_batch = sample_batch.to(device)

    model(sample_batch)

    return model, num_params


def experiment(
    batch_size,
    block_size,
    vocab_size,
    calibration_loader,
    val_loader,
    device: str,
    pruning_strategies: list[list[tuple[str, float]]] = [
        [("width_head", 0.1), ("width_neuron", 0.1), ("width_embedding", 0.1)]
    ],
    learning_rate: float = 2e-3,
    model_path: str = "model",
):

    from pruners import prune_embeddings, prune_heads, prune_neurons

    results = []

    strategies = {
        "width_head": prune_heads,
        "width_neuron": prune_neurons,
        "width_embedding": prune_embeddings,
    }

    # initialize the base model and run a sample through
    base_model, num_params = get_model_with_importances(
        device, model_path, calibration_loader, batch_size, block_size
    )

    base_loss = estimate_loss(base_model, val_loader)["val"].item()

    for run in range(len(pruning_strategies)):
        print("-" * 50)
        strategy = pruning_strategies[run]

        pruning_funcs = [strategies[s] for s, ratio in strategy]
        pruning_func_names = [s for s, ratio in strategy]
        ratios = [ratio for s, ratio in strategy]

        print(f"RUN {run+1} | RATIO: {ratios} | STRATEGIES: {pruning_func_names}")
        model, num_params = get_model_with_importances(
            device, model_path, calibration_loader, batch_size, block_size
        )
        print(f"{'Number of trainable parameters before pruning:':60}", num_params)
        # prune
        for f, r in zip(pruning_funcs, ratios):
            f(model, r)
        #
        pruned_num_params = get_num_params(model)
        param_diff_ratio = (num_params - pruned_num_params) / num_params
        print(
            f"{'Number of training parameters after pruning:':60} {pruned_num_params}"
        )
        print(
            f"{'Ratio of the pruned weights to the base model:':60} {param_diff_ratio*100:.2f}%"
        )
        pruned_eval = estimate_loss(model, val_loader)["val"].item()
        print(f"{'Pruned evaluation loss (before calibration):':60} {pruned_eval:.4f}")
        #
        print("Starting the calibration")
        optimizer = AdamW(model.parameters(), lr=learning_rate)
        losses = kd_train_loop(
            model=model,
            optimizer=optimizer,
            vocab_size=vocab_size,
            train_loader=calibration_loader,
            batch_loaders=[calibration_loader, val_loader],
            max_iters=200,
            teacher_model=base_model,
            eval_interval=50,
            eval_iters=50,
        )
        #
        calibrated_eval = estimate_loss(model, val_loader)["val"].item()
        print(
            f"{'Pruned evaluation loss (after calibration):':60} {calibrated_eval:.4f}"
        )

        result = {
            "run": run + 1,
            "base_num_params": num_params,
            "pruned_num_params": pruned_num_params,
            "pruning_ratio": ratios,
            "param_diff_ratio": param_diff_ratio,
            "before_calibration_loss": pruned_eval,
            "after_calibration_loss": calibrated_eval,
            "base_loss": base_loss,
            "learning_rate": learning_rate,
            "pruning_strategies": pruning_func_names,
            "training_losses": losses,
        }

        results.append(result)
        run_df = pd.DataFrame(results)
        run_df.to_csv("run_results.csv", index=False)

    return results


def init_models(device, model_path: str = "model"):
    loaded_model, tokenizer = load(GPT, model_path)
    loaded_model.to(device)

    remove_all_forward_hooks(loaded_model)
    register_all_forward_hooks(loaded_model)

    return loaded_model, tokenizer


def save(model, tokenizer, model_params, path: str | Path) -> None:
    path = Path(path)

    os.makedirs(path, exist_ok=True)

    torch.save(model.state_dict(), path / "model.pth")

    with open(path / "tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    with open(path / "model_params.json", "w") as f:
        json.dump(model_params, f)


def load(model, save_dir: str | Path) -> tuple:
    save_dir = Path(save_dir)

    with open(save_dir / "tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open(save_dir / "model_params.json", "r") as f:
        model_params = json.load(f)

    model = model(**model_params)
    model.load_state_dict(torch.load(save_dir / "model.pth", weights_only=True))

    return model, tokenizer


@torch.no_grad()
def estimate_loss(
    model, batch_loaders: list[BatchLoader] | BatchLoader, eval_iters=200
):
    if isinstance(batch_loaders, BatchLoader):
        batch_loaders = [batch_loaders]
    out = {}
    model.eval()
    for loader in batch_loaders:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = loader.get_batch()
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[loader.name] = losses.mean()
    model.train()
    return out


def kd_train_loop(
    model,
    teacher_model,
    optimizer,
    vocab_size,
    train_loader,
    batch_loaders: list[BatchLoader],
    max_iters=1000,
    eval_interval=200,
    eval_iters=200,
):
    # uniform baseline score
    baseline_score = -torch.log(torch.tensor(1 / vocab_size)).item()
    print("UNIFORM BASELINE: ", baseline_score)
    training_losses = []

    loss_t = torch.tensor([0])
    teacher_model.eval()
    bar = tqdm(range(max_iters))
    for iter in bar:
        # sample a batch of data
        xb, yb = train_loader.get_batch()

        if iter % eval_interval == 0:
            losses = estimate_loss(model, batch_loaders, eval_iters)
            names = [loader.name for loader in batch_loaders]
            desc = ""
            for name in names:
                desc += f"{name} loss {losses[name]:.4f}, "
            bar.set_description(
                    f"step {iter}: {desc} \t teacher loss: {loss_t.item()} | baseline (uniform random): {baseline_score:.4f}"
            )
        # evaluate the loss

        logits, loss = model(xb, yb)
        teacher_logits, _ = teacher_model(xb, yb)

        loss_t =  torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(logits, dim=-1),
            torch.nn.functional.softmax(teacher_logits, dim=-1),
            reduction="batchmean",
        )


        loss = loss + loss_t

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        training_losses.append(loss.log10().item())

    return training_losses


def train_loop(
    model,
    optimizer,
    vocab_size,
    train_loader,
    batch_loaders: list[BatchLoader],
    max_iters=1000,
    eval_interval=200,
    eval_iters=200,
):
    # uniform baseline score
    baseline_score = -torch.log(torch.tensor(1 / vocab_size)).item()
    print("UNIFORM BASELINE: ", baseline_score)
    training_losses = []

    bar = tqdm(range(max_iters))
    for iter in bar:
        # sample a batch of data
        xb, yb = train_loader.get_batch()

        if iter % eval_interval == 0:
            losses = estimate_loss(model, batch_loaders, eval_iters)
            names = [loader.name for loader in batch_loaders]
            desc = ""
            for name in names:
                desc += f"{name} loss {losses[name]:.4f}, "
            bar.set_description(
                f"step {iter}: {desc} \t | baseline (uniform random): {baseline_score:.4f}"
            )

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        training_losses.append(loss.log10().item())

    return training_losses
