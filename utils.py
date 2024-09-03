import json
import os
import pickle
from pathlib import Path

from copy import deepcopy
import pandas as pd
import torch
from torch.optim.adamw import AdamW
from tqdm.auto import tqdm

from hyperopt import fmin, tpe, Trials, STATUS_OK

from hooks import register_all_forward_hooks, remove_all_forward_hooks
from models import GPT

from pruners import AVAILABLE_PRUNING_STRATEGIES


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


def bayesian_optimization_objective(args):
    (
        param_ub,
        param_lb,
        configuration,
        training_args,
    ) = args  # (ub, lb, model, dict(config))
    # copy the model
    copy_model = deepcopy(training_args["teacher_model"])
    # prune the model and calculate the number of parameters
    for conf in configuration:
        f = AVAILABLE_PRUNING_STRATEGIES[conf[0]]
        f(copy_model, conf[1]/100) # since the hyperparameter comes in between [0, 90], we need to scale it down

    num_params = get_num_params(copy_model)
    training_args["model"] = copy_model

    if param_lb < num_params < param_ub:
        losses = kd_train_loop(**training_args, verbose=False)

        return {
            "loss": sum([10**k for k in losses]) / len(losses),
            "num_params": num_params,
            "status": STATUS_OK,
        }
    else:
        return {"loss": float("inf"), "num_params": num_params, "status": STATUS_OK}


def architecture_search(space, num_evals=100):
    results = Trials()
    results_list = []

    best = fmin(
        bayesian_optimization_objective,
        space,
        algo=tpe.suggest,
        max_evals=num_evals,
        trials_save_file="./trials.hyperopt",
        trials=results,
    )

    for trial in results.trials:
        sample = trial.copy()
        sample["vals"] = sample["misc"]["vals"]
        sample["status"] = sample["result"]["status"]
        sample["loss"] = sample["result"]["loss"]
        sample["num_params"] = sample["result"]["num_params"]
        misc_to_del = ["misc", "spec", "result", "exp_key", "owner", "version"]
        for v in misc_to_del:
            del sample[v]
        results_list.append(sample)

    results_df = pd.DataFrame(results_list)
    results_df = (
        results_df[results_df["loss"] != float("inf")]
        .sort_values(by="loss")
        .reset_index(drop=True)
    )
    results_df.to_csv("trial_results.csv", index=False)

    return results_df, best


def experiment(
    batch_size,
    block_size,
    vocab_size,
    calibration_loader,
    val_loader,
    device: str,
    pruning_strategies: list[list[tuple[str, float | list[int] | int]]] = [
        [("width_head", 0.1), ("width_neuron", 0.1), ("width_embedding", 0.1)]
    ],
    learning_rate: float = 2e-3,
    model_path: str = "model",
):

    results = []

    # initialize the base model and run a sample through
    base_model, num_params = get_model_with_importances(
        device, model_path, calibration_loader, batch_size, block_size
    )

    base_loss = estimate_loss(base_model, val_loader)["val"].item()

    print(f"Base loss after the initial training: {base_loss:.4f}")
    for run in range(len(pruning_strategies)):
        print("-" * 50)
        strategy = pruning_strategies[run]

        pruning_funcs = [AVAILABLE_PRUNING_STRATEGIES[s] for s, _ in strategy]
        pruning_func_names = [s for s, _ in strategy]
        constraints = [constr for _, constr in strategy]

        print(f"RUN {run+1} | RATIO: {constraints} | STRATEGIES: {pruning_func_names}")
        model, num_params = get_model_with_importances(
            device, model_path, calibration_loader, batch_size, block_size
        )

        print(f"{'Number of trainable parameters before pruning:':60}", num_params)
        # prune
        for f, r in zip(pruning_funcs, constraints):
            f(model, r)
        #
        print(model)
        print("-" * 100)
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
            "pruning_constraints": constraints,
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


def load(model, save_dir: str | Path, pruned: bool = False) -> tuple:
    save_dir = Path(save_dir)
    tokenizer_path = save_dir / "tokenizer.pkl"
    model_params_path = save_dir / "model_params.json"
    model_path = save_dir / "model.pth"

    assert (
        tokenizer_path.exists() and model_params_path.exists()
    ), "`tokenizer.pkl` or `model_params.json` couldn't be found!"

    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    with open(model_params_path, "r") as f:
        model_params = json.load(f)

    model = model(**model_params["params"])

    if pruned:
        assert model_params.get(
            "optimal_pruning_strategy", False
        ), "There must be `optimal_pruning_strategy` key in the `model_params`!"

        pruning_strategy = model_params["optimal_pruning_strategy"]
        for name, ratio in pruning_strategy.items():
            f = AVAILABLE_PRUNING_STRATEGIES[name]
            f(model, ratio)
        model.load_state_dict(
            torch.load(save_dir / "model_pruned.pth", weights_only=True)
        )
    else:
        model.load_state_dict(torch.load(model_path, weights_only=True))

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
    verbose=True,
):
    # uniform baseline score
    baseline_score = -torch.log(torch.tensor(1 / vocab_size)).item()
    if verbose:
        print("UNIFORM BASELINE: ", baseline_score)

    training_losses = []

    loss_t = torch.tensor([0])
    loss_s = torch.tensor([0])
    teacher_model.eval()

    if verbose:
        bar = tqdm(range(max_iters))
    else:
        bar = range(max_iters)

    for i in bar:
        # sample a batch of data
        xb, yb = train_loader.get_batch()

        if i % eval_interval == 0:
            losses = estimate_loss(model, batch_loaders, eval_iters)
            names = [loader.name for loader in batch_loaders]

            if verbose:
                desc = ""
                for name in names:
                    desc += f"{name} loss {losses[name]:.4f}, "

                bar.set_description(
                    f"step {i}: {desc} \t teacher loss: {loss_t.item():.4f} \t student loss: {loss_s.item():.4f} | baseline (uniform random): {baseline_score:.4f}"
                )
        # evaluate the loss

        logits, loss_s = model(xb, yb)
        teacher_logits, _ = teacher_model(xb, yb)

        loss_t = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(logits, dim=-1),
            torch.nn.functional.softmax(teacher_logits, dim=-1),
            reduction="batchmean",
        )

        loss = loss_s + loss_t

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
