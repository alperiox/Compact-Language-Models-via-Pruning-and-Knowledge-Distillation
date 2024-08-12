import json
import os
import pickle
from pathlib import Path

import torch
from tqdm import tqdm


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

def save(model, tokenizer, model_params, path: str) -> None:
    path = Path(path)

    os.makedirs(path, exist_ok=True)

    torch.save(model.state_dict(), path / "model.pth")

    with open(path / "tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    with open(path / "model_params.json", "w") as f:
        json.dump(model_params, f)

def load(model, save_dir: str) -> tuple:
    save_dir = Path(save_dir)

    with open(save_dir / "tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open(save_dir / "model_params.json", "r") as f:
        model_params = json.load(f)

    model = model(**model_params)
    model.load_state_dict(torch.load(save_dir / "model.pth"))

    return model, tokenizer


@torch.no_grad()
def estimate_loss(model, batch_loaders: list[BatchLoader] | BatchLoader, eval_iters=200):
    if isinstance(batch_loaders, BatchLoader):
        batch_loaders = [batch_loaders]
    out = {}
    model.eval()
    for loader in batch_loaders:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = loader.get_batch()
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[loader.name] = losses.mean()
    model.train()
    return out


def train_loop(model, optimizer, vocab_size, train_loader, batch_loaders: list[BatchLoader] | BatchLoader, max_iters=1000, eval_interval=200, eval_iters=200):
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
            bar.set_description(
                f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        training_losses.append(loss.log10().item())
    
    return training_losses