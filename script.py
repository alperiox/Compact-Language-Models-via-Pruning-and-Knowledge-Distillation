import matplotlib.pyplot as plt
import torch
from torch.optim.adamw import AdamW
from torch.nn import functional as F

from models import GPT
from tokenizers import Tokenizer
from utils import BatchLoader, experiment, save, train_loop, kd_train_loop

# hyperparameters
batch_size = 16  # number of independent sequences that'll be processed in parallel
block_size = 128  # maximum context length for the preds
max_iters = 1000
eval_interval = 200
learning_rate = 3e-4
device = "mps" if torch.backends.mps.is_available() else ("cuda:0" if torch.cuda.is_available() else "cpu")
eval_iters = 200
n_embd = 256
n_head = 4
n_blocks = 4
dropout = 0.2
# --------------

torch.manual_seed(1337)

# data preparation
text = open("dataset/tinyshakespeare.txt", "r").read()
# set up the vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
tokenizer = Tokenizer(chars)

data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

n = int(0.9 * len(data))  # first 90% will be the training set
n1 = int(
    0.95 * len(data)
)  # 90-95% will be the validation set and the last 5% will be the calibration set for the paper

train_data = data[:n]
val_data = data[n:n1]
calibrate_data = data[n1:]


train_loader = BatchLoader(train_data, block_size, batch_size, device, name="train")
val_loader = BatchLoader(val_data, block_size, batch_size, device, name="val")
calibration_loader = BatchLoader(
    calibrate_data, block_size, batch_size, device, name="calibrate"
)

model = GPT(vocab_size, block_size, n_embd, n_head, n_blocks, device, dropout)
model.to(device)

optimizer = AdamW(model.parameters(), lr=learning_rate)

training_losses = train_loop(
    model,
    optimizer,
    vocab_size,
    train_loader,
    [train_loader, val_loader],
    max_iters,
    eval_interval,
    eval_iters,
)

print("training is done!")

plt.title("training losses")
plt.plot(training_losses)
plt.savefig("training_losses.png")

idx = torch.zeros((1, 1), dtype=torch.long, device=device)
print(tokenizer.decode(model.generate(idx, max_new_tokens=500)[0].tolist()))

model_params = {
    "vocab_size": vocab_size,
    "block_size": block_size,
    "n_embd": n_embd,
    "n_head": n_head,
    "n_blocks": n_blocks,
    "dropout": dropout,
    "device": device,
}

save(model, tokenizer, model_params, "model")

print("-" * 50)
print("The experiment is going to start soon!")

# configurations = get_config_combinations()[:-2]
configurations = [[
    ("width_head", 0.2),
    ("width_neuron", 0.2),
    ("width_embedding", 0.2),
]]


results = experiment(
    batch_size,
    block_size,
    vocab_size,
    calibration_loader,
    val_loader,
    pruning_strategies=configurations,
    learning_rate=learning_rate,
    model_path="./model",
)
