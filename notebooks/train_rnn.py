# ---
# jupyter:
#   jupytext:
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: pred-as-os
#     language: python
#     name: pred-as-os
# ---

# %%
import random
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from common import RNN, run2seq
from constants import DATA_ROOT
from sklearn.preprocessing import MinMaxScaler

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

INPUT_SAMPLES = 5
PREDICTION_OFFSET = 15
HIDDEN_UNITS = 200
LAYERS = 3


# %%
def train_batch_rnn(
    rnn,
    ds: list,
    momentum,
    samples_in: int,
    horizon: int,
    beta=0.5,
    lr=0.001,
    batch_sz=100,
):
    # prepare batch
    batch_idx = random.sample(range(len(ds) - samples_in - horizon), batch_sz)

    mat = torch.tensor(
        ds[batch_idx[0] : batch_idx[0] + samples_in], dtype=torch.float32
    )
    mat = mat.unsqueeze(1)

    targets = torch.tensor(ds[batch_idx[0] + horizon - 1], dtype=torch.float32)
    targets = targets.unsqueeze(1)

    for i in batch_idx[1:]:
        a = torch.tensor(ds[i : i + samples_in], dtype=torch.float32)
        a = a.unsqueeze(1)
        mat = torch.cat((mat, a), dim=1)

        t = torch.tensor(ds[i + samples_in + horizon - 1], dtype=torch.float32)
        t = t.unsqueeze(1)
        targets = torch.cat((targets, t), dim=1)

    rnn.zero_grad()

    h0 = rnn.init_hidden(batch_sz)
    output, _ = rnn(mat, h0)
    output.squeeze_(2)

    loss = torch.mean(torch.pow(output - targets, 2), dim=1)
    loss.sqrt_()
    loss.backward()

    with torch.no_grad():
        for i, p in enumerate(rnn.parameters()):
            momentum[i] = beta * momentum[i] + (1 - beta) * p.grad.data
            p.sub_(momentum[i], alpha=lr)
    return loss.data, momentum


# %%
net = RNN(1, HIDDEN_UNITS, 1, LAYERS)
print(net)

# %%
scaler = MinMaxScaler()
scaler.fit(np.array([[0], [200]]))
joblib.dump(scaler, "rnn_scaler.joblib")

data = scaler.transform(run2seq(DATA_ROOT / "super_steep_behavior.csv"))

# %%
iterations = 10000
lr = 0.01
lr_space = []
for i in range(iterations):
    lr_space.append(lr)
    lr = max(0.001, lr * 0.9995)

plt.plot(lr_space)

# %%
loss_storage = []
momentum = []
for p in net.parameters():
    momentum.append(torch.zeros_like(p))

for i in range(iterations):
    net.zero_grad()
    lr = lr_space[i]
    loss, momentum = train_batch_rnn(
        net,
        data,
        momentum,
        INPUT_SAMPLES,
        horizon=PREDICTION_OFFSET,
        batch_sz=300,
        lr=lr,
    )
    if i % 100 == 99:
        print(f"iter: {i} loss: {loss}, lr {lr:.5f}")
    loss_storage.append(loss)


# %%
current_date = datetime.today().strftime("%Y-%m-%d")
dump_filename = f"rnn-{INPUT_SAMPLES:02}_sum_{current_date}.pt"
print(dump_filename)
torch.save(net.state_dict(), dump_filename)
plt.plot(loss_storage)

# %% [markdown]
# ## Test

# %%
start = random.randint(0, len(data) - 200)
end = start + 200
point = random.randint(start, end)
seq = data[point : point + INPUT_SAMPLES]
plt.figure(figsize=(15, 10))
plt.plot(list(range(start, end)), data[start:end], ".")
plt.grid()
plt.plot(
    list(range(point, point + INPUT_SAMPLES)), data[point : point + INPUT_SAMPLES], "xr"
)
plt.plot(
    point + INPUT_SAMPLES + PREDICTION_OFFSET,
    data[point + INPUT_SAMPLES + PREDICTION_OFFSET],
    "xk",
)

a = torch.tensor(seq, dtype=torch.float32)
a = a.unsqueeze(1)
h0 = net.init_hidden(1)
output, _ = net(a, h0)
output.squeeze_()
plt.plot(point + INPUT_SAMPLES + PREDICTION_OFFSET, output.data, "xg")
plt.legend(["dataset", "input samples", "prediction"])
