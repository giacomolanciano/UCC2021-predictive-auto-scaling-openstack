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

import torch

from constants import DATETIME_FORMAT, DATA_ROOT
from common import MLP, train_batch_mlp, gen_dataset, run2seq

import matplotlib.pyplot as plt


INPUT_SAMPLES = 5
PREDICTION_OFFSET = 15

# %%
file_list = []
for export_file in DATA_ROOT.glob("super_steep_behavior.csv"):
    file_list.append(export_file)

ds, scaler = gen_dataset(file_list, INPUT_SAMPLES, PREDICTION_OFFSET)

# %%
mlp = MLP(INPUT_SAMPLES, [10, 10], 1)

# %%
momentum = []
for p in mlp.parameters():
    momentum.append(torch.zeros_like(p))
iterations = 1000
lr = 0.1
lr_space = []
for i in range(5000):
    lr_space.append(lr)
    lr = max(0.001, lr * 0.995)

for i in range(iterations):
    mlp.zero_grad()
    lr = lr_space[i]
    loss, momentum = train_batch_mlp(
        mlp, ds, momentum, INPUT_SAMPLES, horizon=PREDICTION_OFFSET, batch_sz=500, lr=lr
    )
    if i % 50 == 49:
        print(f"iter: {i} loss: {loss}, lr {lr:.5f}")


# %%
current_date = datetime.today().strftime("%Y-%m-%d")
dump_filename = f"mlp-{INPUT_SAMPLES:02}_sum_{current_date}.pt"
print(dump_filename)
torch.save(mlp.state_dict(), dump_filename)

# %% [markdown] tags=[]
# ## Test

# %%
data = scaler.transform(
    run2seq(DATA_ROOT / "super_steep_behavior.csv", DATETIME_FORMAT)
)


# %%
start = random.randint(0, len(data) - 200)
end = start + 200
point = random.randint(start, end)
a = mlp(torch.tensor(data[point : point + INPUT_SAMPLES], dtype=torch.float32).T)
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
plt.plot(point + INPUT_SAMPLES + PREDICTION_OFFSET, a.data, "xg")
plt.legend(["dataset", "input samples", "prediction"])
