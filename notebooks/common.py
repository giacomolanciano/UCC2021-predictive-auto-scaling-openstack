"""Common utility functions"""

import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA

from constants import DATETIME_FORMAT


class MLP(nn.Module):
    def __init__(self, in_size: int, hidden_size: list, output_size: int):
        super(MLP, self).__init__()
        self.input_size = in_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dimensions = [self.input_size] + self.hidden_size + [self.output_size]
        # self.stack = nn.Linear(self.input_size, self.output_size)
        # self.layers = nn.Sequential(
        #                             nn.Flatten(),
        #                             nn.Linear(INPUT_SAMPLES, 1),
        #                             )
        # self.bias = torch.randn((1,1), requires_grad=True)
        self.stack = []
        for idx in range(len(self.dimensions) - 1):
            self.stack.append(nn.Linear(self.dimensions[idx], self.dimensions[idx + 1]))
            self.stack.append(nn.LeakyReLU())
        self.stack = nn.ModuleList(self.stack)

    def forward(self, x):
        signal = x
        for idx, processing in enumerate(self.stack):
            signal = processing(signal)
        # signal = self.layers(x) + self.bias
        # signal = x
        return signal


def train_batch_mlp(
    mlp,
    dataset: list,
    momentum,
    samples_in: int,
    horizon: int,
    beta=0.8,
    lr=0.001,
    batch_sz=100,
):
    loss = 0

    mlp.zero_grad()
    batch_idx = random.sample(range(len(dataset)), batch_sz)
    batch = [dataset[i] for i in batch_idx]
    for seq, target in batch:
        output = mlp(torch.tensor(seq, dtype=torch.float32).T)
        loss += torch.pow(
            torch.sub(output, torch.tensor(target, dtype=torch.float32)), 2
        )
    loss /= batch_sz
    loss = torch.sqrt(loss)
    loss.backward()

    with torch.no_grad():
        for i, p in enumerate(mlp.parameters()):
            momentum[i] = beta * momentum[i] + p.grad.data
            p.sub_(momentum[i], alpha=lr)
    return loss.data, momentum


def validation_step_mlp(mlp, samples_in: int, dataset: list, on_this_many: int = 0):
    loss = 0

    if on_this_many:
        batch_idx = random.sample(range(len(dataset)), on_this_many)
        val_dataset = [dataset[i] for i in batch_idx]
    else:
        # not sure this is needed but it doesn't hurt
        val_dataset = [dataset[i] for i in range(len(dataset))]

    for seq, target in val_dataset:
        output = mlp(torch.tensor(seq[-samples_in:], dtype=torch.float32).T)
        loss += torch.pow(torch.sub(output, target), 2)
    loss = torch.div(loss, on_this_many)
    loss = torch.sqrt(loss)
    return loss.data


def rolling_predictions_mlp(mlp, scaler, dataset, samples_in, limit=None):
    # assuming ds to be a list of ordered overlapping sub-sequences
    # (stride = 1)
    predictions = []
    for idx, item in enumerate(dataset):
        if limit and idx >= limit - samples_in + 1:
            break

        seq, actual_val = item
        if len(seq) > samples_in:
            raise ValueError("sequence is longer than expected")

        output = mlp(torch.tensor(seq, dtype=torch.float32).T)

        actual_val = float(scaler.inverse_transform(np.array([[actual_val]])).squeeze())
        out_val = float(
            scaler.inverse_transform(np.array([[output.data.item()]])).squeeze()
        )
        out_idx = idx + samples_in - 1
        predictions.append([out_idx, out_val, actual_val])

    return np.array(predictions)


class RNN(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, layers):
        super(RNN, self).__init__()
        self.input_size = in_size
        self.hidden_size = hidden_size
        self.output_size = out_size
        self.layers = layers
        self.rnn = nn.RNN(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.layers,
            batch_first=False,
            nonlinearity="relu",
        )
        self.lin = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, seq, hidden):
        output, _ = self.rnn(seq, hidden)
        output = self.lin(output[-1:])
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.layers, batch_size, self.hidden_size)


def run2seq(filename: str, datetime_format=DATETIME_FORMAT, limit=None):
    df = pd.read_csv(
        filename,
        parse_dates=["timestamp"],
    )

    if "hostname" in df.columns:
        table = pd.pivot_table(
            df, values="cpu.utilization_perc", index=["timestamp"], columns=["hostname"]
        )
        table.index = pd.to_datetime(table.index, format=datetime_format)
        table = table.resample("1min").mean()
        table["vm_count"] = (~table.isnull()).iloc[:, :3].sum(axis=1)
        table["vm_sum"] = table.iloc[:, 0:3].sum(axis=1)
        # table = table[table["vm_sum"] > 20]
        # data = table[['vm_count', 'vm_sum']].values
        data = table[["vm_sum"]].values
    else:
        # assuming df.columns == ['timestamp', '<METRIC_NAME>']
        data = df.iloc[:, 1].values.reshape(-1, 1)

    return data


def scale_seq(data, samples_in: int, horizon_min: int, scaler):
    data_p = scaler.transform(data)

    # separate sequences
    split_sequences = []
    for i in range(len(data_p) - samples_in - horizon_min + 1):
        val = data_p[i : i + samples_in], data_p[i + samples_in + horizon_min - 1, 0]
        # print(val)
        split_sequences.append(val)
    return split_sequences, data_p


def gen_dataset(files: list, samples_in: int, horizon_min: int, scaler=None):
    if scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(np.array([[0], [400]]))

    seq_list = []
    for filename in files:
        print(f"reading {filename} ...")
        res = run2seq(filename)
        if res is not None:
            seq_list.append(res)

    dataset = []
    for seq in seq_list:
        res = scale_seq(seq, samples_in, horizon_min, scaler)
        if res is not None:
            dataset += res[0]

    return dataset, scaler


def train_step(rnn, samples_in: int, dataset: list, momentum, beta=0.8, lr=0.03):
    loss = 0

    rnn.zero_grad()
    for seq, target in dataset:
        hidden = rnn.init_hidden()
        for i in range(samples_in):
            hidden, output = rnn(torch.tensor(seq[i], dtype=torch.float32), hidden)
        loss += torch.pow(torch.sub(output, target), 2)
    loss = torch.div(loss, len(dataset))
    loss = torch.sqrt(loss)
    loss.backward()

    with torch.no_grad():
        for i, p in enumerate(rnn.parameters()):
            momentum[i] = beta * momentum[i] + p.grad.data
            p.sub_(momentum[i], alpha=lr)
    return loss.data, momentum, rnn


def train_step_batched(
    rnn, samples_in: int, dataset: list, momentum, batch_sz=None, beta=0.9, lr=0.01
):
    loss = 0

    rnn.zero_grad()
    batch_idx = random.sample(range(len(dataset)), batch_sz)
    batch = [dataset[i] for i in batch_idx]
    for seq, target in batch:
        hidden = rnn.init_hidden()
        for i in range(samples_in):
            hidden, output = rnn(torch.tensor(seq[i], dtype=torch.float32), hidden)
        loss += (output - target) ** 2
    loss /= batch_sz
    loss = torch.sqrt(loss)
    loss.backward()

    with torch.no_grad():
        for i, p in enumerate(rnn.parameters()):
            momentum[i] = beta * momentum[i] + (1 - beta) * p.grad.data
            p.sub_(momentum[i], alpha=lr)
    return loss.data, momentum, rnn


def validation_step(rnn, samples_in: int, dataset: list, on_this_many: int = 0):
    loss = 0
    if on_this_many:
        batch_idx = random.sample(range(len(dataset)), on_this_many)
        val_dataset = [dataset[i] for i in batch_idx]
    else:
        # not sure this is needed but it doesn't hurt
        val_dataset = [dataset[i] for i in range(len(dataset))]
    rnn.zero_grad()
    for seq, target in val_dataset:
        hidden = rnn.init_hidden()
        for i in range(samples_in):
            hidden, output = rnn(torch.tensor(seq[i], dtype=torch.float32), hidden)
        loss += torch.pow(torch.sub(output, target), 2)
    loss = torch.div(loss, on_this_many)
    loss = torch.sqrt(loss)

    return loss.data


def rolling_predictions(rnn, scaler, dataset, samples_in, limit=None):
    # assuming ds to be a list of ordered overlapping sub-sequences
    # (stride = 1)
    predictions = []
    for idx, item in enumerate(dataset):
        if limit and idx >= limit - samples_in + 1:
            break

        seq, actual_val = item
        if len(seq) > samples_in:
            raise ValueError("sequence is longer than expected")

        a = torch.tensor(seq, dtype=torch.float32)
        a = a.unsqueeze(1)
        h0 = rnn.init_hidden(1)
        output, _ = rnn(a, h0)
        # output.squeeze_()

        actual_val = float(scaler.inverse_transform(np.array([[actual_val]])).squeeze())
        out_val = float(
            scaler.inverse_transform(np.array([[output.data.item()]])).squeeze()
        )
        out_idx = idx + samples_in - 1
        predictions.append([out_idx, out_val, actual_val])

    return np.array(predictions)


def rolling_predictions_lin(scaler, dataset, samples_in, pred_offset=15, limit=None):
    # assuming ds to be a list of ordered overlapping sub-sequences
    # (stride = 1)
    predictions = []
    for idx, item in enumerate(dataset):
        if limit and idx >= limit - samples_in + 1:
            break

        seq, actual_val = item
        # if len(seq) > samples_in:
        #     raise ValueError("sequence is longer than expected")
        seq = seq[-5:]

        time_steps = np.arange(0, len(seq))
        model = LinearRegression().fit(time_steps.reshape(-1, 1), seq)

        output = float(
            model.predict(np.array([[time_steps[-1] + pred_offset]])).squeeze()
        )

        actual_val = float(scaler.inverse_transform(np.array([[actual_val]])).squeeze())
        out_val = float(scaler.inverse_transform(np.array([[output]])).squeeze())
        out_idx = idx + samples_in - 1
        predictions.append([out_idx, out_val, actual_val])

    return np.array(predictions)


def rolling_predictions_arima(
    model, scaler, dataset, samples_in, pred_offset=15, limit=None
):
    # assuming ds to be a list of ordered overlapping sub-sequences
    # (stride = 1)
    retrain = model is None
    predictions = []
    for idx, item in enumerate(dataset):
        if limit and idx >= limit - samples_in + 1:
            break

        seq, actual_val = item

        time_steps = np.arange(0, len(seq))
        pred_start = pred_end = time_steps[-1] + pred_offset

        if retrain:
            model = ARIMA(seq, order=(1, 1, 0)).fit()
            output = float(model.predict(start=pred_start, end=pred_end).squeeze())
        else:
            model = model.apply(seq)
            output = float(model.forecast(pred_offset)[-1].squeeze())

        actual_val = float(scaler.inverse_transform(np.array([[actual_val]])).squeeze())
        out_val = float(scaler.inverse_transform(np.array([[output]])).squeeze())
        out_idx = idx + samples_in - 1
        predictions.append([out_idx, out_val, actual_val])

    return np.array(predictions)
