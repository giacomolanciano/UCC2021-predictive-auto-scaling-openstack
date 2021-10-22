# ---
# jupyter:
#   jupytext:
#     comment_magics: true
#     formats: ipynb,py:percent
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
import json
from datetime import timedelta
from itertools import zip_longest

import holoviews as hv
import pandas as pd
from constants import (
    DATA_ROOT,
    DATETIME_FORMAT,
    LIN_RUNS,
    MLP_RUNS,
    RNN_RUNS,
    STATIC_RUNS,
)
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

hv.extension("bokeh")
pd.options.plotting.backend = "holoviews"

# %%
static_runs_lim = (58, 58)
static_runs_exclude = set([])
lin_runs_lim = (4, 4)
lin_runs_exclude = set([])
mlp_runs_lim = (2, 2)
mlp_runs_exclude = set([])
rnn_runs_lim = (32, 32)
rnn_runs_exclude = set([])
run_time_limit = 85

# %% tags=[]
# load .json export files into DFs
df_list = list()

for real_file, pred_file in zip_longest(
    sorted(DATA_ROOT.glob("*-real.json")), sorted(DATA_ROOT.glob("*-pred.json"))
):
    i = int(real_file.name.split("-")[-2])
    if "-rnn-" in real_file.name:
        if i in rnn_runs_exclude:
            continue
        if rnn_runs_lim[0] is not None and i < rnn_runs_lim[0]:
            continue
        if rnn_runs_lim[1] is not None and i > rnn_runs_lim[1]:
            continue
    elif "-mlp-" in real_file.name:
        if i in mlp_runs_exclude:
            continue
        if mlp_runs_lim[0] is not None and i < mlp_runs_lim[0]:
            continue
        if mlp_runs_lim[1] is not None and i > mlp_runs_lim[1]:
            continue
    elif "-lin-" in real_file.name:
        if i in lin_runs_exclude:
            continue
        if lin_runs_lim[0] is not None and i < lin_runs_lim[0]:
            continue
        if lin_runs_lim[1] is not None and i > lin_runs_lim[1]:
            continue
    elif "-stc-" in real_file.name:
        if i in static_runs_exclude:
            continue
        if static_runs_lim[0] is not None and i < static_runs_lim[0]:
            continue
        if static_runs_lim[1] is not None and i > static_runs_lim[1]:
            continue

    print(f"reading from {real_file} and {pred_file} ...")

    with open(real_file, "r+") as fp:
        real_json_body = json.load(fp)

    real_metric = real_json_body[0]["name"]

    real_df = pd.DataFrame(
        columns=["timestamp", "resource_id", "hostname", real_metric]
    )
    for item in real_json_body:
        resource_id = item["dimensions"]["resource_id"]
        hostname = item["dimensions"]["hostname"]
        measurement_list = item["measurements"]
        real_df = real_df.append(
            [
                pd.Series([m[0], resource_id, hostname, m[1]], index=real_df.columns)
                for m in measurement_list
            ]
        )
    real_df = real_df.astype(
        {
            "resource_id": "string",
            "hostname": "string",
            real_metric: "float64",
        }
    )

    # cast index to DateTimeIndex
    real_df.set_index(["timestamp"], inplace=True)
    real_df.index = pd.to_datetime(real_df.index, format=DATETIME_FORMAT)

    pred_df = None
    if pred_file is not None:
        with open(pred_file, "r+") as fp:
            pred_json_body = json.load(fp)

        pred_metric = pred_json_body[0]["name"]

        pred_df = pd.DataFrame(columns=["timestamp", pred_metric])
        for item in pred_json_body:
            measurement_list = item["measurements"]
            pred_df = pred_df.append(
                [
                    pd.Series([m[0], m[1]], index=pred_df.columns)
                    for m in measurement_list
                ]
            )
        pred_df = pred_df.astype(
            {
                pred_metric: "float64",
            }
        )

        # cast index to DateTimeIndex
        pred_df.set_index(["timestamp"], inplace=True)
        pred_df.index = pd.to_datetime(pred_df.index, format=DATETIME_FORMAT)

    label = real_file.name.split("-real.json")[0]
    df_list.append((label, real_df, pred_df))

# %% tags=[]
fig_list = []
label_list = []
color_cycle = hv.Cycle(
    [
        "#30a2da",
        "#e5ae38",
        "#6d904f",
        "#8b8b8b",
        "#17becf",
        "#9467bd",
        "#e377c2",
        "#8c564b",
        "#bcbd22",
        "#1f77b4",
    ]
)
opts = [
    hv.opts.Scatter(size=5, marker="o"),
    hv.opts.Curve(tools=["hover"]),
]

for label, real_df, pred_df in df_list:
    traces = []
    index = int(label[-2:])

    if "-rnn-" in label:
        mapping = RNN_RUNS
    elif "-mlp-" in label:
        mapping = MLP_RUNS
    elif "-lin-" in label:
        mapping = LIN_RUNS
    elif "-stc-" in label:
        mapping = STATIC_RUNS

    table = pd.pivot_table(
        real_df,
        values="cpu.utilization_perc",
        index=["timestamp"],
        columns=["hostname"],
    )
    # table.index = pd.to_datetime(table.index, format=DATETIME_FORMAT)
    table = table.resample("1min").mean()

    # reorder columns by VMs start time
    table = table[
        table.apply(pd.Series.first_valid_index).sort_values().index.to_list()
    ]

    orig_cols = table.columns.copy()
    orig_cols_num = len(orig_cols)

    # compute spatial statistics to reconstruct info provided to Monasca
    table["sum"] = table.iloc[:, 0:orig_cols_num].sum(axis=1)

    if pred_df is not None:
        # insert prediction data to align timestamps
        table = table.join(pred_df, how="outer")

        # interpolate missing predictions
        table[pred_metric] = table[pred_metric].interpolate()

        # compute prediction error
        new_vm_start = table[orig_cols[2]].dropna().index[0]
        init_vms_interval = pd.date_range(end=new_vm_start, freq="min", periods=20)
        init_vms_data = table[orig_cols[:2]].loc[init_vms_interval].mean(axis=1)
        pred_interval = init_vms_interval - timedelta(minutes=15)
        pred_data = table[pred_metric].loc[pred_interval]
        pred_mape = mean_absolute_percentage_error(init_vms_data, pred_data)
        pred_mae = mean_absolute_error(init_vms_data, pred_data)
        print(f"{label} - MAPE: {pred_mape:.2f} | MAE: {pred_mae:.2f}")

    table.reset_index(inplace=True)

    # insert distwalk trace data to align timestamps
    load_file_basename = mapping[index]["load_profile"]
    load_file = DATA_ROOT / load_file_basename
    load_df = pd.read_csv(load_file, header=None, names=["distwalk"])
    table = table.join(load_df / 10, how="outer")

    # truncate data & remove NaN-only cols
    if run_time_limit:
        table = table.iloc[:run_time_limit, :].dropna(axis=1, how="all")

    # plot scale-out threshold
    traces.append(hv.HLine(80).opts(color="black", line_dash="dashed"))

    # plot distwalk trace
    distwalk_trace_label = "distwalk trace (per thread)"
    traces.append(
        hv.Scatter(
            (table.index, table["distwalk"].values),
            label=distwalk_trace_label,
        ).opts(color=color_cycle)
    )
    traces.append(
        hv.Curve(
            (table.index, table["distwalk"].values),
            label=distwalk_trace_label,
        ).opts(color=color_cycle)
    )

    # plot metrics observed by VMs
    instance_idx = 0
    for group_label in orig_cols:
        if group_label in table.columns:
            load_trace_label = f"instance {instance_idx}"
            traces.append(
                hv.Scatter(
                    (table.index, table[group_label].values),
                    label=load_trace_label,
                    kdims=[],
                ).opts(color=color_cycle)
            )
            traces.append(
                hv.Curve(
                    (table.index, table[group_label].values),
                    label=load_trace_label,
                ).opts(color=color_cycle)
            )
            instance_idx += 1

    if pred_df is not None:
        # plot predictor output
        prediction_trace_label = "predicted cluster avg"
        traces.append(
            hv.Scatter(
                (table.index, table[pred_metric].values),
                label=prediction_trace_label,
            ).opts(color="#d62728")
        )
        traces.append(
            hv.Curve(
                (table.index, table[pred_metric].values),
                label=prediction_trace_label,
            ).opts(color="#d62728")
        )

    title = f"{label} - load: {load_file_basename}"
    input_size = mapping[index].get("input_size")
    vm_delay_min = mapping[index].get("vm_delay_min")
    if input_size:
        title += f" | input size: {input_size}"
    if vm_delay_min:
        title += f" | vm delay: {vm_delay_min}"

    fig = (
        hv.Overlay(traces)
        .opts(
            width=950,
            height=550,
            show_grid=True,
            title=title,
            xlabel="time [min]",
            ylabel="CPU usage [%]",
            legend_position="top_left",
            fontsize={
                "title": 13,
                "legend": 15,
                "labels": 15,
                "xticks": 13,
                "yticks": 13,
            },
            padding=0.05,
        )
        .opts(opts)
    )
    fig_list.append(fig)
    label_list.append(label)

layout = hv.Layout(fig_list).cols(1)
layout
