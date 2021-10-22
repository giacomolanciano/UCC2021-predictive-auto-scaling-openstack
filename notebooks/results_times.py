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
import sys

import holoviews as hv
import pandas as pd
from constants import DATA_ROOT, LIN_RUNS, MLP_RUNS, RNN_RUNS, STATIC_RUNS
from holoviews.operation.datashader import datashade

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

# %% [markdown]
# ## Client-side response times plots

# %%
descr_stats_table = pd.DataFrame()
times_fig_list = []
times_label_list = []
for times_file in sorted(DATA_ROOT.glob("*-times.csv")):
    i = int(times_file.name.split("-")[-2])
    if "-rnn-" in times_file.name:
        if i in rnn_runs_exclude:
            continue
        if rnn_runs_lim[0] is not None and i < rnn_runs_lim[0]:
            continue
        if rnn_runs_lim[1] is not None and i > rnn_runs_lim[1]:
            continue
        label = "RNN"
        mapping = RNN_RUNS
    elif "-mlp-" in times_file.name:
        if i in mlp_runs_exclude:
            continue
        if mlp_runs_lim[0] is not None and i < mlp_runs_lim[0]:
            continue
        if mlp_runs_lim[1] is not None and i > mlp_runs_lim[1]:
            continue
        label = "MLP"
        mapping = MLP_RUNS
    elif "-lin-" in times_file.name:
        if i in lin_runs_exclude:
            continue
        if lin_runs_lim[0] is not None and i < lin_runs_lim[0]:
            continue
        if lin_runs_lim[1] is not None and i > lin_runs_lim[1]:
            continue
        label = "LR"
        mapping = LIN_RUNS
    else:
        if i in static_runs_exclude:
            continue
        if static_runs_lim[0] is not None and i < static_runs_lim[0]:
            continue
        if static_runs_lim[1] is not None and i > static_runs_lim[1]:
            continue
        label = "Static"
        mapping = STATIC_RUNS

    traces = []

    print(times_file)
    df = pd.read_csv(times_file, header=None, names=["timestamp", "delay"])

    # convert microsec to millisec
    df = df / 1000

    # convert timestamp to min
    df["timestamp"] = df["timestamp"] / 1000 / 60

    if run_time_limit:
        df = df[df["timestamp"] < run_time_limit]

    # compute percentiles
    stats_col_name = f"{label}"
    input_size = mapping[i].get("input_size")
    vm_delay_min = mapping[i].get("vm_delay_min")
    if input_size:
        stats_col_name += f" ({input_size:02})"

    descr_stats_table[stats_col_name] = pd.Series(
        {
            "avg (ms)": df["delay"].mean(),
            "p90 (ms)": df["delay"].quantile(0.9),
            "p95 (ms)": df["delay"].quantile(0.95),
            "p99 (ms)": df["delay"].quantile(0.99),
            "p99.5 (ms)": df["delay"].quantile(0.995),
            "p99.9 (ms)": df["delay"].quantile(0.999),
        }
    )

    # filter out outliers before plotting
    df = df[df["delay"] > 0]
    df = df[df["delay"] <= df["delay"].quantile(0.995)]

    trace_label = f"{label}"
    if input_size:
        trace_label += f" ({input_size:02})"
    traces.append(
        hv.Scatter(
            (df["timestamp"].values, df["delay"].values),
            label=trace_label,
        )
    )

    fig = hv.Overlay(traces).opts(
        width=950,
        height=550,
        show_grid=True,
        title=f"{times_file.name}",
        xlabel="time [min]",
        ylabel="delay [ms]",
        legend_position="top_left",
        fontsize={
            "title": 15,
            "legend": 15,
            "labels": 15,
            "xticks": 13,
            "yticks": 13,
        },
        logy=True,
    )
    times_fig_list.append(fig)
    times_label_list.append(times_file.stem)

times_layout = datashade(hv.Layout(times_fig_list).cols(1).opts(shared_axes=False))
times_layout

# %% [markdown]
# ## Client-side response times tables

# %%
# render client-side delays stats table
ordered_groups = ["Static", "LR", "MLP", "RNN"]
ordered_cols = []
for group in ordered_groups:
    ordered_cols += (
        descr_stats_table.filter(regex=f"^{group}", axis=1)
        .columns.sort_values()
        .to_list()
    )
printable_table = descr_stats_table[ordered_cols]

col_fmt = "r" + "r" * printable_table.columns.size
printable_table.round(2).to_latex(sys.stdout, column_format=col_fmt)
printable_table
