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

import pandas as pd
from constants import DATA_ROOT

TOTAL_TIME_SEC_COL = "total (sec)"
PROCESSING_TIME_SEC_COL = "processing (sec)"
TOTAL_TIME_MS_COL = "total (ms)"
PROCESSING_TIME_MS_COL = "processing (ms)"

# %% [markdown]
# ## Average overhead table

# %%
descr_stats_table = pd.DataFrame()
for times_file in sorted(DATA_ROOT.glob("predictor-times-*.csv")):
    if "-rnn." in times_file.name:
        label = "RNN"
    elif "-mlp." in times_file.name:
        label = "MLP"
    elif "-lin." in times_file.name:
        label = "LR"

    print(times_file)
    df = pd.read_csv(times_file)

    # convert sec to millisec
    df = df * 1000
    df.rename(
        columns={
            TOTAL_TIME_SEC_COL: TOTAL_TIME_MS_COL,
            PROCESSING_TIME_SEC_COL: PROCESSING_TIME_MS_COL,
        },
        inplace=True,
    )

    # compute stats
    descr_stats_table[label] = pd.Series(
        {
            "forecasting time (ms)": df[PROCESSING_TIME_MS_COL].mean(),
            "total time (ms)": df[TOTAL_TIME_MS_COL].mean(),
        }
    )

# %%
ordered_groups = ["LR", "MLP", "RNN"]
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

# %%
