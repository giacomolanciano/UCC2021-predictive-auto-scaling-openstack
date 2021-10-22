#!/bin/bash

set -e
set -o pipefail

predictor_list=("lin" "mlp" "rnn")

log_file=$(realpath "$1")
log_file_prefix=${log_file%.*}

# Retain timing entries only
#
# If two entries reporting the predictor type occur one after the other,
# the first one is assumed to be an error and is discarded.
grep "Loaded model\|overhead \[sec\]:" "$log_file" \
    | cut -d'|' -f5 \
    | sed -e "{ N; /Loaded model.*\n[[:space:]+]Loaded model/D; }" > "${log_file_prefix}-times.log"

# Group timing entries according to the predictor type
grep -A2 "MLP" "${log_file_prefix}-times.log" | grep "overhead" > "${log_file_prefix}-times-mlp.log"
grep -A2 "RNN" "${log_file_prefix}-times.log" | grep "overhead" > "${log_file_prefix}-times-rnn.log"
## https://unix.stackexchange.com/a/213395/405588
grep -n -A2 "RNN\|MLP" "${log_file_prefix}-times.log" \
    | sed -n 's/^\([0-9]\{1,\}\).*/\1d/p' \
    | sed -f - "${log_file_prefix}-times.log" > "${log_file_prefix}-times-lin.log"

# Convert to csv
for predictor in "${predictor_list[@]}"; do
    (
        echo "total (sec),processing (sec)"
        sed -e "{ s/Total overhead \[sec\]://; N; s/\n Processing overhead \[sec\]: /,/g; s/[[:space:]]//g; }" \
            "${log_file_prefix}-times-${predictor}.log"
    ) > "${log_file_prefix}-times-${predictor}.csv"
done
