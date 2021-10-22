"""Costant values"""

from pathlib import Path

DATA_ROOT = Path("../data/")
DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"

STATIC_RUNS = {
    58: {
        "load_profile": "test_behavior_02_distwalk-6t_last100.dat",
        "start_real": "2021-09-08T11:10:41Z",
        "end_real": "2021-09-08T12:54:57Z",
        "vm_delay_min": 5.5,
    },
}

RNN_RUNS = {
    32: {
        "load_profile": "test_behavior_02_distwalk-6t_last100.dat",
        "start_real": "2021-10-16T09:51:18Z",
        "end_real": "2021-10-16T11:33:22Z",
        "model": "rnn-20_sum_2021-07-22.pt",
        "scaler": "rnn_scaler.joblib",
        "input_size": 20,
        "vm_delay_min": 5.5,
    },
}

LIN_RUNS = {
    4: {
        "load_profile": "test_behavior_02_distwalk-6t_last100.dat",
        "start_real": "2021-09-09T11:05:01Z",
        "end_real": "2021-09-09T12:47:29Z",
        "scaler": "scaler.joblib",
        "input_size": 20,
        "vm_delay_min": 5.5,
    },
}

MLP_RUNS = {
    2: {
        "load_profile": "test_behavior_02_distwalk-6t_last100.dat",
        "start_real": "2021-09-08T07:20:03Z",
        "end_real": "2021-09-08T09:02:04Z",
        "model": "mlp-20_sum_2021-07-20.pt",
        "scaler": "scaler.joblib",
        "input_size": 20,
        "vm_delay_min": 5.5,
    },
}
