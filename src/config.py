"""Project-wide defaults: base scenario, sweep specs, paths."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR  = PROJECT_ROOT / "results"
CACHE_PATH   = RESULTS_DIR / "cache.jsonl"
PLOTS_DIR    = RESULTS_DIR / "plots"
SINGLE_DIR   = RESULTS_DIR / "single"


BASE_SCENARIO: dict = {
    "n": 128,
    "d": 16,
    "num_blocks": 8,
    "num_codewords": 64,
    "num_devices_active": 10,
    "num_antennas": 4,
    "esn0_db": 10.0,
}


SWEEP_CONFIGS: dict[str, dict] = {
    "K": {
        "param": "num_devices_active",
        "values": [5, 10, 20, 30, 40, 50, 60],
        "label": "Active devices K",
    },
    "SNR": {
        "param": "esn0_db",
        "values": [-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0],
        "label": "Es/N0 (dB)",
    },
    "antennas": {
        "param": "num_antennas",
        "values": [1, 2, 4, 8],
        "label": "Receive antennas M_ant",
    },
    "blocks": {
        "param": "num_blocks",
        "values": [4, 8, 12, 16],
        "label": "ODMA blocks B",
    },
    "codewords": {
        "param": "num_codewords",
        "values": [32, 64, 128],
        "label": "Total codewords M",
    },
}
