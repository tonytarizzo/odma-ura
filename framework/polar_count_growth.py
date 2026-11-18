"""Native-only repetition with bounded strict cancellation-count-growth continuation."""

import copy
import time

import numpy as np

from baselines.polar import PolarCode
from framework.polar_count_feedback import run as count_feedback
from framework.polar_repetition_scl import RepetitionPolarCode
from framework.polar_tractable import PolarScenario


def run(scenario):
    started = time.perf_counter()
    if not isinstance(scenario, PolarScenario) or type(scenario.model.code) is not PolarCode:
        raise ValueError("Count-growth continuation requires the original PolarCode in a procedural PolarScenario")
    original = scenario.model.code
    if original.list_size != 128 or scenario.model.p < 32:
        raise ValueError("Require the native SCL128 configuration and at least32 patterns")
    replacement = RepetitionPolarCode(original.payload_bits, original.n, original.crc_bits, original.list_size)
    if not np.array_equal(replacement.information, original.information) or not np.array_equal(replacement.frozen, original.frozen):
        raise ValueError("Original code layout differs from the declared encoder")
    model = copy.copy(scenario.model)
    model.code = replacement
    copied = PolarScenario(model, scenario.y, scenario.k, scenario.noise_var, scenario.sampling)
    preparation = time.perf_counter() - started
    counts, meta = count_feedback(copied, memoize=True, complete_support=True, progress_extra_sweep=True, count_growth=True)
    return counts, {**meta, "variant": "native_repetition_count_growth", "native_repetition_preparation_seconds": preparation,
                    "underlying_receiver_seconds": meta["total_seconds"], "total_seconds": time.perf_counter() - started,
                    "provider_scope": "Only native SCL128 uses repetition compression. Rollback512 and continuation128 "
                                      "retain original PolarCode. Strict endpoint cancellation-count growth after block4 "
                                      "permits further fixed queues; the same bank/cache persist. Completion is terminal."}
