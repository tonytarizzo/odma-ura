"""DEC-38 execution option with rate-zero/repetition SCL in native, rollback and count growth."""
import copy
from importlib import import_module
import time

import numpy as np

from baselines.polar import PolarCode
from framework.polar_conditional_serial import native
from framework.polar_count_feedback import continue_feedback, support_completion
from framework.polar_repetition_scl import RepetitionPolarCode
from framework.polar_serial_repair import finish as scan_rate
from framework.polar_tractable import PolarScenario
from framework.polar_two_raw import finish as two_raw

# Job 037's published rollback code uses unqualified sibling imports and provides their search path at import time.
import_module("jobs.037_polar_tractable.rollback_rescue")
from rollback_progressive import progressive


def prefix(scenario):
    started = time.perf_counter()
    if not isinstance(scenario, PolarScenario) or type(scenario.model.code) is not PolarCode or scenario.sampling != "iid":
        raise ValueError("Require the original procedural polar encoder and IID messages")
    original = scenario.model.code
    if original.list_size != 128 or scenario.model.p < 32:
        raise ValueError("Require native width 128 and at least 32 placements for rollback")
    model = copy.copy(scenario.model)
    model.code = RepetitionPolarCode(original.payload_bits, original.n, original.crc_bits, 128)
    if not np.array_equal(model.code.information, original.information) or not np.array_equal(model.code.frozen, original.frozen):
        raise ValueError("Incompatible native polar layout")
    working = PolarScenario(model, scenario.y, scenario.k, scenario.noise_var, scenario.sampling)
    accepted, native_meta = native(working)
    initial, fresh, batches, seconds = progressive(working, list(accepted), native_meta["valid_ids"], "ungated",
                                                  code_factory=RepetitionPolarCode)
    base_bank = set(native_meta["valid_ids"]).union(fresh, initial)
    fitted, bank, trace = continue_feedback(working, initial, base_bank, code_factory=RepetitionPolarCode, memoize=True,
                                            progress_extra_sweep=True, count_growth=True)
    output, completion = support_completion(working, fitted, bank)
    meta = dict(native=native_meta, native_accepted=list(accepted), native_demapper="conditional_serial", rollback_seconds=seconds,
                rollback_batches=batches, rollback_action=sorted(initial.items()), base_bank=sorted(base_bank), bank=sorted(bank),
                cancellation_action=sorted(fitted.items()), continuation=trace, support_completion=completion, complete_support=True,
                complete=trace["complete"], execution="repetition_tail", total_seconds=time.perf_counter() - started)
    return dict(output), meta


def run(scenario, *, terminal_policy="scan_rate"):
    """Run with the catalogue terminal policy and the original provider for each wide terminal query."""
    if terminal_policy not in ("scan_rate", "two_raw"):
        raise ValueError("DEC-38 terminal_policy must be scan_rate or two_raw")
    started = time.perf_counter()
    output, meta = prefix(scenario)
    finish = two_raw if terminal_policy == "two_raw" else scan_rate
    output, meta = finish(scenario, output, meta)
    return output, {**meta, "total_seconds": time.perf_counter() - started}
