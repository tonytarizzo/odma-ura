"""Complete polar receiver catalogue, including the byte-frozen DEC-01--35 registry."""
from functools import partial
import time

from framework.polar_conditional_serial import prefix as tagged_prefix, run as tagged_two_raw
from framework.polar_serial_repair import finish as scan_then_challenge, prefix as serial_prefix, run_held
from framework.polar_constituent_challenge import finish as constituent_challenge
from framework.polar_terminal_scan import finish as deficit_scan
from framework.polar_decoders import DecoderSpec, REGISTRY as HISTORICAL, run as run_historical
from framework.polar_held_interference import run as held_resource
from framework.polar_rate_challenge import run as rate_challenge
from framework.polar_terminal_query import run as terminal_query
from framework.polar_terminal_scan import run as terminal_scan
from framework.polar_two_raw import run as two_raw


def tagged_repair(scenario, *, terminal_policy="scan_rate"):
    """Condition native SIC on unknown placement evidence and on each accepted physical waveform."""
    if terminal_policy == "two_raw":
        return tagged_two_raw(scenario)
    if terminal_policy != "scan_rate":
        raise ValueError("DEC-38 terminal_policy must be scan_rate or two_raw")
    started = time.perf_counter()
    counts, meta = tagged_prefix(scenario)
    counts, meta = scan_then_challenge(scenario, counts, meta)
    return counts, {**meta, "total_seconds": time.perf_counter() - started}


def joined_repair(scenario):
    """Use the dimension-203 alternative law to allocate one occupied challenge after serial discovery."""
    model = scenario.model
    if (model.b, model.code.n, model.code.crc_bits) != (87, 512, 16):
        raise ValueError("DEC-39 requires the validated 87-payload-bit CRC16/polar512 geometry")
    started = time.perf_counter()
    counts, meta = serial_prefix(scenario)
    counts, meta = deficit_scan(scenario, counts, meta)
    counts, meta = constituent_challenge(scenario, counts, meta)
    return counts, {**meta, "native_demapper": "serial_held_resource", "total_seconds": time.perf_counter() - started}


REGISTRY = {**HISTORICAL,
            "DEC-36": DecoderSpec("DEC-36: placement-weighted native SCL with count repair", held_resource, ("iid",), partial_counts=True),
            "DEC-37": DecoderSpec("DEC-37: rate-based occupied-word repair", rate_challenge, ("iid",), partial_counts=True),
            "DEC-38": DecoderSpec("DEC-38: tag-conditioned serial SCL with count repair", tagged_repair, ("iid",), partial_counts=True),
            "DEC-39": DecoderSpec("DEC-39: serial SCL with a polar-constituent occupied challenge", joined_repair, ("iid",), partial_counts=True)}
TERMINAL_POLICIES = {"magnitude": partial(terminal_query, selector="magnitude"), "scan": terminal_scan, "two_raw": two_raw}


def run(name, scenario, **params):
    if name == "DEC-35" and "terminal_policy" in params:
        policy = params.pop("terminal_policy")
        if params or policy not in TERMINAL_POLICIES:
            raise ValueError("Select one complete DEC-35 terminal policy with its fixed count-growth configuration")
        receiver = TERMINAL_POLICIES[policy]
    elif name == "DEC-36" and "terminal_policy" in params:
        policy = params.pop("terminal_policy")
        if params or policy != "scan_rate":
            raise ValueError("Select scan_rate for the combined DEC-36 terminal policy")
        receiver = run_held
    elif name in HISTORICAL:
        return run_historical(name, scenario, **params)
    else:
        receiver = REGISTRY[name].run
    counts, meta = receiver(scenario, **params)
    if any(type(w) is not int or not 0 <= w < 1 << scenario.model.B or type(c) is not int or c <= 0
           for w, c in counts.items()) or sum(counts.values()) > scenario.k:
        raise ValueError(f"{name} violated the sparse count/output-budget contract")
    return counts, {**meta, "catalogue_id": name, "legacy_dense_adapter": False}
