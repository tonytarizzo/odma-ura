"""Complete count-growth DEC-35 followed by at most one terminal wide-list query."""

from collections import Counter
import time

from framework.polar_count_growth import run as count_growth
from framework.polar_folded_terminal import terminal_query


def finish(scenario, baseline, original, selector):
    """The immutable cold prefix is common to both selectors; completion counts never enter the new residual."""
    if selector not in ("magnitude", "folded"):
        raise ValueError("Require the magnitude or folded terminal selector")
    started = time.perf_counter()
    fitted = Counter(dict(original["cancellation_action"]))
    counts, trace = baseline, None
    meta = {**original, "terminal_selector": selector}
    if original["complete"] and scenario.k - fitted.total() == 1:
        counts, trace = terminal_query(scenario, fitted, original["bank"], selector)
        meta.update(bank=sorted(set(original["bank"]).union(trace["new_ids"])), cancellation_action=trace["cancellation"],
                    support_completion=trace["completion"], complete=trace["fit"] is None or trace["fit"]["converged"])
    meta.update(terminal_query=trace, terminal_seconds=time.perf_counter()-started)
    return dict(counts), meta


def run(scenario, *, selector):
    if selector not in ("magnitude", "folded"):
        raise ValueError("Require the magnitude or folded terminal selector")
    started = time.perf_counter()
    counts, meta = count_growth(scenario)
    output, combined = finish(scenario, counts, meta, selector)
    combined["total_seconds"] = time.perf_counter()-started
    return output, combined
