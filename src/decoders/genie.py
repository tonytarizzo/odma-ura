"""Genie-support least-squares reference decoder.

This is not a practical decoder. It is handed the *true* support (the set of
transmitted messages) and only has to estimate the amplitudes on that support
by non-negative least squares. It therefore isolates the codebook *geometry*
from the support-recovery difficulty: any gap between a real decoder and this
reference is attributable to detection/algorithmic error, while any gap between
dense and ODMA *under this reference* is attributable purely to how well each
codebook's active columns can be disentangled once the support is known.

Because the support is known exactly, the only errors that remain are
amplitude/count errors from column collinearity within the true support (and,
for ODMA, cross-pattern overlap). It is an oracle baseline and must be labelled
as such in any figure.
"""

from __future__ import annotations

import numpy as np

from ..scenario import Scenario
from .omp import _build_global_dictionary, _matched_filter_y, _nnls_solve, _project_to_integer_total


def run_oracle_support(scenario: Scenario, *, integer_project: bool = True) -> tuple[np.ndarray, dict]:
    """Non-negative LS on the true support.

    ``integer_project`` snaps the amplitudes to integer counts summing to the
    true number of active users (matching the count target); with it disabled
    the raw rounded NNLS amplitudes are returned instead.
    """
    num_codewords = scenario.num_codewords
    y_mf = _matched_filter_y(scenario)
    Phi = _build_global_dictionary(scenario)

    support = np.flatnonzero(scenario.message_counts > 0)
    counts = np.zeros(num_codewords, dtype=np.float64)
    if support.size == 0:
        return counts, {"decoder": "genie_support_ls", "selected_k": 0, "K_target": 0}

    x = _nnls_solve(Phi[:, support], y_mf)
    K = int(round(float(np.sum(scenario.message_counts))))
    if integer_project and K > 0:
        counts[support] = _project_to_integer_total(x, K)
    else:
        counts[support] = np.maximum(0.0, np.round(x))
    return counts, {
        "decoder": "genie_support_ls",
        "selected_k": int(support.size),
        "K_target": K,
        "K_hat": float(np.sum(counts)),
        "oracle": "support",
    }
