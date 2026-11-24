"""Always spend one ordinary wide-list call on the fitted word with the largest rate score."""
from collections import Counter
import time
import numpy as np

from baselines.polar import PolarCode
from framework.polar_count_feedback import support_completion
from framework.polar_full_pattern import CountBank
from framework.polar_rate_challenge import _select


def finish(scenario, output, meta):
    started = time.perf_counter()
    fitted = Counter(dict(meta["cancellation_action"]))
    if not meta["complete"] or fitted.total() > scenario.k:
        raise ValueError("Require a complete prefix with a valid fitted count")
    repair = dict(issued=False)
    if fitted:
        model = scenario.model
        bank = CountBank(scenario, meta["bank"])
        residual = scenario.y - bank.matrix @ bank.vector(fitted)
        variance = scenario.noise_var + (scenario.k-fitted.total())/model.n
        w, score = _select(scenario, bank, fitted, residual, variance)
        p, _ = model.split(w)
        at, waveform = bank.atoms[w]
        llr = 2*model.amplitudes[p]*(residual[at]+waveform)/variance
        code = PolarCode(model.b, model.code.n, model.code.crc_bits, 4096)
        if not np.array_equal(model.code.information, code.information):
            raise ValueError("Incompatible ordinary wide-list input layout")
        payloads, _ = code.decode_list(llr)
        ids = [model.word(p,u) for u in payloads]
        fresh, _ = bank.extend(ids)
        if fresh:
            fitted, log = bank.refit(fitted)
            if not log["converged"]:
                raise RuntimeError("Signed restoration count fit exhausted its move cap")
        output, done = support_completion(scenario, fitted, bank.words)
        repair.update(issued=True, selected=w, score=score, pattern=p, ids=ids)
        meta = {**meta, "bank": bank.words, "cancellation_action": sorted(fitted.items()), "support_completion": done}
    return dict(output), {**meta, "signed_restoration": repair, "restoration_seconds": time.perf_counter()-started}
