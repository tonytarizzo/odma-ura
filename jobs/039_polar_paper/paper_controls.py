"""Paid-query and elapsed-time controls for the flat B100 count-growth paper panel."""

from collections import Counter
import copy
import hashlib
import math
from pathlib import Path
import sys
from time import perf_counter
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "jobs/037_polar_tractable"))
from baselines.polar import PolarCode
from framework.polar_count_cavity import action_hash, cavity_query
from framework.polar_count_feedback import PayloadMemo, support_completion
from framework.polar_decoders import run as registered
from framework.polar_full_pattern import CountBank
from framework.polar_repetition_scl import RepetitionPolarCode
from framework.polar_tractable import PolarScenario, Work
import rollback_frames
import rollback_progressive
from paper_curve import make_frame, make_geometry, score as count_metrics

SEEDS = tuple(range(933001, 933129))
ENVELOPE = 210
OPTIONS = dict(native_repetition=True, memoize=True, complete_support=True, progress_extra_sweep=True, count_growth=True)
RULES = ("conventional", "add_only", "reversible")


class AddOnlyBank(CountBank):
    """The original count objective and add tie ordering; drop and swap are unavailable."""

    def gains(self, counts, residual, gram_cache):
        h = self.scenario.k - int(counts.sum())
        best = (0., "none", -1, -1)
        if not h or not len(counts):
            return best, {"candidate_gain_evaluations": 0, "gram_rows_constructed": 0}
        variance = self.scenario.noise_var + h * self.kappa
        smaller = self.scenario.noise_var + (h - 1) * self.kappa
        square = float(residual @ residual)
        dots = np.asarray(self.matrix.T @ residual).ravel()
        centered_dots = dots - float(residual @ self.mean)
        common = -.5 * self.model.n * math.log1p(-self.kappa / variance)
        common -= .5 * square * self.kappa / (variance * smaller)
        values = common + (centered_dots - .5 * self.centered_norms) / smaller
        values += math.log(h) - np.log(counts + 1) - self.log_outside
        j = int(np.argmax(values))
        if values[j] > best[0]:
            best = (float(values[j]), "add", -1, j)
        return best, {"candidate_gain_evaluations": len(counts), "gram_rows_constructed": 0}


def raw(bank, action):
    return bank.scenario.y - bank.matrix @ bank.vector(action)


def fit(bank, action):
    next_action, meta = bank.refit(action)
    if not meta["converged"]:
        raise ArithmeticError("Count refit failed its declared move cap")
    return next_action, meta


def prefix(scenario, ref):
    """Record actual native/rollback input bytes on cold common initialization, before any continuation."""
    started = perf_counter()
    history = {128:set(), 512:set()}
    model = copy.copy(scenario.model)
    original = model.code
    active = [None]
    recording_cost = [0.]
    calls = Counter()
    chronology = []

    def remember(width, llr):
        tick = perf_counter()
        key = llr.tobytes()
        history[width].add((int(active[0]), key))
        chronology.append((width,int(active[0]),hashlib.sha256(key).hexdigest()))
        calls[width] += 1
        recording_cost[0] += perf_counter()-tick

    class ObservedWork(Work):
        def patterns(self, residual, num):
            for pattern in super().patterns(residual,num):
                active[0] = int(pattern)
                yield pattern

    class ObservedNative(RepetitionPolarCode):
        def decode_list(self, llr):
            remember(128,llr)
            return super().decode_list(llr)

    class ObservedRollback(PolarCode):
        def decode_list(self, llr):
            remember(512,llr)
            return super().decode_list(llr)

    ordinary_list = rollback_progressive.polar_list

    def polar_list(cmodel, code, pattern, residual, variance):
        active[0] = int(pattern)
        return ordinary_list(cmodel,code,pattern,residual,variance)

    model.code = ObservedNative(model.b, original.n, original.crc_bits, original.list_size)
    copied = PolarScenario(model, scenario.y, scenario.k, scenario.noise_var, scenario.sampling)
    with patch.object(rollback_frames, "Work", ObservedWork):
        accepted, native = rollback_frames.native(copied)
    with patch.object(rollback_progressive, "polar_list", polar_list):
        action, found, batches, rollback_s = rollback_progressive.progressive(
            copied, list(accepted), native["valid_ids"], "ungated", code_factory=ObservedRollback)
    ids = set(native["valid_ids"]).union(found,action)
    elapsed = perf_counter()-started
    if (dict(action) != dict(ref["rollback_action"]) or ids != set(ref["base_bank"]) or batches != ref["rollback_batches"]
            or any(native[k] != ref["native"][k] for k in ("calls","valid_ids","rounds","residual_sha256"))):
        raise AssertionError("Cold common prefix differs from registered DEC-35")
    if calls[128] != native["calls"] or calls[512] != sum(len(b["patterns"]) for b in batches):
        raise AssertionError("Missing actual prefix queries")
    work = dict(seconds=elapsed, native_seconds=native["seconds"], rollback_seconds=rollback_s, record_input_seconds=recording_cost[0],
                calls128=calls[128], calls512=calls[512], distinct128=len(history[128]), distinct512=len(history[512]),
                cached_input_bytes=sum(len(bits) for bucket in history.values() for _,bits in bucket), attempted=chronology)
    return action, ids, history, work


def endpoint(scenario, state, words, counts, elapsed, geometry, prefix_s):
    tick = perf_counter()
    decision, meta = support_completion(scenario, state, words)
    end_seconds = perf_counter()-tick
    return dict(action=sorted(decision.items()), cancellation=sorted(state.items()), bank=sorted(words), completion=meta,
                work=dict(counts), search_seconds=elapsed, completion_seconds=end_seconds,
                receiver_seconds=prefix_s+elapsed+end_seconds, geometry_inclusive_seconds=geometry+prefix_s+elapsed+end_seconds)


def search(s, initial, known, history, prefix_s, geometry, rule, budget, target=None, reference_calls=None):
    """The equal-call policy immediately reranks after count changes and visits each physical context at most once."""
    if rule not in RULES or budget < 1 or (target is not None and rule != "conventional"):
        raise ValueError("Invalid extension")
    started = perf_counter()
    factory = AddOnlyBank if rule == "add_only" else CountBank
    conventional = rule == "conventional"
    bank = None if conventional else factory(s, known)
    words, atoms = set(known), {}
    action = initial.copy()
    work = Counter(actual_scl_calls=0, prior128_skips=0, prior512_requeries=0, own_context_skips=0, cache_hits=0,
                   top_already_accepted_other_unreported=0,
                   ranked_contexts=0, rerankings=0, refits=0, add_moves=0, drop_moves=0, swap_moves=0,
                   candidate_gain_evaluations=0, gram_rows_constructed=0)
    fits_s = 0.

    def refit():
        nonlocal action, fits_s
        tick = perf_counter()
        action, meta = fit(bank,action)
        fits_s += perf_counter()-tick
        work["refits"] += 1
        for move in meta["moves"]:
            work[move["kind"]+"_moves"] += 1
        for key in ("candidate_gain_evaluations","gram_rows_constructed"):
            work[key] += meta[key]

    if not conventional:
        refit()
    model = s.model
    code = PayloadMemo(PolarCode(model.b,model.code.n,model.code.crc_bits,128))
    visited, logs, skips = set(), [], []
    rank, residual, offset = None, None, 0
    endpoints, bracket = {}, {"under":None,"over":None}
    offline = 0.
    exhausted = False
    score_s = llr_s = 0.

    def snapshot():
        nonlocal offline
        # Each endpoint represents a receiver that performs one completion. Intermediate
        # completion probes are experimental readouts and never update the search state.
        now = perf_counter()
        end = endpoint(s,action,words,work,now-started-offline,geometry,prefix_s)
        offline += perf_counter()-now
        return end

    def record():
        n = code.calls
        requested = n == budget or n == reference_calls
        matched = target is not None and bracket["over"] is None
        if requested or matched:
            end = snapshot()
            if requested and str(n) not in endpoints:
                endpoints[str(n)] = end
            if matched:
                if end["receiver_seconds"] <= target:
                    bracket["under"] = end
                else:
                    bracket["over"] = end

    record()
    while (code.calls < budget or (reference_calls is not None and code.calls < reference_calls)
           or (target is not None and bracket["over"] is None)):
        if rank is None:
            tick = perf_counter()
            if conventional:
                # Accumulate only cancelled IDs in ascending order, as in the sparse matrix
                # product of the fitted arms. Hard SIC needs no new-ID CountBank construction.
                predicted = np.zeros_like(s.y)
                for word,count in sorted(action.items()):
                    if word not in atoms:
                        branch, values = model.symbols(word)
                        atoms[word] = model.positions[branch],values
                    ix,values = atoms[word]
                    predicted[ix] += count*values
                residual = s.y-predicted
            else:
                residual = raw(bank,action)
            rank = np.argsort(-np.abs(residual[model.positions]).sum(1),kind="stable")
            offset = 0
            work["rerankings"] += 1
            score_s += perf_counter()-tick
        if offset == len(rank):
            exhausted = True
            break
        tick = perf_counter()
        pattern = int(rank[offset])
        offset += 1
        if conventional:
            pending = s.k-action.total()
            variance = s.noise_var + max(pending-1,0)/model.n
            llr = 2*model.amplitudes[pattern]*residual[model.positions[pattern]]/variance
        else:
            llr, context = cavity_query(s,action,residual,pattern)
            pending = context["pending"]
        payload_key = llr.tobytes()
        key = (pattern,payload_key)
        work["ranked_contexts"] += 1
        llr_s += perf_counter()-tick
        if key in history[128]:
            reason = "prior128_skips"
            work[reason] += 1
            skips.append((reason,pattern,hashlib.sha256(payload_key).hexdigest()))
            continue
        if key in visited:
            work["own_context_skips"] += 1
            skips.append(("own_context_skips",pattern,hashlib.sha256(payload_key).hexdigest()))
            continue
        visited.add(key)
        cross_width = key in history[512]
        work["prior512_requeries"] += int(cross_width)
        prior_action = action.copy()
        payloads,_ = code.decode_list(llr)
        ids = [model.word(pattern,w) for w in payloads]
        if conventional:
            fresh = sorted(set(ids)-words)
            if ids and action[ids[0]] and any(not action[w] for w in ids[1:]):
                work["top_already_accepted_other_unreported"] += 1
            if ids and not action[ids[0]] and action.total() < s.k:
                action[ids[0]] += 1
                work["add_moves"] += 1
        else:
            fresh,_ = bank.extend(ids)
            if fresh:
                refit()
        words.update(fresh)
        if action != prior_action:
            rank = None
        work["actual_scl_calls"], work["cache_hits"] = code.calls, code.hits
        logs.append(dict(p=pattern, llr=hashlib.sha256(payload_key).hexdigest(), pending=pending, cache=code.last_hit,
                         tried_at_512=cross_width, ids=ids, fresh=fresh, changed=bool(action!=prior_action)))
        if not code.last_hit:
            record()
    end = snapshot()
    elapsed = perf_counter()-started
    return dict(fixed=endpoints.get(str(budget)), matched_registered_calls=endpoints.get(str(reference_calls)), bracket=bracket,
                exhausted=exhausted, last=end, log=logs, skipped=skips, analysis_endpoint_seconds=offline, actual_study_extension_seconds=elapsed,
                refit_seconds=fits_s, ranking_seconds=score_s, llr_seconds=llr_s, cache=code.stats())


def replay(s, ref, reference_action, labels):
    """Intervene only on the numerator along the exogenous registered query/state schedule."""
    started = perf_counter()
    model = s.model
    base = set(ref["base_bank"])
    cont = ref["continuation"]
    current = Counter(dict(cont["initial_refit_action"]))
    previous = Counter(dict(ref["rollback_action"]))
    move_kinds = sorted({m["kind"] for m in cont["initial_refit"]["moves"]})
    corrections = {r["query"]:r for r in cont["count_action_path"]}
    atoms = CountBank(s,ref["bank"])
    live_raw, frozen_raw = raw(atoms,current), raw(atoms,previous)
    modes = ("live","frozen")
    decoders, banks, actions = {}, {}, {}
    for key in modes:
        decoders[key] = PayloadMemo(PolarCode(model.b,model.code.n,model.code.crc_bits,128))
        banks[key] = CountBank(s,base)
        actions[key],_ = fit(banks[key],previous)
    truth = set(labels)
    observations, prior_discovery = [], {mode:{} for mode in modes}
    for record in cont["queries"]:
        i,p = record["query"], record["pattern"]
        if i != len(observations) or record["pre_action_sha256"] != action_hash(current):
            raise AssertionError("Registered physical chronology changed")
        changed = current != previous
        found, details, hashes = {}, {}, {}
        for mode, residual in (("live",live_raw), ("frozen",frozen_raw)):
            llr, info = cavity_query(s,current,residual,p)
            hashes[mode] = hashlib.sha256(llr.tobytes()).hexdigest()
            if info["variance"] != record["variance"]:
                raise AssertionError("Replay changed query variance")
            payloads,_ = decoders[mode].decode_list(llr)
            found[mode] = [model.word(p,w) for w in payloads]
            new,_ = banks[mode].extend(found[mode])
            if new:
                actions[mode],_ = fit(banks[mode],actions[mode])
            for w in new:
                prior_discovery[mode][w] = i
            details[mode] = dict(ids=found[mode], fresh=new, cache_hit=decoders[mode].last_hit, llr=hashes[mode])
        if (found["live"] != record["ids"] or hashes["live"] != record["llr_sha256"]
                or details["live"]["fresh"] != record["new_ids"]):
            raise AssertionError("Live replay must reproduce every actual list and discovery")
        if not changed and (hashes["live"] != hashes["frozen"] or found["live"] != found["frozen"]):
            raise AssertionError("Equal count states yielded different observations")
        observations.append(dict(query=i, sweep=record["sweep"], pattern=p, variance=info["variance"], eligible=changed,
            changed_input=hashes["live"]!=hashes["frozen"], live_state=action_hash(current), frozen_state=action_hash(previous),
            preceding_refit_move_kinds=move_kinds, **details))
        if i in corrections:
            step = corrections[i]
            if current != Counter(dict(step["before"])):
                raise AssertionError("Wrong physical count predecessor")
            previous,current = current,Counter(dict(step["after"]))
            move_kinds = sorted({m["kind"] for m in cont["refits"][step["refit_index"]]["meta"]["moves"]})
            frozen_raw,live_raw = live_raw,raw(atoms,current)
        if current != actions["live"]:
            raise AssertionError("Live refit action differs from the reported path")
    outputs = {}
    for mode in modes:
        accepted,completion = support_completion(s,actions[mode],banks[mode].words)
        outputs[mode] = dict(action=sorted(accepted.items()), bank=banks[mode].words, cancellation=sorted(actions[mode].items()),
            completion=completion, metrics=count_metrics(labels,accepted), new_true=sorted(set(prior_discovery[mode])&truth),
            new_false=sorted(set(prior_discovery[mode])-truth), retained_new_true=sorted(set(prior_discovery[mode])&truth&set(accepted)),
            searches=decoders[mode].stats())
    if dict(outputs["live"]["action"]) != dict(reference_action) or set(outputs["live"]["bank"]) != set(ref["bank"]):
        raise AssertionError("Live complete readout differs from the registered receiver")
    return dict(queries=observations, arms=outputs, opportunities=sum(q["eligible"] for q in observations),
                changed_inputs=sum(q["changed_input"] for q in observations),
                live_only_true_ids=sorted(set(outputs["live"]["new_true"])-set(outputs["frozen"]["new_true"])),
                frozen_only_true_ids=sorted(set(outputs["frozen"]["new_true"])-set(outputs["live"]["new_true"])),
                seconds=perf_counter()-started)


def add_scores(row, labels):
    if row is not None:
        row["metrics"] = count_metrics(labels,Counter(dict(row["action"])))


def validate_row(r):
    """Validate stored logical/paid work, matching, count metrics and every registered replay list."""
    labels, ref, prefix_meta = r["labels"], r["registered"], r["prefix"]
    if (len(labels) != 100 or r["source"]["repeated_messages"] != 100-len(set(labels))
            or r["source"]["repeated_patterns"] != 100-len({w>>87 for w in labels})):
        raise ValueError("Invalid source multiplicity profile")
    def require(condition, message):
        if not condition:
            raise ValueError(message)
    def verify_action(view):
        pairs = view["action"]
        require(len(dict(pairs))==len(pairs), "Duplicate sparse action rows")
        require(count_metrics(labels,Counter(dict(pairs)))==view["metrics"], "Wrong membership or multiplicity metrics")
    verify_action(ref)
    require(math.isclose(r["source"]["variance"],1/(200*10**.05)) and all(len(r["source"][key])==64 for key in
            ("y_sha256","positions_sha256","amplitudes_sha256","standard_noise_sha256","bits_sha256","information_sha256")),
            "Wrong flat B100 input contract")
    require(ref["meta"]["dec35_options"]==OPTIONS and ref["meta"]["complete"], "Wrong registered receiver")
    pre = {j:{(p,h) for width,p,h in prefix_meta["attempted"] if width==j} for j in (128,512)}
    for j in (128,512):
        require(len(pre[j])==prefix_meta[f"distinct{j}"] and sum(w==j for w,_,_ in prefix_meta["attempted"])==prefix_meta[f"calls{j}"],
                "Missing common-prefix query")
    require((prefix_meta["calls128"],prefix_meta["calls512"])==(ref["native_calls"],ref["rollback_calls"]) and ref["rollback_calls"]==32,
            "Prefix call count differs from the registered receiver")
    for rule, arm in r["controls"].items():
        seen, inputs, n = set(), set(), 0
        union = set(ref["meta"]["base_bank"])
        for q in arm["log"]:
            key=(q["p"],q["llr"])
            require(key not in pre[128] and key not in seen, "Attempted duplicate physical SCL128 context")
            require(q["tried_at_512"]==(key in pre[512]), "Wrong prior-512 context classification")
            require(q["cache"]==(q["llr"] in inputs), "Wrong cross-pattern payload cache hit")
            require(all(w>>87==q["p"] for w in q["ids"]), "Invalid full pattern/payload association")
            require(q["fresh"]==sorted(set(q["ids"])-union), "Discovery list does not match previous bank")
            union.update(q["ids"])
            seen.add(key)
            inputs.add(q["llr"])
            n+=int(not q["cache"])
        for reason,p,h in arm["skipped"]:
            candidates={"prior128_skips":pre[128],"own_context_skips":seen}
            require((p,h) in candidates[reason], "Unsubstantiated skipped context")
        last=arm["last"]["work"]
        require(last["actual_scl_calls"]==n==arm["cache"]["actual_scl_calls"] and last["ranked_contexts"]==len(arm["log"])+len(arm["skipped"]),
                "Logical/paid search counters disagree")
        require(last["prior512_requeries"]==sum(q["tried_at_512"] for q in arm["log"]), "Wrong cross-width requery counter")
        for reason in ("prior128_skips","own_context_skips"):
            require(last[reason]==sum(z[0]==reason for z in arm["skipped"]), "Wrong skip counter")
        views = [arm["last"],arm["fixed"],arm["matched_registered_calls"],*arm["bracket"].values()]
        for view in (v for v in views if v is not None):
            verify_action(view)
            logical = view["work"]["actual_scl_calls"]+view["work"]["cache_hits"]
            prefix = arm["log"][:logical]
            require(logical==len(prefix) and view["work"]["actual_scl_calls"]==sum(not q["cache"] for q in prefix)
                    and set(view["bank"])==set(ref["meta"]["base_bank"]).union(*(q["ids"] for q in prefix)),
                    "Endpoint bank or paid/logical envelope differs from the query chronology")
            require(set(dict(view["action"])).issubset(view["bank"]) and set(dict(view["cancellation"])).issubset(view["bank"]),
                    "Endpoint emits a complete ID outside the discovered bank")
            require(math.isclose(view["receiver_seconds"],prefix_meta["seconds"]+view["search_seconds"]+view["completion_seconds"]),
                    "Incorrect receiver time")
            require(math.isclose(view["geometry_inclusive_seconds"],r["geometry_seconds"]+view["receiver_seconds"]),"Geometry charge missing")
            require(0 <= view["work"]["actual_scl_calls"] <= n, "Invalid prefix endpoint")
        require(arm["fixed"] is None or arm["fixed"]["work"]["actual_scl_calls"]==ENVELOPE, "Wrong fixed envelope")
        if rule in ("add_only","conventional"):
            require(last["drop_moves"]==last["swap_moves"]==0, "Forbidden reversible move")
        if rule=="conventional":
            lo,hi=arm["bracket"]["under"],arm["bracket"]["over"]
            if lo and hi:
                require(hi["work"]["actual_scl_calls"]==lo["work"]["actual_scl_calls"]+1, "Wall endpoints are not adjacent paid calls")
            require(lo is None or lo["receiver_seconds"]<=ref["receiver_seconds"],"Over-budget endpoint labeled under")
            require(hi is None or hi["receiver_seconds"]>ref["receiver_seconds"],"Under-budget endpoint labeled over")
    evidence=r["replay"]
    original=ref["meta"]["continuation"]["queries"]
    require(len(original)==len(evidence["queries"]), "Missing registered query opportunity")
    for actual,q in zip(original,evidence["queries"],strict=True):
        require((actual["query"],actual["pattern"],actual["llr_sha256"],actual["ids"],actual["variance"])==
                (q["query"],q["pattern"],q["live"]["llr"],q["live"]["ids"],q["variance"]), "Wrong factual replay")
        require(q["eligible"]==(q["live_state"]!=q["frozen_state"]) and q["changed_input"]==(q["live"]["llr"]!=q["frozen"]["llr"]),
                "Wrong matched residual classification")
    require(evidence["opportunities"]==sum(q["eligible"] for q in evidence["queries"]), "Missing eligible numerator change")
    require(dict(evidence["arms"]["live"]["action"])==dict(ref["action"]), "Factual replay changed terminal output")
    for mode in ("live","frozen"):
        a=evidence["arms"][mode]
        verify_action(a)
        new=set(a["bank"])-set(ref["meta"]["base_bank"])
        require(new=={w for q in evidence["queries"] for w in q[mode]["fresh"]} and set(a["new_true"])==new.intersection(labels)
                and set(a["retained_new_true"])==set(a["new_true"]).intersection(dict(a["action"])), "Discovery/readout attribution mismatch")


def run_control_frame(seed):
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("Require one held-out paper-control seed 933001 through 933128")
    study = perf_counter()
    model,geometry = make_geometry(100,seed)
    tick = perf_counter()
    s,labels,source = make_frame(model,100,seed,50)
    source_seconds = perf_counter()-tick
    row = compare_scenario(s,labels,source,seed,geometry,source_seconds)
    row["study_seconds"] = perf_counter()-study
    return row


def compare_scenario(s, labels, source, seed, geometry, source_seconds):
    """The same complete study is callable on the already consumed flat B100 pilot for verification."""
    tick = perf_counter()
    ref_action,ref = registered("DEC-35",s,**OPTIONS)
    ref_wall = perf_counter()-tick
    if not ref["complete"]:
        raise ArithmeticError("Registered receiver is incomplete")
    initial,bank,history,prep = prefix(s,ref)
    calls = ref["continuation"]["actual_scl_calls"]
    order = RULES[(seed-SEEDS[0])%3:] + RULES[:(seed-SEEDS[0])%3]
    arms = {}
    for rule in order:
        conventional = rule=="conventional"
        a = search(s,initial,bank,history,prep["seconds"],geometry,rule,ENVELOPE,ref_wall if conventional else None,
                   calls if conventional else None)
        for view in (a["fixed"],a["matched_registered_calls"],a["last"],*a["bracket"].values()):
            add_scores(view,labels)
        arms[rule] = a
    observed = replay(s,ref,ref_action,labels)
    reference = dict(action=sorted(ref_action.items()), metrics=count_metrics(labels,Counter(ref_action)), native_calls=ref["native"]["calls"],
        rollback_calls=sum(len(q["patterns"]) for q in ref["rollback_batches"]), continuation_calls=calls, meta=ref,
        receiver_seconds=ref_wall, geometry_inclusive_seconds=ref_wall+geometry)
    row = dict(seed=seed, frame=0, k=100, snr_cdb=50, mapping="flat", geometry_seconds=geometry, source_seconds=source_seconds,
               source=source, labels=labels, order=order, prefix=prep, registered=reference, controls=arms, replay=observed)
    validate_row(row)
    return row


def summarize_controls(rows):
    rows = sorted(rows,key=lambda r:r["seed"])
    if [r["seed"] for r in rows] != list(SEEDS):
        raise ValueError("Require all 128 unique held-out frames")
    if any((r["frame"],r["k"],r["snr_cdb"],r["mapping"]) != (0,100,50,"flat") for r in rows):
        raise ValueError("Mismatched held-out contract")
    for r in rows:
        validate_row(r)
    if len({r["source"]["y_sha256"] for r in rows})!=len(rows):
        raise ValueError("Repeated physical observation on distinct purported held-out frames")
    from scipy.stats import binomtest

    def sum_metrics(arms):
        return {key:sum(a["metrics"][key] for a in arms) for key in ("count_misses","false_copies","list_misses","output_copies")}

    def paired(a,b):
        d = np.array([u["metrics"]["count_misses"]-v["metrics"]["count_misses"] for u,v in zip(a,b,strict=True)])
        wins,losses = int((d>0).sum()),int((d<0).sum())
        boot = np.random.default_rng(933001).choice(d,(10000,len(d)),replace=True).mean(1)/100 if len(d) else None
        return dict(frames=len(d), count_savings=int(d.sum()), improved=wins, tied=int((d==0).sum()), worse=losses,
                    count_savings_after_removing_best=int(d.sum()-max(0,d.max())) if len(d) else None,
                    paired_bootstrap_pupe_difference_95=np.quantile(boot,[.025,.975]).tolist() if len(d) else None,
                    paired_sign_p=binomtest(wins,wins+losses).pvalue if wins+losses else 1.)

    fixed_ok = [r for r in rows if all(r["controls"][k]["fixed"] is not None for k in RULES)]
    def stats(arms):
        return dict(totals=sum_metrics(arms), geometry_inclusive_seconds=sum(a["geometry_inclusive_seconds"] for a in arms),
                    receiver_seconds=sum(a["receiver_seconds"] for a in arms))

    fixed = {k:[r["controls"][k]["fixed"] for r in fixed_ok] for k in RULES}
    all_fixed = all(a["work"]["actual_scl_calls"]==ENVELOPE for group in fixed.values() for a in group)
    if not all_fixed:
        raise ValueError("Mislabeled equal-call endpoint")
    time_pairs = {}
    for name in ("under","over","matched_registered_calls"):
        def pick(r):
            a=r["controls"]["conventional"]
            return a[name] if name=="matched_registered_calls" else a["bracket"][name]
        selected = [(r,pick(r)) for r in rows if pick(r) is not None]
        for r,a in selected:
            if name=="matched_registered_calls" and a["work"]["actual_scl_calls"] != r["registered"]["continuation_calls"]:
                raise ValueError("Mislabeled paid-call match to registered continuation")
            if name!="matched_registered_calls" and (a["receiver_seconds"] <= r["registered"]["receiver_seconds"])!=(name=="under"):
                raise ValueError("Invalid clock bracket")
        time_pairs[name]=dict(excluded_seeds=[r["seed"] for r in rows if pick(r) is None],
            comparison=paired([a for r,a in selected],[r["registered"] for r,a in selected]),
            conventional=stats([a for r,a in selected]), registered=stats([r["registered"] for r,a in selected]))
    live = [r["replay"]["arms"]["live"] for r in rows]
    frozen = [r["replay"]["arms"]["frozen"] for r in rows]
    eligible = [r for r in rows if r["replay"]["opportunities"]]
    sides = {label:sum(len(r["replay"][label]) for r in rows) for label in ("live_only_true_ids","frozen_only_true_ids")}
    contrasts = {rule:paired(fixed[rule],fixed["reversible"]) for rule in ("conventional","add_only")}
    replay_contrast = paired(frozen,live)
    groups = {}
    for r in rows:
        for q in r["replay"]["queries"]:
            kinds = q["preceding_refit_move_kinds"]
            group = "unchanged" if not q["eligible"] else "includes_drop_or_swap" if {"drop","swap"}.intersection(kinds) else "add_only"
            g = groups.setdefault(group, Counter())
            g["queries"] += 1
            g["changed_inputs"] += int(q["changed_input"])
            g["changed_ordered_lists"] += int(q["live"]["ids"] != q["frozen"]["ids"])
    genuine_gains = all(a["count_savings"]>0 and a["count_savings_after_removing_best"]>0 for a in contrasts.values())
    clock = {k:time_pairs[k] for k in ("under","over")}
    gates = dict(fixed_paid_coverage=len(fixed_ok)==len(rows),
        fixed_reversible_gain_both_controls=len(fixed_ok)==len(rows) and genuine_gains,
        wall_comparison_fully_bracketed=all(not a["excluded_seeds"] for a in clock.values()),
        wall_registered_gain_both_neighbors=all(not a["excluded_seeds"] and a["comparison"]["count_savings"]>0 for a in clock.values()),
        replay_capability=sides["live_only_true_ids"]>sides["frozen_only_true_ids"] and replay_contrast["count_savings"]>0
            and replay_contrast["improved"]>=2 and replay_contrast["count_savings_after_removing_best"]>0)
    ok_seeds={r["seed"] for r in fixed_ok}
    return dict(frames=len(rows), transmitted_copies=100*len(rows), fixed_budget=ENVELOPE,
        gates=gates, fixed_all_arm_frames=len(fixed_ok), fixed_failed_seeds=[r["seed"] for r in rows if r["seed"] not in ok_seeds],
        fixed_arms={k:dict(**stats(v),work=dict(sum((Counter(a["work"]) for a in v),Counter()))) for k,v in fixed.items()},
        fixed_paired=contrasts, registered=stats([r["registered"] for r in rows]),
        conventional_vs_registered=time_pairs, replay=dict(all_scheduled_queries=sum(len(r["replay"]["queries"]) for r in rows),
        eligible_queries=sum(r["replay"]["opportunities"] for r in rows), eligible_frames=len(eligible),
        changed_input_queries=sum(r["replay"]["changed_inputs"] for r in rows), preceding_refit=groups, final_paired=replay_contrast,
        by_arm={name:dict(totals=sum_metrics(group), discoveries=sum(len(a["new_true"]) for a in group),
            retained_discoveries=sum(len(a["retained_new_true"]) for a in group), false_discoveries=sum(len(a["new_false"]) for a in group))
            for name,group in (("live",live),("frozen",frozen))}, **sides),
        replay_seconds=sum(r["replay"]["seconds"] for r in rows), actual_study_seconds=sum(r["study_seconds"] for r in rows))
