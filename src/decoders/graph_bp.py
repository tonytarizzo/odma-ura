"""Graph-BP / EP decoder (V2 case, h = 1_M, multi-antenna).

Turbo-scheduled block-EP with stable cavity-by-omission, EM updates for
lambda (Poisson rate) and sigma^2 (noise variance). See
docs/decoder_summary.tex for the algebra. Original implementation lives in
the v2e family of historical scripts.
"""

from __future__ import annotations

import time
from itertools import combinations, product as iproduct

import numpy as np
import numpy.linalg as nla

from ..metrics import assemble_global_counts
from ..scenario import Scenario


def _graph_bp_core(
    Y: np.ndarray,
    P_mats: dict[int, np.ndarray],
    block_dicts: dict[int, np.ndarray],
    *,
    max_iter: int = 20,
    damping: float = 0.3,
    tol: float = 1e-4,
    lambda_init: float | None = None,
    noise_var_init: float | None = None,
    poisson_tail_tol: float = 1e-4,
    support_tail_tol: float = 1e-4,
    rng_seed: int | None = None,
    max_wall_seconds: float | None = None,
    verbose: bool = False,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict]:
    """Turbo-scheduled block-EP decoder with cavity-by-omission. Returns
    (coeffs_mmse, coeffs_map, meta)."""
    n, num_ant = Y.shape
    dtype = Y.dtype
    var_floor = 1e-10
    site_floor = 1e-10

    h = np.ones(num_ant, dtype=dtype)
    gamma = float(np.real(np.dot(h.conj(), h)))

    rng = np.random.default_rng(rng_seed)

    # ----- helpers --------------------------------------------------------

    def logsumexp(v: np.ndarray) -> float:
        vmax = float(np.max(v))
        return vmax + float(np.log(np.sum(np.exp(v - vmax)))) if np.isfinite(vmax) else vmax

    def poisson_pmf_vec(lam: float) -> np.ndarray:
        lam = max(float(lam), 1e-12)
        probs = [np.exp(-lam)]
        total = probs[0]
        c = 0
        while 1.0 - total > poisson_tail_tol:
            c += 1
            probs.append(probs[-1] * lam / c)
            total += probs[-1]
            if probs[-1] == 0.0:
                break
        p = np.array(probs, dtype=np.float64)
        return p / np.sum(p)

    def max_active_per_block(L_b: int, p_nz: float) -> int:
        p_nz = float(np.clip(p_nz, 1e-12, 1.0 - 1e-12))
        p0 = 1.0 - p_nz
        pk = p0 ** L_b
        cdf = pk
        for k in range(L_b):
            pk = pk * ((L_b - k) / (k + 1)) * (p_nz / p0)
            cdf += pk
            if 1.0 - cdf <= support_tail_tol:
                return k + 1
        return L_b

    def decode_block(C_b, r_cav, V_cav, lam):
        pmf = poisson_pmf_vec(lam)
        c_max = len(pmf) - 1
        L_b = C_b.shape[0]
        p_nz = float(1.0 - pmf[0])
        k_max = max_active_per_block(L_b, p_nz)
        log_pmf = np.log(pmf + 1e-300)

        d_b = V_cav.shape[0]
        V_reg = V_cav + var_floor * np.eye(d_b, dtype=V_cav.dtype)
        try:
            Prec = nla.solve(V_reg, np.eye(d_b, dtype=V_cav.dtype))
        except nla.LinAlgError:
            Prec = np.eye(d_b, dtype=V_cav.dtype) / var_floor
        Prec = 0.5 * (Prec + Prec.conj().T)

        states: list[np.ndarray] = [np.zeros(L_b, dtype=np.float64)]
        log_prior: list[float] = [L_b * log_pmf[0]]
        for k in range(1, k_max + 1):
            lp_zeros = (L_b - k) * log_pmf[0]
            for idxs in combinations(range(L_b), k):
                for cnts in iproduct(range(1, c_max + 1), repeat=k):
                    a = np.zeros(L_b, dtype=np.float64)
                    a[list(idxs)] = np.array(cnts, dtype=np.float64)
                    states.append(a)
                    log_prior.append(lp_zeros + sum(log_pmf[c] for c in cnts))

        A = np.array(states, dtype=np.float64)
        X = A @ C_b
        err = X - r_cav[None, :]
        ll = -np.real(np.einsum('si,ij,sj->s', err.conj(), Prec, err))
        log_post = ll + np.array(log_prior, dtype=np.float64)
        log_post -= logsumexp(log_post)
        w = np.exp(log_post)
        x_mean = w @ X
        a_mean = w @ A
        X_c = X - x_mean[None, :]
        Sigma_xb = np.einsum('s,si,sj->ij', w, X_c.conj(), X_c)
        Sigma_xb = 0.5 * (Sigma_xb + Sigma_xb.conj().T)
        Sigma_xb = Sigma_xb + var_floor * np.eye(d_b, dtype=Sigma_xb.dtype)
        a_map = A[int(np.argmax(log_post))]
        return a_mean, x_mean, Sigma_xb, a_map

    # ----- coordinate layout ---------------------------------------------
    block_keys = list(block_dicts.keys())
    B = len(block_keys)
    block_dim = {b: block_dicts[b].shape[1] for b in block_keys}
    block_offset = {}
    offset = 0
    for b in block_keys:
        block_offset[b] = offset
        offset += block_dim[b]
    N = offset

    block_supports = {b: np.argmax(P_mats[b], axis=0).astype(int) for b in P_mats}

    resource_to_edges: list[list[tuple[int, int, int]]] = [[] for _ in range(n)]
    for b in block_keys:
        S_b = block_supports[b]
        off = block_offset[b]
        for j, r in enumerate(S_b):
            resource_to_edges[r].append((b, j, off + j))

    M_total = float(sum(block_dicts[b].shape[0] for b in block_keys))
    lambda_est = float(lambda_init) if lambda_init is not None else 1.0 / M_total
    noise_var = (float(noise_var_init) if noise_var_init is not None
                 else float(np.real(np.vdot(Y.ravel(), Y.ravel()))) / (n * num_ant))

    def build_resource_system(nv: float):
        Lg_res = np.zeros((N, N), dtype=dtype)
        eg_res = np.zeros(N, dtype=dtype)
        J_sc = gamma / nv
        for r in range(n):
            edges = resource_to_edges[r]
            if not edges:
                continue
            gidxs = [gp for (_, _, gp) in edges]
            u_sc = np.dot(h.conj(), Y[r]) / nv
            for gi in gidxs:
                eg_res[gi] += u_sc
                for gj in gidxs:
                    Lg_res[gi, gj] += J_sc
        return Lg_res, eg_res

    site_Lambda = {b: site_floor * np.eye(block_dim[b], dtype=dtype) for b in block_keys}
    site_eta = {b: np.zeros(block_dim[b], dtype=dtype) for b in block_keys}

    coeffs_hat = {b: np.zeros(block_dicts[b].shape[0]) for b in block_keys}
    coeffs_map = {b: np.zeros(block_dicts[b].shape[0]) for b in block_keys}

    def cavity_by_omission(Lg, eg, b):
        o = block_offset[b]
        d = block_dim[b]
        Lg_omit = Lg.copy()
        Lg_omit[o:o+d, o:o+d] -= site_Lambda[b]
        eg_omit = eg.copy()
        eg_omit[o:o+d] -= site_eta[b]
        Lg_reg = Lg_omit + 1e-12 * np.eye(N, dtype=dtype)
        rhs = np.zeros((N, d + 1), dtype=dtype)
        rhs[:, 0] = eg_omit
        rhs[o:o+d, 1:] = np.eye(d, dtype=dtype)
        try:
            sol = nla.solve(Lg_reg, rhs)
        except nla.LinAlgError:
            r_cav = np.zeros(d, dtype=dtype)
            V_cav = (1.0 / site_floor) * np.eye(d, dtype=dtype)
            return r_cav, V_cav
        r_cav = sol[o:o+d, 0]
        V_raw = sol[o:o+d, 1:]
        V_cav = 0.5 * (V_raw + V_raw.conj().T) + var_floor * np.eye(d, dtype=dtype)
        return r_cav, V_cav

    def update_site(x_mean, Sigma_xb, r_cav, V_cav, b):
        d = block_dim[b]
        Sig_reg = Sigma_xb + var_floor * np.eye(d, dtype=Sigma_xb.dtype)
        rhs_S = np.zeros((d, d + 1), dtype=dtype)
        rhs_S[:, 0] = x_mean
        rhs_S[:, 1:] = np.eye(d, dtype=dtype)
        try:
            sol_S = nla.solve(Sig_reg, rhs_S)
            Sig_inv_xm = sol_S[:, 0]
            Sig_inv = sol_S[:, 1:]
        except nla.LinAlgError:
            Sig_inv = np.eye(d, dtype=dtype) / var_floor
            Sig_inv_xm = x_mean / var_floor

        V_reg = V_cav + var_floor * np.eye(d, dtype=V_cav.dtype)
        rhs_V = np.zeros((d, d + 1), dtype=dtype)
        rhs_V[:, 0] = r_cav
        rhs_V[:, 1:] = np.eye(d, dtype=dtype)
        try:
            sol_V = nla.solve(V_reg, rhs_V)
            Vcav_inv_r = sol_V[:, 0]
            Vcav_inv = sol_V[:, 1:]
        except nla.LinAlgError:
            Vcav_inv = np.eye(d, dtype=dtype) / var_floor
            Vcav_inv_r = r_cav / var_floor

        Lambda_new = 0.5 * ((Sig_inv - Vcav_inv) + (Sig_inv - Vcav_inv).conj().T)
        eta_new = Sig_inv_xm - Vcav_inv_r
        eigs = nla.eigvalsh(Lambda_new)
        if np.min(eigs) < site_floor:
            Lambda_new += (site_floor - np.min(eigs)) * np.eye(d, dtype=dtype)
        return Lambda_new, eta_new

    converged = False
    timed_out = False
    it_used = 0
    history: list[dict] = []
    wall_start = time.time()

    for it in range(1, max_iter + 1):
        it_used = it
        if max_wall_seconds is not None and (time.time() - wall_start) > max_wall_seconds:
            timed_out = True
            break

        Lg_resource, eg_resource = build_resource_system(noise_var)
        Lg = Lg_resource.copy()
        eg = eg_resource.copy()
        for b in block_keys:
            o = block_offset[b]
            d = block_dim[b]
            Lg[o:o+d, o:o+d] += site_Lambda[b]
            eg[o:o+d] += site_eta[b]

        visit_order = rng.permutation(B).tolist()
        delta = 0.0
        total_mean_count = 0.0
        total_x_var_post = 0.0
        n_site_floored = 0

        for idx in visit_order:
            b = block_keys[idx]
            C_b = block_dicts[b]
            o = block_offset[b]
            d = block_dim[b]

            r_cav, V_cav = cavity_by_omission(Lg, eg, b)
            a_mean, x_mean, Sigma_xb, a_map = decode_block(C_b, r_cav, V_cav, lambda_est)
            coeffs_hat[b] = a_mean
            coeffs_map[b] = a_map
            total_mean_count += float(np.sum(a_mean))
            total_x_var_post += float(np.trace(Sigma_xb).real)

            Lambda_old = site_Lambda[b]
            eta_old = site_eta[b]
            Lambda_new, eta_new = update_site(x_mean, Sigma_xb, r_cav, V_cav, b)

            eigs_new = nla.eigvalsh(Lambda_new)
            if np.min(eigs_new) <= site_floor + 1e-15:
                n_site_floored += 1

            Lambda_damp = (1.0 - damping) * Lambda_new + damping * Lambda_old
            eta_damp = (1.0 - damping) * eta_new + damping * eta_old

            mean_old = nla.solve(Lambda_old + site_floor * np.eye(d, dtype=dtype), eta_old)
            mean_new = nla.solve(Lambda_damp + site_floor * np.eye(d, dtype=dtype), eta_damp)
            delta = max(delta, float(np.max(np.abs(mean_new - mean_old))))

            Lg[o:o+d, o:o+d] += (Lambda_damp - Lambda_old)
            eg[o:o+d] += (eta_damp - eta_old)
            site_Lambda[b] = Lambda_damp
            site_eta[b] = eta_damp

        lambda_est = max(total_mean_count / M_total, 1e-12)
        y_hat = np.zeros(n, dtype=dtype)
        for b in block_keys:
            y_hat[block_supports[b]] += block_dicts[b].T @ coeffs_hat[b]
        resid_mat = Y - np.outer(y_hat, h)
        resid_energy = float(np.real(np.vdot(resid_mat.ravel(), resid_mat.ravel())))
        noise_var = max(
            (resid_energy + gamma * total_x_var_post) / (n * num_ant), var_floor)

        history.append({
            "iter": it,
            "delta": delta,
            "lambda": lambda_est,
            "noise_var": noise_var,
            "k_est": total_mean_count,
            "n_site_floored": n_site_floored,
        })
        if verbose:
            print(f"  [iter {it:03d}] delta={delta:.3e}  k_est={total_mean_count:.2f}"
                  f"  lambda={lambda_est:.3e}  sigma2={noise_var:.3e}"
                  f"  site_floor={n_site_floored}/{B}", flush=True)

        if delta < tol:
            converged = True
            break

    return coeffs_hat, coeffs_map, {
        "converged":   converged,
        "timed_out":   timed_out,
        "iterations":  it_used,
        "history":     history,
        "tol":         tol,
        "damping":     damping,
        "lam":         lambda_est,
        "noise_var_est": noise_var,
        "wall_s":      time.time() - wall_start,
    }


def run(scenario: Scenario, *, max_iter: int = 20, damping: float = 0.3,
        tol: float = 1e-4, max_wall_seconds: float | None = None,
        verbose: bool = False) -> tuple[np.ndarray, dict]:
    _, coeffs_map, meta = _graph_bp_core(
        scenario.Y, scenario.P_mats, scenario.block_dicts,
        max_iter=max_iter, damping=damping, tol=tol,
        rng_seed=scenario.seed,
        max_wall_seconds=max_wall_seconds, verbose=verbose,
    )
    counts = assemble_global_counts(coeffs_map, scenario.block_to_msg_list,
                                     scenario.num_codewords)
    return counts, meta
