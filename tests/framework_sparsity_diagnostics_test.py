"""Small deterministic checks for density-sweep codebook diagnostics."""

from __future__ import annotations

from argparse import Namespace

import torch

from tests.framework_product_experiment import build_experiment_encoder
from tests.framework_sparsity_diagnostics import analyse_encoder_sparsity, log2_binomial


def build_sparse(n: int, B: int, support: int, seed: int, nested: bool = False):
    args = Namespace(encoder="sparse_global_fixed", payload_bits=B, n=n, num_antennas=1,
                     Q=4, odma_d=None, sparse_support=support, k_min=2, k_max=4,
                     sparse_nested=nested, eval_k=None, extrapolate_k=False)
    return build_experiment_encoder(args, torch.Generator().manual_seed(seed))[0]


def build_dense(n: int, B: int, seed: int):
    args = Namespace(encoder="dense_fixed", payload_bits=B, n=n, num_antennas=1,
                     Q=1, odma_d=None, sparse_support=None, sparse_nested=False,
                     k_min=2, k_max=4, eval_k=None, extrapolate_k=False)
    return build_experiment_encoder(args, torch.Generator().manual_seed(seed))[0]


def main() -> None:
    n, B = 16, 6
    sparse_one = analyse_encoder_sparsity(build_sparse(n, B, 1, 3), num_pairs=400, active_samples=20, active_k=4, seed=4)
    assert sparse_one["support_size"] == 1
    assert sparse_one["max_unit_energy_deviation"] < 1e-6
    assert sparse_one["distinct_signed_support_patterns"] <= 2 * n
    assert sparse_one["signed_support_duplicate_fraction"] > 0
    assert sparse_one["low_density_support_mask_shortfall"]

    sparse_four = analyse_encoder_sparsity(build_sparse(n, B, 4, 5), num_pairs=400, active_samples=20, active_k=4, seed=6)
    assert sparse_four["support_size"] == 4
    assert sparse_four["nonzero_fraction"] == 0.25
    assert abs(sparse_four["log2_available_support_masks"] - log2_binomial(n, 4)) < 1e-10
    assert sparse_four["max_unit_energy_deviation"] < 1e-5
    assert sparse_four["sampled_pair_count"] == 400
    assert sparse_four["active_geometry_samples"] == 20

    nested_one = build_sparse(n, B, 1, 7, nested=True).explicit_matrix()
    nested_four = build_sparse(n, B, 4, 7, nested=True).explicit_matrix()
    assert torch.all((nested_one != 0) <= (nested_four != 0))
    shared = nested_one != 0
    assert torch.all(torch.sign(nested_one[shared]) == torch.sign(nested_four[shared]))

    # This seed previously produced one exact float32 zero in a 256 x 4096 Gaussian draw.
    for encoder in (build_sparse(256, 12, 256, 2702, nested=True), build_dense(256, 12, 2702)):
        full = analyse_encoder_sparsity(encoder)
        assert full["support_size"] == 256
        assert full["nonzero_fraction"] == 1.0
    print("framework sparsity diagnostics test passed")


if __name__ == "__main__":
    main()
