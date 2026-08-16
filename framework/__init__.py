"""URA factor framework with explicit-global and scalable section backends.

Mirrors the factorisation Phi = sum_l B_l U_l T_l from
docs/reports/04_ura_framework.tex. The legacy ODMA code in `src/` is left
untouched. The explicit backend retains M-shaped objects for small-system
certification; `framework.sectioned` executes from local section counts only.
"""

from .core import (ComponentSpec, DecoderOutput, SectionedDecoderOutput,
                   SectionedURABatch, SectionedURASpec, URABatch, URASpec)
from .datasets import CountDataset, DatasetConfig, generate_uniform_count_dataset
from .encoder import ComponentConstraints, Encoder, LocalAtomBank, ProductComponent, build_encoder
from .outer_code import (IdentityOuterCode, LinearCheck, OuterCode, OuterFactorGraph,
                         SparseLinearOuterCode, gf_multiply, random_sparse_outer_code, triadic_outer_code)
from .sectioned import (SectionedEncoder, build_sectioned_encoder, outer_code_path_generator, sample_sectioned_batch,
                        sectioned_from_explicit, uniform_section_paths_generator)

__all__ = [
    "ComponentConstraints",
    "ComponentSpec",
    "CountDataset",
    "DatasetConfig",
    "DecoderOutput",
    "Encoder",
    "IdentityOuterCode",
    "LinearCheck",
    "LocalAtomBank",
    "OuterCode",
    "OuterFactorGraph",
    "ProductComponent",
    "SectionedDecoderOutput",
    "SectionedEncoder",
    "SectionedURABatch",
    "SectionedURASpec",
    "SparseLinearOuterCode",
    "URABatch",
    "URASpec",
    "build_encoder",
    "build_sectioned_encoder",
    "generate_uniform_count_dataset",
    "gf_multiply",
    "outer_code_path_generator",
    "random_sparse_outer_code",
    "sample_sectioned_batch",
    "sectioned_from_explicit",
    "uniform_section_paths_generator",
    "triadic_outer_code",
]
