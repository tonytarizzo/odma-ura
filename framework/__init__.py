"""Explicit URA framework package.

Mirrors the factorisation Phi = sum_l B_l U_l T_l from
docs/reports/04_ura_framework.tex. The legacy ODMA code in `src/` is left
untouched; new experiments should be written against `framework`.
"""

from .core import ComponentSpec, DecoderOutput, URABatch, URASpec
from .datasets import CountDataset, DatasetConfig, generate_uniform_count_dataset
from .encoder import ComponentConstraints, Encoder, ProductComponent, build_encoder

__all__ = [
    "ComponentConstraints",
    "ComponentSpec",
    "CountDataset",
    "DatasetConfig",
    "DecoderOutput",
    "Encoder",
    "ProductComponent",
    "URABatch",
    "URASpec",
    "build_encoder",
    "generate_uniform_count_dataset",
]
