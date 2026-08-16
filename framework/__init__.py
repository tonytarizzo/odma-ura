"""URA factor framework with explicit-global and scalable section backends.

Mirrors the factorisation Phi = sum_l B_l U_l T_l from
docs/reports/04_ura_framework.tex. The legacy ODMA code in `src/` is left
untouched. The explicit backend retains M-shaped objects for small-system
certification; `framework.sectioned` executes from local section counts only.
"""

from .core import (ComponentSpec, DecoderOutput, OuterBPOutput, PathListOutput, SectionedDecoderOutput,
                   SectionedURABatch, SectionedURASpec, URABatch, URASpec)
from .datasets import CountDataset, DatasetConfig, generate_uniform_count_dataset
from .encoder import ComponentConstraints, Encoder, LocalAtomBank, ProductComponent, SubsampledHadamardAtomBank, build_encoder
from .outer_code import (IdentityOuterCode, LinearCheck, OuterCode, OuterFactorGraph,
                         SparseLinearOuterCode, ccs_amp_paper_outer_code, random_sparse_outer_code, triadic_outer_code)
from .outer_decoder import (DifferentiableOuterBP, SectionedOuterDecoder, ValidPathListDecoder,
                            outer_marginal_loss, outer_path_contrastive_loss, path_list_pupe,
                            sectioned_outer_training_loss)
from .sectioned import (FixedOrthogonalMixer, SectionedEncoder, build_orthogonal_sectioned_encoder,
                        build_default_scalable_setup, build_sectioned_encoder, outer_code_path_generator, sample_sectioned_batch,
                        sampled_energy_report, sectioned_from_explicit, uniform_section_paths_generator)

__all__ = [
    "ComponentConstraints",
    "ComponentSpec",
    "CountDataset",
    "DatasetConfig",
    "DecoderOutput",
    "DifferentiableOuterBP",
    "Encoder",
    "FixedOrthogonalMixer",
    "IdentityOuterCode",
    "LinearCheck",
    "LocalAtomBank",
    "OuterCode",
    "OuterBPOutput",
    "OuterFactorGraph",
    "ProductComponent",
    "PathListOutput",
    "SectionedDecoderOutput",
    "SectionedEncoder",
    "SectionedOuterDecoder",
    "SectionedURABatch",
    "SectionedURASpec",
    "SparseLinearOuterCode",
    "SubsampledHadamardAtomBank",
    "URABatch",
    "URASpec",
    "ValidPathListDecoder",
    "build_encoder",
    "build_default_scalable_setup",
    "build_orthogonal_sectioned_encoder",
    "build_sectioned_encoder",
    "ccs_amp_paper_outer_code",
    "generate_uniform_count_dataset",
    "outer_code_path_generator",
    "outer_marginal_loss",
    "outer_path_contrastive_loss",
    "path_list_pupe",
    "random_sparse_outer_code",
    "sample_sectioned_batch",
    "sampled_energy_report",
    "sectioned_from_explicit",
    "sectioned_outer_training_loss",
    "uniform_section_paths_generator",
    "triadic_outer_code",
]
