"""Core types for the explicit URA framework.

The framework is organised around the factorisation
    Phi = sum_l B_l U_l T_l,  B_l = [R_{l,1} C_l | ... | R_{l,Q_l} C_l]
described in docs/reports/03_factorised_encoder_framework.tex. One *experiment* corresponds to
one Encoder instance that holds the (R, C, U, T) factors for every component.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import torch


@dataclass(frozen=True)
class URASpec:
    """Scalar configuration of a URA experiment."""

    n: int                          # resource-grid length
    num_codewords: int              # M = 2^B in Polyanskiy URA notation
    num_active: int                 # K_a
    num_antennas: int = 1
    payload_bits: int | None = None
    energy_per_codeword: float = 1.0

    def __post_init__(self) -> None:
        if self.n <= 0:
            raise ValueError(f"n must be positive, got {self.n}")
        if self.num_codewords < 2:
            raise ValueError(f"num_codewords must be >= 2, got {self.num_codewords}")
        if self.num_active <= 0:
            raise ValueError(f"num_active must be positive, got {self.num_active}")
        if self.num_antennas <= 0:
            raise ValueError(f"num_antennas must be positive, got {self.num_antennas}")
        if self.payload_bits is None:
            from math import log2
            object.__setattr__(self, "payload_bits", int(log2(self.num_codewords)))
        if self.payload_bits <= 0:
            raise ValueError(f"payload_bits must be positive, got {self.payload_bits}")


@dataclass(frozen=True)
class SectionedURASpec:
    """Scalar configuration for the scalable section-domain backend.

    Unlike :class:`URASpec`, this type intentionally has no ``num_codewords``
    field. A payload may describe ``2^B`` messages, but the executable state is
    carried by bounded local section alphabets instead of a global message axis.
    """

    n: int
    payload_bits: int
    num_active: int
    num_antennas: int = 1
    energy_per_codeword: float = 1.0

    def __post_init__(self) -> None:
        if self.n <= 0:
            raise ValueError(f"n must be positive, got {self.n}")
        if self.payload_bits <= 0:
            raise ValueError(f"payload_bits must be positive, got {self.payload_bits}")
        if self.num_active <= 0:
            raise ValueError(f"num_active must be positive, got {self.num_active}")
        if self.num_antennas <= 0:
            raise ValueError(f"num_antennas must be positive, got {self.num_antennas}")
        if self.energy_per_codeword <= 0.0:
            raise ValueError(f"energy_per_codeword must be positive, got {self.energy_per_codeword}")


@dataclass
class ComponentSpec:
    """Configuration of one product component B_l U_l T_l.

    Init strings drive `framework.initializers` and select how the underlying
    matrices are constructed. Concrete values are chosen there.
    """

    Q: int                          # number of operators in the bank
    d: int                          # local codeword length
    V: int                          # local codebook size
    N: int | None = None           # number of valid atoms (defaults to Q*V)
    R_init: str = "random_gaussian"
    C_init: str = "random_gaussian"
    U_init: str = "all_pairs"        # one of {all_pairs, random_subset, explicit}
    T_init: str = "round_robin"      # one of {round_robin, identity, random, explicit}
    learn_R: bool = False
    learn_C: bool = False
    explicit_R: torch.Tensor | None = None
    explicit_C: torch.Tensor | None = None
    explicit_atom_q: torch.Tensor | None = None
    explicit_atom_v: torch.Tensor | None = None
    explicit_msg_to_atom: torch.Tensor | None = None


@dataclass
class URABatch:
    """One synthesised batch.

    Shape convention (B = batch size, n = resource length, M_ant = antennas):
        counts:   (B, M)            -- nonnegative integer counts a in Z_>=0
        y_clean:  (B, n)            -- Phi a (real or complex)
        Y_clean:  (B, n, M_ant)     -- after fading channel: outer(y, h)
        Y:        (B, n, M_ant)     -- Y_clean + AWGN
        H:        (B, M_ant)        -- per-realisation channel vector (known)
        num_active: (B,)            -- realised K_a for each sample
    """

    counts: torch.Tensor
    y_clean: torch.Tensor
    Y_clean: torch.Tensor
    Y: torch.Tensor
    H: torch.Tensor
    noise_var: float
    active_messages: torch.Tensor
    num_active: torch.Tensor
    ebn0_db: float


@dataclass
class SectionedURABatch:
    """A batch whose targets live on local section alphabets, never on ``M``.

    ``active_paths`` has shape ``(batch, K_max, L)`` and stores one local-atom
    index per section. ``section_counts[l]`` has shape ``(batch, N_l)``.
    A procedural outer code constrains which paths are legal without changing
    this physical-channel representation.
    """

    section_counts: tuple[torch.Tensor, ...]
    active_paths: torch.Tensor
    y_clean: torch.Tensor
    Y_clean: torch.Tensor
    Y: torch.Tensor
    H: torch.Tensor
    noise_var: float
    num_active: torch.Tensor
    ebn0_db: float


@dataclass
class DecoderOutput:
    """Result of a global decoder over message counts."""

    counts: torch.Tensor                # (B, M) or (M,)
    meta: dict = field(default_factory=dict)


@dataclass
class SectionedDecoderOutput:
    """Local section-count output; complete message association is deliberately separate."""

    section_counts: tuple[torch.Tensor, ...]
    meta: dict = field(default_factory=dict)


@dataclass
class OuterBPOutput:
    """Differentiable outer-code beliefs over one randomly selected active path."""

    log_beliefs: torch.Tensor             # (batch,L,2^J), normalised over the final axis
    meta: dict = field(default_factory=dict)


@dataclass
class PathListOutput:
    """Discrete valid-path list and inferred complete-message multiplicities."""

    paths: torch.Tensor                   # (batch,list_size,L)
    counts: torch.Tensor                  # (batch,list_size)
    scores: torch.Tensor                  # (batch,list_size)
    bits: torch.Tensor                    # (batch,list_size,B)
    meta: dict = field(default_factory=dict)
