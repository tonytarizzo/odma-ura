"""Differentiable modular BP and discrete multiuser path extraction.

D0 supplies section-local evidence.  BP applies only the sparse outer validity
constraints and remains differentiable over the complete local alphabets.  A
separate evaluation-only beam converts those soft beliefs into complete paths.
"""

from __future__ import annotations

import torch
from torch import nn

from .core import OuterBPOutput, PathListOutput
from .decoders import active_count_vector, project_nonneg_integer_total
from .learned_decoders import UnrolledSectionedCountPGD, _inv_softplus, _sigmoid_logit
from .losses import section_support_count_loss, sectioned_power_penalty
from .outer_code import OuterFactorGraph, SparseLinearOuterCode
from .sectioned import SectionedEncoder


def _stack_section_tensors(values: tuple[torch.Tensor, ...] | list[torch.Tensor], name: str) -> torch.Tensor:
    if not values or any(value.ndim != 2 for value in values):
        raise ValueError(f"{name} must contain nonempty (batch,alphabet) tensors")
    if any(value.shape != values[0].shape for value in values):
        raise ValueError(f"{name} requires a common section alphabet")
    return torch.stack(tuple(values), dim=1)


class DifferentiableOuterBP(nn.Module):
    """Full-alphabet sum-product for sparse checks ``H x = 0 mod 2^J``."""

    def __init__(self, num_iterations: int = 3, init_temperature: float = 1.0,
                 init_damping: float = 0.15, init_check_gain: float = 1.0) -> None:
        super().__init__()
        if num_iterations <= 0:
            raise ValueError(f"num_iterations must be positive, got {num_iterations}")
        self.num_iterations = int(num_iterations)
        self.raw_temperature = nn.Parameter(torch.tensor(_inv_softplus(init_temperature)))
        self.raw_damping = nn.Parameter(torch.full((num_iterations,), _sigmoid_logit(init_damping)))
        self.raw_check_gain = nn.Parameter(torch.full((num_iterations,), _inv_softplus(init_check_gain)))

    @staticmethod
    def _coefficient_maps(coefficient: int, modulus: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        inverse = pow(int(coefficient), -1, modulus)
        values = torch.arange(modulus, dtype=torch.long, device=device)
        inverse_map = (inverse * values).remainder(modulus)       # z -> x with coefficient*x=z
        negative_map = (-int(coefficient) * values).remainder(modulus)  # x -> -coefficient*x
        return inverse_map, negative_map

    @classmethod
    def _check_messages(cls, graph: OuterFactorGraph,
                        variable_messages: dict[tuple[int, int], torch.Tensor]) -> dict[tuple[int, int], torch.Tensor]:
        outputs: dict[tuple[int, int], torch.Tensor] = {}
        modulus = graph.modulus
        for factor, check in enumerate(graph.checks):
            transforms: dict[int, torch.Tensor] = {}
            negative_maps: dict[int, torch.Tensor] = {}
            for variable, coefficient in zip(check.variables, check.coefficients):
                incoming = variable_messages[(factor, variable)]
                inverse_map, negative_map = cls._coefficient_maps(coefficient, modulus, incoming.device)
                transformed = incoming.index_select(-1, inverse_map)
                transforms[variable] = torch.fft.rfft(transformed, n=modulus, dim=-1)
                negative_maps[variable] = negative_map
            for variable in check.variables:
                product = None
                for other in check.variables:
                    if other == variable:
                        continue
                    product = transforms[other] if product is None else product * transforms[other]
                convolution = torch.fft.irfft(product, n=modulus, dim=-1).clamp_min(0.0)
                outgoing = convolution.index_select(-1, negative_maps[variable]).clamp_min(1e-30)
                outputs[(factor, variable)] = outgoing / outgoing.sum(dim=-1, keepdim=True).clamp_min(1e-30)
        return outputs

    def forward(self, graph: OuterFactorGraph, section_logits: tuple[torch.Tensor, ...] | list[torch.Tensor]
                ) -> OuterBPOutput:
        logits = _stack_section_tensors(section_logits, "section_logits")
        if logits.shape[1:] != (graph.num_variables, graph.modulus):
            raise ValueError(f"BP logits must have shape (batch,{graph.num_variables},{graph.modulus}), got {tuple(logits.shape)}")
        temperature = torch.nn.functional.softplus(self.raw_temperature).clamp_min(1e-4)
        unary_log = torch.log_softmax(logits / temperature, dim=-1)
        variable_neighbors = [[] for _ in range(graph.num_variables)]
        variable_messages: dict[tuple[int, int], torch.Tensor] = {}
        for factor, check in enumerate(graph.checks):
            for variable in check.variables:
                variable_neighbors[variable].append(factor)
                variable_messages[(factor, variable)] = unary_log[:, variable].exp()
        layer_log_beliefs = []
        for iteration in range(self.num_iterations):
            check_messages = self._check_messages(graph, variable_messages)
            gain = torch.nn.functional.softplus(self.raw_check_gain[iteration])
            damping = torch.sigmoid(self.raw_damping[iteration])
            updated: dict[tuple[int, int], torch.Tensor] = {}
            for variable, factors in enumerate(variable_neighbors):
                for destination in factors:
                    log_message = unary_log[:, variable]
                    for factor in factors:
                        if factor != destination:
                            log_message = log_message + gain * torch.log(check_messages[(factor, variable)].clamp_min(1e-30))
                    proposal = torch.softmax(log_message, dim=-1)
                    mixed = damping * variable_messages[(destination, variable)] + (1.0 - damping) * proposal
                    updated[(destination, variable)] = mixed / mixed.sum(dim=-1, keepdim=True).clamp_min(1e-30)
            beliefs = []
            for variable, factors in enumerate(variable_neighbors):
                log_belief = unary_log[:, variable]
                for factor in factors:
                    log_belief = log_belief + gain * torch.log(check_messages[(factor, variable)].clamp_min(1e-30))
                beliefs.append(torch.log_softmax(log_belief, dim=-1))
            variable_messages = updated
            layer_log_beliefs.append(torch.stack(beliefs, dim=1))
        return OuterBPOutput(log_beliefs=layer_log_beliefs[-1],
                             meta={"layer_log_beliefs": layer_log_beliefs, "temperature": temperature,
                                   "decoder": "differentiable_modular_outer_bp"})


class SectionedOuterDecoder(nn.Module):
    """End-to-end differentiable D0 followed by generic outer-code BP."""

    def __init__(self, d0_layers: int = 8, bp_iterations: int = 3, power_iters: int = 12) -> None:
        super().__init__()
        self.d0 = UnrolledSectionedCountPGD(num_layers=d0_layers, power_iters=power_iters)
        self.bp = DifferentiableOuterBP(num_iterations=bp_iterations)

    def forward(self, encoder: SectionedEncoder, outer_code: SparseLinearOuterCode,
                Y: torch.Tensor, H: torch.Tensor, num_active: int | torch.Tensor,
                noise_var: float | torch.Tensor | None = None) -> OuterBPOutput:
        encoder._validate_outer_code(outer_code)
        d0_output = self.d0(encoder, Y, H, num_active, noise_var)
        bp_output = self.bp(outer_code.factor_graph, d0_output.meta["section_support_logits"])
        bp_output.meta["d0_output"] = d0_output
        return bp_output

    def decode(self, encoder: SectionedEncoder, outer_code: SparseLinearOuterCode,
               Y: torch.Tensor, H: torch.Tensor, num_active: int | torch.Tensor,
               noise_var: float | torch.Tensor | None = None,
               list_decoder: ValidPathListDecoder | None = None) -> PathListOutput:
        """Run the complete evaluation path: D0, BP, then discrete valid-path extraction."""
        beliefs = self.forward(encoder, outer_code, Y, H, num_active, noise_var)
        extractor = ValidPathListDecoder() if list_decoder is None else list_decoder
        soft_counts = beliefs.meta["d0_output"].meta["soft_section_counts"]
        return extractor.decode(beliefs, outer_code, soft_counts, num_active)


def outer_marginal_loss(output: OuterBPOutput, true_section_counts: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Cross-entropy for the section symbol of a uniformly selected active user."""
    target = _stack_section_tensors(list(true_section_counts), "true_section_counts").to(output.log_beliefs.dtype)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return -torch.sum(target * output.log_beliefs, dim=-1).mean()


def path_scores(log_beliefs: torch.Tensor, paths: torch.Tensor) -> torch.Tensor:
    """Add section log-beliefs along candidate paths."""
    if paths.ndim != 3 or log_beliefs.ndim != 3 or paths.shape[0] != log_beliefs.shape[0] or paths.shape[2] != log_beliefs.shape[1]:
        raise ValueError("log_beliefs and paths must have shapes (batch,L,q) and (batch,P,L)")
    safe = paths.clamp_min(0).to(torch.long)
    score = torch.zeros(paths.shape[:2], dtype=log_beliefs.dtype, device=log_beliefs.device)
    for section in range(paths.shape[2]):
        score += log_beliefs[:, section].gather(1, safe[:, :, section])
    return score.masked_fill((paths < 0).any(dim=-1), float("-inf"))


def outer_path_contrastive_loss(output: OuterBPOutput, true_paths: torch.Tensor,
                                outer_code: SparseLinearOuterCode, num_negatives: int = 32,
                                generator: torch.Generator | None = None) -> torch.Tensor:
    """Rank every true path above random and cross-user recombination negatives."""
    if num_negatives <= 0 or true_paths.ndim != 3 or true_paths.shape[2] != outer_code.num_sections:
        raise ValueError("num_negatives must be positive and true_paths must have shape (batch,K,L)")
    device = output.log_beliefs.device
    true_paths = true_paths.to(device=device, dtype=torch.long)
    batch = true_paths.shape[0]
    mixed_information = torch.zeros(batch, num_negatives, outer_code.num_information_sections,
                                    dtype=torch.long, device=device)
    for b in range(batch):
        valid = true_paths[b, :, 0] >= 0
        information = true_paths[b, valid][:, list(outer_code.info_positions)]
        if information.shape[0] == 0:
            raise ValueError("every batch item must contain at least one active path")
        choices = torch.randint(information.shape[0], (num_negatives, information.shape[1]),
                                generator=generator, device=device)
        mixed_information[b] = information[choices, torch.arange(information.shape[1], device=device)]
    mixed_paths = outer_code.encode_symbols(mixed_information)
    random_bits = torch.randint(2, (batch, num_negatives, outer_code.payload_bits),
                                generator=generator, device=device)
    negative_paths = torch.cat((mixed_paths, outer_code.encode_bits(random_bits)), dim=1)
    positive = path_scores(output.log_beliefs, true_paths)
    negative = path_scores(output.log_beliefs, negative_paths)
    negative_is_true = (negative_paths.unsqueeze(2) == true_paths.unsqueeze(1)).all(dim=-1).any(dim=-1)
    negative = negative.masked_fill(negative_is_true, float("-inf"))
    normalizer = torch.logsumexp(negative, dim=1, keepdim=True)
    losses = torch.logaddexp(positive, normalizer) - positive
    valid_positive = torch.isfinite(positive)
    return losses[valid_positive].mean()


def sectioned_outer_training_loss(output: OuterBPOutput, true_section_counts: tuple[torch.Tensor, ...],
                                  true_paths: torch.Tensor, outer_code: SparseLinearOuterCode,
                                  encoder: SectionedEncoder | None = None, lambda_d0_count: float = 0.1,
                                  lambda_marginal: float = 1.0, lambda_path: float = 0.2,
                                  lambda_power: float = 0.0, num_path_negatives: int = 32,
                                  generator: torch.Generator | None = None
                                  ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """One differentiable objective for D0, BP marginals, path ranking, and sampled power."""
    if "d0_output" not in output.meta:
        raise ValueError("sectioned_outer_training_loss requires output from SectionedOuterDecoder")
    d0, d0_parts = section_support_count_loss(output.meta["d0_output"], true_section_counts,
                                              lambda_count=lambda_d0_count)
    marginal = outer_marginal_loss(output, true_section_counts)
    path = outer_path_contrastive_loss(output, true_paths, outer_code, num_path_negatives, generator)
    total = d0 + float(lambda_marginal) * marginal + float(lambda_path) * path
    parts = {"d0": d0, "d0_support": d0_parts["support"], "d0_count": d0_parts["count"],
             "outer_marginal": marginal, "outer_path": path}
    if lambda_power > 0.0:
        if encoder is None:
            raise ValueError("encoder is required when lambda_power is positive")
        power = sectioned_power_penalty(encoder, true_paths[true_paths[..., 0] >= 0])
        total = total + float(lambda_power) * power
        parts["power"] = power
    parts["total"] = total
    return total, parts


class ValidPathListDecoder:
    """Evaluation-only systematic beam search followed by optional multiplicity fitting."""

    def __init__(self, beam_width: int = 256, list_extra: int = 16,
                 candidate_cap: int | None = 512, collision_payload_bits: int = 20,
                 multiplicity_iterations: int = 80) -> None:
        if beam_width <= 0 or list_extra < 0 or candidate_cap is not None and candidate_cap <= 0:
            raise ValueError("beam_width and candidate_cap must be positive and list_extra nonnegative")
        self.beam_width = int(beam_width)
        self.list_extra = int(list_extra)
        self.candidate_cap = None if candidate_cap is None else int(candidate_cap)
        self.collision_payload_bits = int(collision_payload_bits)
        self.multiplicity_iterations = int(multiplicity_iterations)

    def _beam_one(self, beliefs: torch.Tensor, code: SparseLinearOuterCode, list_size: int
                  ) -> tuple[torch.Tensor, torch.Tensor]:
        q = beliefs.shape[1]
        cap = q if self.candidate_cap is None else min(q, self.candidate_cap)
        assignments = torch.empty(1, code.num_information_sections, dtype=torch.long, device=beliefs.device)
        scores = torch.zeros(1, dtype=beliefs.dtype, device=beliefs.device)
        completion = [[] for _ in range(code.num_information_sections)]
        for parity, support in enumerate(code.parity_supports):
            completion[max(support)].append(parity)
        for information_index, position in enumerate(code.info_positions):
            candidates = torch.topk(beliefs[position], cap).indices
            expanded = assignments.repeat_interleave(cap, dim=0)
            expanded[:, information_index] = candidates.repeat(assignments.shape[0])
            expanded_scores = (scores.unsqueeze(1) + beliefs[position, candidates].unsqueeze(0)).reshape(-1)
            for parity_index in completion[information_index]:
                support = code.parity_supports[parity_index]
                coefficients = code.parity_coefficients[parity_index]
                parity = torch.zeros(expanded.shape[0], dtype=torch.long, device=beliefs.device)
                for source, coefficient in zip(support, coefficients):
                    parity += coefficient * expanded[:, source]
                parity = (-parity).remainder(q)
                expanded_scores += beliefs[code.parity_positions[parity_index], parity]
            keep = min(self.beam_width, expanded_scores.numel())
            scores, indices = torch.topk(expanded_scores, keep)
            assignments = expanded.index_select(0, indices)
        paths = code.encode_symbols(assignments)
        keep = min(list_size, paths.shape[0])
        return paths[:keep], scores[:keep]

    def _fit_multiplicities(self, paths: torch.Tensor, section_counts: tuple[torch.Tensor, ...], total: int) -> torch.Tensor:
        candidates = paths.shape[0]
        gathered = torch.stack([section_counts[ell][paths[:, ell]] for ell in range(paths.shape[1])], dim=1)
        state = gathered.mean(dim=1).clamp_min(1e-8)
        state = state * float(total) / state.sum().clamp_min(1e-12)
        step = 1.0 / max(paths.shape[1] * candidates, 1)
        for _ in range(self.multiplicity_iterations):
            gradient = torch.zeros_like(state)
            for ell, local_target in enumerate(section_counts):
                prediction = torch.zeros_like(local_target)
                prediction.scatter_add_(0, paths[:, ell], state)
                gradient += (prediction - local_target)[paths[:, ell]]
            state = torch.relu(state - step * gradient)
            state = state * float(total) / state.sum().clamp_min(1e-12)
        return project_nonneg_integer_total(state, total).to(state.dtype)

    def decode(self, output: OuterBPOutput, outer_code: SparseLinearOuterCode,
               soft_section_counts: tuple[torch.Tensor, ...], num_active: int | torch.Tensor) -> PathListOutput:
        beliefs = output.log_beliefs
        K = active_count_vector(num_active, beliefs.shape[0], beliefs.device)
        list_size = min(self.beam_width, int(K.max().item()) + self.list_extra)
        all_paths = []; all_scores = []
        for b in range(beliefs.shape[0]):
            paths, scores = self._beam_one(beliefs[b], outer_code, list_size)
            if paths.shape[0] < list_size:
                pad = list_size - paths.shape[0]
                paths = torch.cat((paths, torch.full((pad, outer_code.num_sections), -1, dtype=torch.long, device=paths.device)))
                scores = torch.cat((scores, torch.full((pad,), float("-inf"), dtype=scores.dtype, device=scores.device)))
            all_paths.append(paths); all_scores.append(scores)
        paths = torch.stack(all_paths); scores = torch.stack(all_scores)
        counts = torch.zeros_like(scores)
        if outer_code.payload_bits <= self.collision_payload_bits:
            for b in range(beliefs.shape[0]):
                local = tuple(section[b].to(scores.dtype) for section in soft_section_counts)
                valid = paths[b, :, 0] >= 0
                counts[b, valid] = self._fit_multiplicities(paths[b, valid], local, int(K[b].item()))
            collision_mode = "complete_path_multiplicity"
        else:
            for b in range(beliefs.shape[0]):
                counts[b, :min(int(K[b].item()), list_size)] = 1.0
            collision_mode = "unique_complete_paths"
        bits = outer_code.decode_bits(paths.clamp_min(0), validate=False)
        bits = bits.masked_fill((paths[:, :, :1] < 0), -1)
        return PathListOutput(paths=paths, counts=counts, scores=scores, bits=bits,
                              meta={"collision_mode": collision_mode, "beam_width": self.beam_width,
                                    "candidate_cap": self.candidate_cap, "list_size": list_size})


def path_list_pupe(output: PathListOutput, true_paths: torch.Tensor) -> torch.Tensor:
    """Multiplicity-aware per-user probability of error for complete paths."""
    values = []
    for b in range(true_paths.shape[0]):
        truth: dict[tuple[int, ...], int] = {}
        for path in true_paths[b]:
            if int(path[0]) < 0:
                continue
            key = tuple(int(value) for value in path.tolist())
            truth[key] = truth.get(key, 0) + 1
        matched = 0
        for path, count in zip(output.paths[b], output.counts[b].round().to(torch.long)):
            if int(path[0]) < 0 or int(count) <= 0:
                continue
            key = tuple(int(value) for value in path.tolist())
            available = truth.get(key, 0)
            take = min(available, int(count))
            matched += take; truth[key] = available - take
        total = sum(truth.values()) + matched
        values.append((total - matched) / max(total, 1))
    return torch.tensor(values, dtype=torch.float64, device=true_paths.device)
