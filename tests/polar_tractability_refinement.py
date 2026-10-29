"""Finite checks of explored-set envelope repair and exact complement sampling; no decoder benchmark."""

import itertools
import json

import numpy as np

from baselines.polar import PolarCode


def bitrows(b):
    return np.array(list(itertools.product((0, 1), repeat=b)), dtype=int)


def exclusion_trie(excluded, probabilities):
    root = {}
    for row in excluded:
        node = root
        for bit in row:
            node = node.setdefault(int(bit), {})

    def remaining(node, depth):
        if node is None:
            return 1.
        if depth == len(probabilities):
            return 0.
        p = probabilities[depth]
        node["remaining"] = (1 - p) * remaining(node.get(0), depth + 1) + p * remaining(node.get(1), depth + 1)
        return node["remaining"]

    mass = remaining(root if len(excluded) else None, 0)

    def path_probability(row):
        if mass == 0:
            return 0.
        node, result = root if len(excluded) else None, 1.
        for depth, bit in enumerate(row):
            p = probabilities[depth] if bit else 1 - probabilities[depth]
            if node is None:
                result *= p
                continue
            child = node.get(int(bit))
            child_mass = 1. if child is None else (0. if depth + 1 == len(probabilities) else child["remaining"])
            if child_mass == 0:
                return 0.
            result *= p * child_mass / node["remaining"]
            node = child
        return result

    return mass, path_probability


def capped_count(probabilities, t):
    table = np.zeros((len(probabilities) + 1, t + 2))
    table[-1, 0] = 1.
    for depth in range(len(probabilities) - 1, -1, -1):
        p = probabilities[depth]
        for count in range(t + 2):
            table[depth, count] += (1 - p) * table[depth + 1, count]
            table[depth, min(t + 1, count + 1)] += p * table[depth + 1, count]
    return table


def component_ranks(generator):
    keys = sorted({sum(int(bit) << j for j, bit in enumerate(row)) for row in generator} - {0})
    pivots, basis = {}, []
    for key in keys:
        value, coordinate = key, 1 << len(basis)
        while value:
            bit = value.bit_length() - 1
            if bit not in pivots:
                pivots[bit] = (value, coordinate)
                basis.append(key)
                break
            value ^= pivots[bit][0]
            coordinate ^= pivots[bit][1]
    parent = list(range(len(basis)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for key in keys:
        value, coordinate = key, 0
        while value:
            pivot, combination = pivots[value.bit_length() - 1]
            value ^= pivot
            coordinate ^= combination
        support = [i for i in range(len(basis)) if (coordinate >> i) & 1]
        for j in support[1:]:
            parent[find(j)] = find(support[0])
    roots = [find(i) for i in range(len(basis))]
    return sorted([roots.count(root) for root in set(roots)], reverse=True)


def main():
    rng = np.random.default_rng(11027)
    errors, acceptance_gain = [], []
    for _ in range(30):
        b = 5
        rows = bitrows(b)
        h = rng.normal(0, .9, b)
        extra_keys = rng.choice([key for key in range(1, 1 << b) if key.bit_count() > 1], 5, replace=False)
        extra = rng.normal(0, .8, len(extra_keys))
        parity = np.array([[(int(sum(int(bit) << j for j, bit in enumerate(row))) & int(key)).bit_count() % 2
                            for key in extra_keys] for row in rows])
        reference_score = (1 - 2 * rows) @ h
        correction_score = (1 - 2 * parity) @ extra
        f = np.exp(reference_score + correction_score)
        zf = np.prod(2 * np.cosh(h))
        q = np.exp(reference_score) / zf
        upper = zf * np.exp(np.abs(extra).sum())
        p = 1 / (1 + np.exp(2 * h))
        previous = upper
        order = np.argsort(-q)
        for size in (0, 1, 3, 8, len(rows)):
            chosen = order[:size]
            keep = np.zeros(len(rows), dtype=bool)
            keep[chosen] = True
            tau, path = exclusion_trie(rows[chosen], p)
            complement = np.array([path(row) for row in rows])
            expected = np.where(keep, 0., q / tau) if tau > 0 else np.zeros_like(q)
            errors.extend([abs(tau - q[~keep].sum()), float(np.max(abs(complement - expected)))])
            list_mass = f[keep].sum()
            improved = list_mass + upper * tau
            assert f.sum() <= improved * (1 + 1e-12) <= previous * (1 + 1e-12) ** 2
            previous = improved
            proposal = np.where(keep, f / improved, upper * q / improved)
            acceptance = np.where(keep, 1., f / (upper * q))
            errors.extend([abs(proposal.sum() - 1), float(np.max(abs(proposal * acceptance - f / improved)))])
            miss = f[~keep].sum() / f.sum()
            assert miss <= upper * tau / improved + 1e-12
        acceptance_gain.append(float(upper / (f[order[:1]].sum() + upper * (1 - q[order[:1]].sum()))))
        for t in range(4):
            flip = rows != (h < 0)
            probabilities = 1 / (1 + np.exp(2 * np.abs(h)))
            table = capped_count(probabilities, t)
            errors.append(abs(table[0, -1] - q[flip.sum(axis=1) > t].sum()))

    triangle = []
    for strength in (1., 2., 4., 8.):
        rows = bitrows(2)
        signs = 1 - 2 * rows
        f = np.exp(strength * (signs[:, 0] + signs[:, 1] - signs[:, 0] * signs[:, 1]))
        q = np.exp(strength * signs.sum(axis=1)) / (2 * np.cosh(strength)) ** 2
        upper = np.exp(strength) * (2 * np.cosh(strength)) ** 2
        tau, _ = exclusion_trie(rows[:1], np.full(2, 1 / (1 + np.exp(2 * strength))))
        improved = f[0] + upper * tau
        exact_upper = 3 * np.exp(strength) + np.exp(-strength)
        errors.append(abs(improved / exact_upper - 1))
        assert np.isclose(f.sum(), 3 * np.exp(strength) + np.exp(-3 * strength))
        # Independent full-payload check: first two coordinates are parities; four others are truly free.
        full_rows = bitrows(6)
        full_signs = 1 - 2 * full_rows[:, :2]
        full_f = np.exp(strength * (full_signs[:, 0] + full_signs[:, 1] - full_signs[:, 0] * full_signs[:, 1]))
        full_q = np.exp(strength * full_signs.sum(axis=1)) / ((2 * np.cosh(strength)) ** 2 * 16)
        full_selected = (full_rows[:, :2] == 0).all(axis=1)
        full_upper, full_improved = 16 * upper, 16 * improved
        full_proposal = np.where(full_selected, full_f / full_improved, full_upper * full_q / full_improved)
        full_acceptance = np.where(full_selected, 1., full_f / (full_upper * full_q))
        errors.extend([abs(full_proposal.sum() - 1), float(np.max(abs(full_proposal * full_acceptance - full_f / full_improved)))])
        assert np.isclose(full_f.sum(), 16 * f.sum())
        triangle.append({"strength": strength, "old_acceptance": float(f.sum() / upper),
                         "one_cell_acceptance": float(f.sum() / improved),
                         "certified_omitted_mass": float(upper * tau / improved)})

    repeated = []
    for blocks in (1, 2, 3, 4):
        strength = 4.
        rows = bitrows(2 * blocks)
        signs = (1 - 2 * rows).reshape(-1, blocks, 2)
        score = strength * (signs[:, :, 0] + signs[:, :, 1] - signs[:, :, 0] * signs[:, :, 1]).sum(axis=1)
        f = np.exp(score)
        zf = (2 * np.cosh(strength)) ** (2 * blocks)
        upper = np.exp(blocks * strength) * zf
        q = np.exp(strength * (1 - 2 * rows).sum(axis=1)) / zf
        values = []
        for radius in (0, 1):
            chosen = rows.sum(axis=1) <= radius
            tau = capped_count(np.full(2 * blocks, 1 / (1 + np.exp(2 * strength))), radius)[0, -1]
            improved = f[chosen].sum() + upper * tau
            assert improved >= f.sum() * (1 - 1e-12)
            values.append(float(f.sum() / improved))
        product_z = (3 * np.exp(strength) + np.exp(-3 * strength)) ** blocks
        errors.append(abs(f.sum() / product_z - 1))
        repeated.append({"independent_triangles": blocks, "radius_0_acceptance": values[0], "radius_1_acceptance": values[1]})

    structures = []
    for b, n, crc in ((6, 32, 8), (32, 128, 16), (88, 512, 16)):
        code = PolarCode(b, n, crc_bits=crc)
        generator = code.encode(np.eye(b, dtype=np.uint8)).T
        structures.append({"b": b, "d": n, "crc": crc, "independent_component_ranks": component_ranks(generator)})
    assert max(errors) < 1e-11
    print(json.dumps({"random_fixtures": 30, "payloads_per_fixture": 32, "maximum_identity_error": max(errors),
                      "one_state_acceptance_gain_range": [min(acceptance_gain), max(acceptance_gain)],
                      "triangle": triangle, "independent_triangle_boundary": repeated, "generator_structure": structures}, indent=2))


if __name__ == "__main__":
    main()
