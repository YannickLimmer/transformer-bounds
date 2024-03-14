import numba as nb
import numpy as np
from numba import njit

from src.DerivativeBounds import DBoundDict, make_dbound_dict, compute_bound_for_alpha
from src.DerivativeTypes import generate_derivative_subtypes
from src.Hashing import der_types_to_hashes, der_type_to_hash
from src.Util import factorial, dt_sum


@njit
def der_bound_dp(
        der_type: nb.int16[:],
        ndim_k: nb.int16,
        max_weight_k: nb.float64,
        max_weight_q: nb.float64,
        domain_bound: nb.float64,
) -> nb.float64:
    if der_type.sum() == 1:
        return 2 * len(der_type) * ndim_k * max_weight_q * max_weight_k * domain_bound
    if der_type.sum() == 2:
        return 2 * ndim_k * max_weight_q * max_weight_k

    return 0.0


@njit
def der_bounds_dp(
        n: nb.int16,
        k: nb.int16,
        ndim_k: nb.int16,
        max_weight_k: nb.float64,
        max_weight_q: nb.float64,
        domain_bound: nb.float64,
) -> DBoundDict:
    der_types = generate_derivative_subtypes(n, k)
    hashes = der_types_to_hashes(der_types, n, k)
    bounds = np.zeros(len(hashes), dtype=np.float64)
    for i in range(len(der_types)):
        bounds[i] = der_bound_dp(der_types[i], ndim_k, max_weight_k, max_weight_q, domain_bound)
    return make_dbound_dict(hashes, bounds)


@njit
def der_bounds_softmax(n: nb.int16, k: nb.int16) -> DBoundDict:
    der_types = generate_derivative_subtypes(n, k)
    hashes = der_types_to_hashes(der_types, n, k)
    bounds = np.zeros(len(hashes), dtype=np.float64)
    for i in range(len(der_types)):
        bounds[i] = factorial(dt_sum(der_types[i]))
    return make_dbound_dict(hashes, bounds)


@njit
def der_bounds_softmax_circ_dp(
        n: nb.int16,
        input_dim: nb.int16,
        sequence_length: nb.int16,
        ndim_k: nb.int16,
        max_weight_k: nb.float64,
        max_weight_q: nb.float64,
        domain_bound: nb.float64,
) -> DBoundDict:
    k = sequence_length * input_dim
    der_types = generate_derivative_subtypes(n, k)
    hashes = der_types_to_hashes(der_types, n, k)
    bounds = np.zeros(len(hashes), dtype=np.float64)

    softmax_bounds = der_bounds_softmax(n, k)
    dp_bounds = der_bounds_dp(n, k, ndim_k, max_weight_k, max_weight_q, domain_bound)

    for i in range(len(der_types)):
        bounds[i] = compute_bound_for_alpha(n, sequence_length, k, der_types[i], softmax_bounds, dp_bounds)
    return make_dbound_dict(hashes, bounds)


@njit
def der_bounds_attention(
        n: nb.int16,
        input_dim: nb.int16,
        sequence_length: nb.int16,
        ndim_k: nb.int16,
        max_weight_k: nb.float64,
        max_weight_q: nb.float64,
        max_weight_v: nb.float64,
        domain_bound: nb.float64,
):
    softmax_circ_dp_bounds = der_bounds_softmax_circ_dp(
        n,
        input_dim,
        sequence_length,
        ndim_k,
        max_weight_k,
        max_weight_q,
        domain_bound,
    )

    k = sequence_length * input_dim
    der_types = generate_derivative_subtypes(n, k)
    hashes = der_types_to_hashes(der_types, n, k)
    bounds = np.zeros(len(hashes), dtype=np.float64)

    for j in range(len(der_types)):
        if der_types[j].sum() > 1:
            for i in range(len(der_types[j])):
                if der_types[j][i] > 0:
                    cur_type = der_types[j].copy()
                    cur_type[i] -= 1
                    score = der_type_to_hash(np.sort(cur_type)[::-1], n, k)
                    bounds[j] += der_types[j][i] * softmax_circ_dp_bounds[score]

        bounds[j] += domain_bound * softmax_circ_dp_bounds[der_type_to_hash(der_types[j], n, k)]
        bounds[j] *= k * max_weight_v

    return make_dbound_dict(hashes, bounds)
