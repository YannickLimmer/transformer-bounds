import numba as nb
import numpy as np
from numba import njit

from src.DerivativeBounds import DBoundDict, make_dbound_dict, compute_bound_for_alpha
from src.DerivativeTypes import generate_derivative_subtypes
from src.Hashing import der_types_to_hashes
from src.Util import dt_factorial


@njit
def der_bound_affine(der_type: nb.int16[:], max_weight: nb.float64) -> nb.float64:
    if der_type.sum() == 1:
        return max_weight

    return 0.0


@njit
def der_bounds_affine(n: nb.int16, k: nb.int16, max_weight: nb.float64) -> DBoundDict:
    der_types = generate_derivative_subtypes(n, k)
    hashes = der_types_to_hashes(der_types, n, k)
    bounds = np.zeros(len(hashes), dtype=np.float64)
    for i in range(len(der_types)):
        bounds[i] = der_bound_affine(der_types[i], max_weight)
    return make_dbound_dict(hashes, bounds)


@njit
def der_bound_activation(der_type: nb.int16[:]) -> nb.float64:
    return nb.float64(1)  # Placeholder for now


@njit
def der_bounds_activation(n: nb.int16) -> DBoundDict:
    der_types = generate_derivative_subtypes(n, 1)
    hashes = der_types_to_hashes(der_types, n, 1)
    bounds = np.zeros(len(hashes), dtype=np.float64)
    for i in range(len(der_types)):
        bounds[i] = der_bound_activation(der_types[i])
    return make_dbound_dict(hashes, bounds)


@njit
def der_bound_neural_network(
        der_type: nb.int16[:],
        hidden_dim: nb.int16,
        max_weight_a: nb.float64,
        max_weight_b: nb.float64,
) -> nb.float64:
    n = der_type.sum()
    m = nb.int16(1)
    k = nb.int16(len(der_type))

    activation_bounds = der_bounds_activation(n)
    affine_bounds = der_bounds_affine(n, k, max_weight_a)

    factor = dt_factorial(der_type) * max_weight_b * hidden_dim
    addon = 0.0 if der_type.sum() != 1 else max_weight_b

    return compute_bound_for_alpha(n, m, k, der_type, activation_bounds, affine_bounds) * factor + addon


@njit
def der_bounds_neural_network(
        n: nb.int16,
        k: nb.int16,
        hidden_dim: nb.int16,
        max_weight_a: nb.float64,
        max_weight_b: nb.float64,
) -> DBoundDict:
    der_types = generate_derivative_subtypes(n, k)
    hashes = der_types_to_hashes(der_types, n, k)
    bounds = np.zeros(len(hashes), dtype=np.float64)
    for i in range(len(der_types)):
        bounds[i] = der_bound_neural_network(der_types[i], hidden_dim, max_weight_a, max_weight_b)
    return make_dbound_dict(hashes, bounds)