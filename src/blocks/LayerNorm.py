import numba as nb
import numpy as np
from numba import njit

from src.DerivativeBounds import compute_bound_for_alpha, DBoundDict, make_dbound_dict
from src.DerivativeTypes import generate_derivative_subtypes
from src.Hashing import der_types_to_hashes, der_type_to_hash
from src.Util import double_factorial


@njit
def der_bound_variance(der_type: nb.int16[:], domain_bound: nb.float64, weights: nb.float64) -> nb.float64:
    if der_type.sum() == 1:
        return 2 * weights * domain_bound
    if der_type.sum() == 2:
        return 2 * weights

    return 0.0


@njit
def der_bounds_variance(n: nb.int16, k: nb.int16, domain_bound: nb.float64, weights: nb.float64) -> DBoundDict:
    der_types = generate_derivative_subtypes(n, k)
    hashes = der_types_to_hashes(der_types, n, k)
    bounds = np.zeros(len(hashes), dtype=np.float64)
    for i in range(len(der_types)):
        bounds[i] = der_bound_variance(der_types[i], domain_bound, weights)
    return make_dbound_dict(hashes, bounds)


@njit
def der_bound_g(der_type: nb.int16[:]) -> nb.float64:
    return nb.float64(double_factorial(2 * der_type[0] + 1)) / 2 ** (2 * der_type[0])


@njit
def der_bounds_g(n: nb.int16) -> DBoundDict:
    der_types = generate_derivative_subtypes(n, 1)
    hashes = der_types_to_hashes(der_types, n, 1)
    bounds = np.zeros(len(hashes), dtype=np.float64)
    for i in range(len(der_types)):
        bounds[i] = der_bound_g(der_types[i])
    return make_dbound_dict(hashes, bounds)


@njit
def der_bound_g_circ_variance(der_type: nb.int16[:], domain_bound: nb.float64, weights: nb.float64) -> nb.float64:
    n = der_type.sum()
    m = nb.int16(1)
    k = nb.int16(len(der_type))

    g_bounds = der_bounds_g(n)
    variance_bounds = der_bounds_variance(n, k, domain_bound, weights)

    return compute_bound_for_alpha(n, m, k, der_type, g_bounds, variance_bounds)


@njit
def der_bounds_g_circ_variance(n: nb.int16, k: nb.int16, domain_bound: nb.float64, weights: nb.float64) -> DBoundDict:
    der_types = generate_derivative_subtypes(n, k)
    hashes = der_types_to_hashes(der_types, n, k)
    bounds = np.zeros(len(hashes), dtype=np.float64)
    for i in range(len(der_types)):
        bounds[i] = der_bound_g_circ_variance(der_types[i], domain_bound, weights)
    return make_dbound_dict(hashes, bounds)


@njit
def der_bounds_layer_norm(n: nb.int16, k: nb.int16, domain_bound: nb.float64, weights: nb.float64):
    g_circ_variance_bounds = der_bounds_g_circ_variance(n, k, domain_bound, weights)

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
                    bounds[j] += der_types[j][i] * g_circ_variance_bounds[score]
        else:
            bounds[j] += domain_bound * g_circ_variance_bounds[der_type_to_hash(der_types[j], n, k)]

    return make_dbound_dict(hashes, bounds)
