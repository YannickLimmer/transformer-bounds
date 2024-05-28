from typing import Optional

import numba as nb
import numpy as np
from numba import njit

from src.DerivativeBounds import DBoundDict, make_dbound_dict, compute_bound_for_alpha, adjust_bounds
from src.DerivativeTypes import generate_derivative_subtypes
from src.Hashing import der_types_to_hashes
from src.Util import dt_factorial


@njit
def der_bound_affine(der_type: nb.int16[:], max_weight: nb.float64) -> nb.float64:
    if der_type.sum() == 1:
        return max_weight

    return 0.0


@njit
def der_bounds_affine(n: nb.int16, k: nb.int16, max_weight: nb.float64, by_level: bool = False) -> DBoundDict:
    der_types = generate_derivative_subtypes(n, k)
    hashes = der_types_to_hashes(der_types, n, k)
    bounds = np.zeros(len(hashes), dtype=np.float64)
    for i in range(len(der_types)):
        bounds[i] = der_bound_affine(der_types[i], max_weight)
    return make_dbound_dict(hashes, adjust_bounds(bounds, der_types, by_level))


@njit
def der_bound_activation(der_type: nb.int16[:], activation_type: str) -> nb.float64:
    n = der_type[0]
    if activation_type == "softplus":
        return np.array((
            0.2499937375858079,
            0.0962235075187988,
            0.12498747540692284,
            0.12767176692735638,
            0.24994677183240732,
            0.40829206775030613,
            1.0621117599680119,
            2.389542792103157,
            7.745673170620478,
            22.250481611410578,
        ), dtype=np.float64)[n-1]
    if activation_type == "GeLU":
        return np.array((
            1.1289025122335812,
            0.483902054114163,
            0.7537641738878936,
            1.6604202092207108,
            4.343069523840463,
            12.951146761684768,
            42.77396923768182,
            153.75936037576147,
            594.1691297439795,
            2445.6871342102463,
        ), dtype=np.float64)[n-1]
    if activation_type == "tanh":
        return np.array((
            2.0,
            4.0,
            8.0,
            16.0,
            32.0,
            156.6477975035857,
            1651.3150725846722,
            20405.428679687368,
            292561.94803214446,
            4769038.092076319,
            87148321.71215685,
        ), dtype=np.float64)[n-1]
    if activation_type == "SWISH":
        return np.array((
            1.0998393194985212,
            0.499999749950055,
            0.3081813157522826,
            0.4999992498503332,
            0.6580276660942772,
            1.4999957491531088,
            2.9085518831984785,
            8.499961242292986,
            21.75604346094711,
            77.49948164716983,
        ), dtype=np.float64)[n-1]


@njit
def der_bounds_activation(n: nb.int16, activation_type: str, by_level: bool = False) -> DBoundDict:
    der_types = generate_derivative_subtypes(n, 1)
    hashes = der_types_to_hashes(der_types, n, 1)
    bounds = np.zeros(len(hashes), dtype=np.float64)
    for i in range(len(der_types)):
        bounds[i] = der_bound_activation(der_types[i], activation_type)

    return make_dbound_dict(hashes, adjust_bounds(bounds, der_types, by_level))


@njit
def der_bound_neural_network(
        der_type: nb.int16[:],
        hidden_dim: nb.int16,
        max_weight_a: nb.float64,
        max_weight_b: nb.float64,
        activation_type: str,
        by_level: bool = False,
) -> nb.float64:
    n = der_type.sum()
    m = nb.int16(1)
    k = nb.int16(len(der_type))

    activation_bounds = der_bounds_activation(n, activation_type, by_level)
    affine_bounds = der_bounds_affine(n, k, max_weight_a, by_level)

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
        activation_type: Optional[str] = None,
        by_level: bool = False,
) -> DBoundDict:
    activation_type = "" if activation_type is None else activation_type
    der_types = generate_derivative_subtypes(n, k)
    hashes = der_types_to_hashes(der_types, n, k)
    bounds = np.zeros(len(hashes), dtype=np.float64)
    for i in range(len(der_types)):
        bounds[i] = der_bound_neural_network(
            der_types[i],
            hidden_dim,
            max_weight_a,
            max_weight_b,
            activation_type,
            by_level,
        )
    return make_dbound_dict(hashes, adjust_bounds(bounds, der_types, by_level))
