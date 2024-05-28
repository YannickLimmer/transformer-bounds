import numpy as np
from numba.typed.typeddict import Dict
import numba as nb
from numba import njit, prange
from numba.typed import List

from src.Combinatorics import compute_etas, compute_sorted_zetas, number_of_representations
from src.DerivativeTypes import generate_derivative_subtypes, _generate_derivative_subtypes
from src.Hashing import der_type_to_hash
from src.Util import dt_factorial, dt_sum

DBoundDict = Dict[nb.int64, nb.float64]


def _make_dbound_dict(hashes: nb.int64[::1], bounds: nb.int64[::1]) -> DBoundDict:
    d = Dict.empty(key_type=nb.int64, value_type=nb.float64)
    for i in range(len(hashes)):
        d[hashes[i]] = bounds[i]
    return d


make_dbound_dict = njit(_make_dbound_dict)


@nb.njit
def adjust_bounds(bounds: nb.float64[::1], dtypes: List[nb.int16[::1]], by_level: bool = False) -> nb.float64[::1]:
    if not by_level:
        return bounds
    result = np.zeros(len(bounds), dtype=np.float64)
    n = len(dtypes)

    for i in range(n):
        for j in range(n):
            if dtypes[i].sum() <= dtypes[j].sum():
                result[j] = max(result[j], bounds[i])

    return result


def _compute_cumulated_g_bounds_for_zeta(
        n: np.int16,
        k: np.int16,
        j: np.int16,
        eta: nb.int64[:, ::1],
        zeta: nb.int64[:, ::1],
        g_dbounds: DBoundDict,
) -> nb.int64:
    g_bound_product: nb.float64 = 1
    for i in range(n - j):
        numerator = g_dbounds[der_type_to_hash(zeta[i], n, k)] ** dt_sum(eta[i])
        denominator = dt_factorial(eta[i]) * (dt_factorial(zeta[i]) ** dt_sum(eta[i]))
        g_bound_product *= (numerator / denominator)
    return g_bound_product


compute_cumulated_g_bounds_for_zeta = njit(_compute_cumulated_g_bounds_for_zeta)


def _compute_cumulated_g_bounds_for_eta(
        n: np.int16,
        k: np.int16,
        j: np.int16,
        eta: nb.int64[:, ::1],
        h_der_type: nb.int16[::1],
        g_dbounds: DBoundDict,
) -> nb.int64:
    zetas = compute_sorted_zetas(n, k, j, eta, h_der_type)

    cumulated_g_bounds: nb.float64 = 0
    for i in prange(len(zetas)):
        cumulated_g_bounds += compute_cumulated_g_bounds_for_zeta(
            n, k, j, eta, zetas[i], g_dbounds
        )
    return cumulated_g_bounds


compute_cumulated_g_bounds_for_eta = njit(_compute_cumulated_g_bounds_for_eta, parallel=False)


def _compute_cumulated_g_bounds_for_j(
        n: np.int16,
        m: np.int16,
        k: np.int16,
        j: np.int16,
        h_der_type: nb.int16[::1],
        f_der_type: nb.int16[::1],
        g_dbounds: DBoundDict,
) -> nb.int64:
    etas = compute_etas(n, m, j, f_der_type)

    cumulated_g_bounds: nb.float64 = 0
    for i in prange(len(etas)):
        cumulated_g_bounds += compute_cumulated_g_bounds_for_eta(
            n, k, j, etas[i], h_der_type, g_dbounds
        )
    return cumulated_g_bounds


compute_cumulated_g_bounds_for_j = njit(_compute_cumulated_g_bounds_for_j, parallel=True)


def _compute_cumulated_g_bounds(
        n: np.int16,
        m: np.int16,
        k: np.int16,
        h_der_type: nb.int16[::1],
        f_der_type: nb.int16[::1],
        g_dbounds: DBoundDict,
) -> nb.int64:
    cumulated_g_bounds: nb.float64 = 0
    js = np.arange(1, n+1, dtype=np.int16)
    for j in prange(0, len(js)):
        cumulated_g_bounds += compute_cumulated_g_bounds_for_j(
            n, m, k, js[j], h_der_type, f_der_type, g_dbounds
        )
    return cumulated_g_bounds


compute_cumulated_g_bounds = njit(_compute_cumulated_g_bounds, parallel=False)


def _compute_bound_for_alpha(
        n: np.int16,
        m: np.int16,
        k: np.int16,
        h_der_type: nb.int16[::1],
        f_dbounds: DBoundDict,
        g_dbounds: DBoundDict,
):
    result: nb.float64 = 0

    f_der_types = generate_derivative_subtypes(n, m)

    for i in prange(len(f_der_types)):
        f_dbound = f_dbounds[der_type_to_hash(f_der_types[i], n, m)]
        representations_for_der_type = number_of_representations(f_der_types[i])
        cumulated_g_bounds = compute_cumulated_g_bounds(n, m, k, h_der_type, f_der_types[i], g_dbounds)
        result += representations_for_der_type * f_dbound * cumulated_g_bounds

    return result


compute_bound_for_alpha = njit(_compute_bound_for_alpha, parallel=True)
