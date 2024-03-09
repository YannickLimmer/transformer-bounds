import numba as nb
import numpy as np
from numba import njit
from numba.typed import List

from src.Util import dt_sum, sort_and_fill, binomial_coefficient


def _compute_etas(
        n: np.int16,
        m: np.int16,
        j: np.int16,
        f_der_type: nb.int16[::1],
):
    arr_dim = m if np.min(f_der_type) > 0 else np.argmin(f_der_type != 0)
    cur_arrs = List([np.zeros((n - j, arr_dim), dtype=np.int16)])
    cur_targets = List([f_der_type])
    for row in range(n - j):
        for col in range(arr_dim):
            next_arrs = List.empty_list(nb.int16[:, ::1])
            next_targets = List.empty_list(nb.int16[::1])
            for i in range(len(cur_arrs)):
                min_val = 0 if row < n - j - 1 else cur_targets[i][col]
                if col == arr_dim - 1:
                    if dt_sum(cur_arrs[i][row]) == 0:
                        min_val = max(min_val, 1)
                max_val = cur_targets[i][col]
                for val in range(min_val, max_val + 1):
                    next_arr = cur_arrs[i].copy()
                    next_arr[row][col] = val
                    next_arrs.append(next_arr)

                    next_target = cur_targets[i].copy()
                    next_target[col] -= val
                    next_targets.append(next_target)
            cur_arrs = next_arrs
            cur_targets = next_targets
    return cur_arrs


compute_etas = njit(_compute_etas)


def _compute_zetas(
        n: np.int16,
        k: np.int16,
        j: np.int16,
        eta: nb.int16[::1],
        h_der_type: nb.int16[::1],
):
    arr_dim = k if np.min(h_der_type) > 0 else np.argmin(h_der_type != 0)
    cur_arrs = List([np.zeros((n - j, arr_dim), dtype=np.int16)])
    cur_targets = List([h_der_type])
    cur_allowance = List(np.array([n], dtype=np.int16))

    for row in range(n - j):
        factor = dt_sum(eta[row])

        if row > 0:
            for i in range(len(cur_allowance)):
                cur_allowance[i] = dt_sum(cur_arrs[i][row - 1])

        for col in range(arr_dim):
            next_arrs = List.empty_list(nb.int16[:, ::1])
            next_targets = List.empty_list(nb.int16[::1])
            next_allowance = List.empty_list(nb.int16)

            for i in range(len(cur_arrs)):
                min_val = 0 if row < n - j - 1 else (cur_targets[i][col] + factor - 1) // factor
                if col == arr_dim - 1:
                    if dt_sum(cur_arrs[i][row]) == 0:
                        min_val = max(min_val, 1)

                max_val = min(cur_targets[i][col] // factor, cur_allowance[i])
                for val in range(min_val, max_val + 1):
                    next_arr = cur_arrs[i].copy()
                    next_arr[row][col] = val
                    next_arrs.append(next_arr)

                    next_target = cur_targets[i].copy()
                    next_target[col] -= val * factor
                    next_targets.append(next_target)

                    next_allowance.append(np.int16(cur_allowance[i] - val))

            cur_arrs = next_arrs
            cur_targets = next_targets
            cur_allowance = next_allowance

    return cur_arrs


compute_zetas = njit(_compute_zetas)


def _compute_sorted_zetas(
        n: np.int16,
        k: np.int16,
        j: np.int16,
        eta: nb.int16[::1],
        h_der_type: nb.int16[::1],
):
    cur_arrs = compute_zetas(n, k, j, eta, h_der_type)
    result = List.empty_list(nb.int16[:, ::1])
    for i in range(len(cur_arrs)):
        result.append(sort_and_fill(cur_arrs[i], k))

    return result


compute_sorted_zetas = njit(_compute_sorted_zetas)


def _number_of_representations(der_type: nb.int16[::1]) -> nb.int64:
    i = 0
    counter = 1
    result: nb.int64 = 1
    to_choose = len(der_type)
    prev = der_type[i]
    while prev > 0 and i < len(der_type):
        i += 1
        cur = der_type[i]
        if cur == prev:
            counter += 1
        else:
            result *= binomial_coefficient(to_choose, counter)
            to_choose -= counter
            counter = 1
        prev = cur
    return result


number_of_representations = njit(_number_of_representations)
