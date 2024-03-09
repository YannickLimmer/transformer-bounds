import numba as nb
import numpy as np
from numba import njit
from numba.typed import List


def _dt_factorial(arr: nb.int16[::1]) -> nb.int64:
    res: nb.int64 = 1
    for i in range(len(arr)):
        for j in range(1, arr[i] + 1):
            if j > 0:
                res *= j
    return res


dt_factorial = njit(_dt_factorial)


@njit
def dt_sum(arr: nb.int16[::1]) -> nb.int16:
    res: nb.int64 = 0
    for i in range(len(arr)):
        res += arr[i]
    return res


def _sort_and_fill(arr: nb.int16[:, ::1], k: nb.int16):
    rows, cols = arr.shape
    sorted_and_filled = np.zeros((rows, k), dtype=np.int16)
    for i in range(arr.shape[0]):
        sorted_and_filled[i][:cols] = np.sort(arr[i])[::-1]
    return sorted_and_filled


sort_and_fill = njit(_sort_and_fill)


@nb.njit
def binomial_coefficient(n: nb.int64, k: nb.int64) -> nb.int64:
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1

    result = 1
    for i in range(1, min(k, n - k) + 1):
        result *= (n - i + 1) // i
    return result


def _fill_zeros(arrs: List[nb.int16[::1]], k: np.int16) -> List[nb.int16[::1]]:
    new_arrs = List.empty_list(nb.int16[::1])
    for i in range(len(arrs)):
        new_arr = np.zeros(k, dtype=np.int16)
        for j in range(arrs[i].shape[0]):
            new_arr[j] = arrs[i][j]
        new_arrs.append(new_arr)
    return new_arrs


fill_zeros = njit(_fill_zeros)


def _remove_tail(arrs: List[nb.int16[::1]], n: np.int16) -> List[nb.int16[::1]]:
    new_arrs = List(np.zeros((len(arrs), n), dtype=np.int16))
    for i in range(len(arrs)):
        new_arrs[i] = arrs[i][:n]
    return new_arrs


remove_tail = njit(_remove_tail)
