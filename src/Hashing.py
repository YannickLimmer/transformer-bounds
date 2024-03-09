import numba as nb
import numpy as np
from numba import njit
from numba.typed import List


def _der_type_to_hash(arr: nb.int16[:], n: int, k: int) -> nb.int64:
    score: nb.int64 = 0
    base: nb.int64 = 1
    for i in range(k):
        if arr[i] == 0:
            return score

        score += arr[i] * base
        base *= n

    return score


der_type_to_hash = njit(_der_type_to_hash)


def _der_types_to_hashes(arrs: List[nb.int16[:]], n: int, k: int) -> nb.int64[:]:
    hashes = np.zeros(len(arrs), dtype=np.int64)
    for i in range(len(arrs)):
        hashes[i] = der_type_to_hash(arrs[i], n, k)
    return hashes


der_types_to_hashes = njit(_der_types_to_hashes)
