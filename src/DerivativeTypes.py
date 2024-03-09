import numba as nb
import numpy as np
from numba import njit
from numba.typed import List

from src.Util import fill_zeros


def _generate_derivative_types(n: np.int16, k: np.int16) -> List[nb.int16[::1]]:
    cur_arrs = List([np.zeros(min(n, k), dtype=np.int16)])
    cur_targets = List(np.array([n], dtype=np.int16))
    for i in range(min(n, k)):
        if sum(cur_targets) == 0:
            break
        next_arrs = List.empty_list(nb.int16[::1])
        next_targets = List.empty_list(nb.int16)
        for j in range(len(cur_arrs)):
            min_val = (cur_targets[j] + (k - i - 1)) // (k - i)
            max_val = cur_targets[j] if i == 0 else min(cur_arrs[j][i - 1], cur_targets[j])
            for val in range(min_val, max_val + 1):
                next_arr = cur_arrs[j].copy()
                next_arr[i] = val
                next_arrs.append(next_arr)
                next_targets.append(np.int16(cur_targets[j] - val))
        cur_arrs = next_arrs
        cur_targets = next_targets

    return fill_zeros(cur_arrs, k)


generate_derivative_types = njit(_generate_derivative_types)

