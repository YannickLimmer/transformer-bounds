from typing import List, Dict

import numpy as np
from prettytable import PrettyTable

from src.DerivativeBounds import DBoundDict
from src.DerivativeTypes import generate_derivative_subtypes
from src.Hashing import der_type_to_hash


def der_type_to_str(der_type: np.array):
    result = "("
    for i, val in enumerate(der_type):
        result += f", {val}" if i > 0 else f"{val}"
        if val == 0:
            if i < len(der_type) - 1:
                result += ", ..."
            break
    return result + ")"


def pretty_results(n: int, k: int, results: Dict[str, DBoundDict]) -> PrettyTable:

    der_types = generate_derivative_subtypes(np.int16(n), np.int16(k))
    result_names = list(results.keys())
    table = PrettyTable(["Type"] + result_names, align="l")

    for der_type in der_types:
        der_type_hash = der_type_to_hash(der_type, np.int16(n), np.int16(k))
        table.add_row([der_type_to_str(der_type)] + [f"{results[n].get(der_type_hash, np.nan):_.2f}" for n in result_names])

    return table

