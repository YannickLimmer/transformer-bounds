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


def pretty_results(n: int, k: int, results: DBoundDict) -> PrettyTable:

    der_types = generate_derivative_subtypes(np.int16(n), np.int16(k))
    table = PrettyTable(["Type"] + ['Bound'], align="l")

    for der_type in der_types:
        der_type_hash = der_type_to_hash(der_type, np.int16(n), np.int16(k))
        if der_type_hash in results.keys():
            table.add_row([der_type_to_str(der_type), f"{results[der_type_hash]:0.2f}"])

    return table

