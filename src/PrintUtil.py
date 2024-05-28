from collections import defaultdict
from itertools import product
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
        table.add_row([der_type_to_str(der_type)] + [(
            f"{results[n].get(der_type_hash, np.nan):_.2f}"
            if results[n].get(der_type_hash, np.nan) < 1e+3
            else f"{results[n].get(der_type_hash, np.nan):_.2E}"
        ) for n in result_names])

    return table


def pretty_results_latex(n: int, k: int, results: Dict[str, DBoundDict]) -> str:

    der_types = generate_derivative_subtypes(np.int16(n), np.int16(k))
    result_names = list(results.keys())

    latex_table = "\\begin{table}[htbp]\n"
    latex_table += "\\centering\n"
    latex_table += "\\caption{Your caption here}\n"
    latex_table += "\\begin{tabular}{l" + "|".join(["c" for _ in result_names]) + "}\n"
    latex_table += "\\toprule\n"
    latex_table += "Type & " + " & ".join(result_names) + " \\\\\n"
    latex_table += "\\midrule\n"

    for der_type in der_types:
        der_type_hash = der_type_to_hash(der_type, np.int16(n), np.int16(k))
        row_values = [
            "\\texttt{\\footnotesize" + (
            f"{results[n].get(der_type_hash, np.nan):_.2f}" if results[n].get(der_type_hash, np.nan) < 1e+3
            else f"{results[n].get(der_type_hash, np.nan):_.2E}"
            ) + "}"
            for n in result_names
        ]
        latex_table += f"{der_type_to_str(der_type)} & {' & '.join(row_values)} \\\\\n"

    latex_table += "\\bottomrule\n"
    latex_table += "\\end{tabular}\n"
    latex_table += "\\label{tab:my_table}\n"
    latex_table += "\\end{table}\n"

    return latex_table


def pretty_max_values(n: int, k: int, results: Dict[str, DBoundDict]) -> PrettyTable:
    der_types = generate_derivative_subtypes(np.int16(n), np.int16(k))
    result_names = list(results.keys())
    table = PrettyTable([""] + list(range(1, n+1)), align="l")

    for name in result_names:
        table.add_row([name] + [(f"{v:_.2f}" if v < 1e+3 else f"{v:_.2E}") for v in [max([
            results[name].get(der_type_to_hash(der_type, np.int16(n), np.int16(k)), np.nan)
            for der_type in der_types if np.sum(der_type) == i
        ]) for i in range(1, n+1)]])

    return table


def pretty_max_values_latex(n: int, k: int, results: Dict[str, DBoundDict]) -> str:
    result_names = list(results.keys())

    latex_table = "\\begin{table}[htbp]\n"
    latex_table += "\\centering\n"
    latex_table += "\\begin{tabular}{l" + "|".join(["c" for _ in range(n)]) + "}\n"
    latex_table += "\\hline\n"
    latex_table += " & " + " & ".join([f"{i}" for i in range(1, n+1)]) + " \\\\\n"
    latex_table += "\\hline\n"

    for name in result_names:
        max_vals = get_max_vals(results[name], n, k)
        row_values = ["\\texttt{\\footnotesize \t" + (
            f"{v:_.2f}" if v < 1e+3 else f"{v:_.2E}"
        ) + "}" for v in max_vals]
        latex_table += f"{name} & {' & '.join(row_values)} \\\\\n"

    latex_table += "\\hline\n"
    latex_table += "\\end{tabular}\n"
    latex_table += "\\caption{Your caption here}\n"
    latex_table += "\\label{tab:my_table}\n"
    latex_table += "\\end{table}\n"

    return latex_table


def get_max_vals(res, n, k):
    der_types = generate_derivative_subtypes(np.int16(n), np.int16(k))
    return [max(
        [
            res.get(der_type_to_hash(der_type, np.int16(n), np.int16(k)), np.nan)
            for der_type in der_types if np.sum(der_type) == i
        ]
    ) for i in range(1, n + 1)]

