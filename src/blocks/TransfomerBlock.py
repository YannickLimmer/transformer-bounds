import numba as nb
import numpy as np

from src.DerivativeBounds import make_dbound_dict, compute_bound_for_alpha, adjust_bounds
from src.DerivativeTypes import generate_derivative_subtypes
from src.Hashing import der_types_to_hashes
from src.PrintUtil import pretty_max_values, get_max_vals
from src.blocks.Attention import der_bounds_multi_head_attention
from src.blocks.LayerNorm import der_bounds_layer_norm
from src.blocks.NeuralNet import der_bounds_neural_network


def der_bounds_tblock(
        n: nb.int16,
        input_dim: nb.int16,
        output_dim: nb.int16,
        sequence_length: nb.int16,
        ndim_k: nb.int16,
        ndim_v: nb.int16,
        ndim_nn: nb.int16,
        max_weight_k: nb.float64,
        max_weight_q: nb.float64,
        max_weight_v: nb.float64,
        max_weight_w: nb.float64,
        max_weight_ln1: nb.float64,
        max_weight_a: nb.float64,
        max_weight_b: nb.float64,
        max_weight_ln2: nb.float64,
        domain_bound: nb.float64,
        activation_type: str = 'softplus',
        verbose: int = 0,
        by_level: bool = False,
):
    k = sequence_length * input_dim
    der_types = generate_derivative_subtypes(n, k)
    hashes = der_types_to_hashes(der_types, n, k)

    if verbose > 0:
        print("Computing Bound for Multi-Head Attention")
    mha_bounds = der_bounds_multi_head_attention(
        n,
        input_dim,
        sequence_length,
        ndim_k,
        ndim_v,
        max_weight_k,
        max_weight_q,
        max_weight_v,
        max_weight_w,
        domain_bound,
        by_level,
    )
    if verbose > 1:
        print(pretty_max_values(n, k, dict(tblock=mha_bounds)))

    if verbose > 0:
        print("Computing Bound for Layer Norm 1")
    ln1_bounds = der_bounds_layer_norm(n, input_dim, domain_bound, max_weight_ln1, by_level)
    if verbose > 1:
        print(pretty_max_values(n, input_dim, dict(tblock=ln1_bounds)))

    if verbose > 0:
        print("Computing Bound for Neural Net")
    slp_bounds = der_bounds_neural_network(n, input_dim, ndim_nn, max_weight_a, max_weight_b, activation_type, by_level)
    if verbose > 1:
        print(pretty_max_values(n, input_dim, dict(tblock=slp_bounds)))

    if verbose > 0:
        print("Computing Bound for Layer Norm 2")
    ln2_bounds = der_bounds_layer_norm(n, output_dim, domain_bound, max_weight_ln2, by_level)
    if verbose > 1:
        print(pretty_max_values(n, output_dim, dict(tblock=ln2_bounds)))

    if verbose > 0:
        print("Combining Results 1")
    bounds = np.zeros(len(hashes), dtype=np.float64)
    for i in range(len(der_types)):
        bounds[i] = compute_bound_for_alpha(n, input_dim, k, der_types[i], ln1_bounds, mha_bounds)
    der_bounds_2 = make_dbound_dict(hashes, adjust_bounds(bounds, der_types, by_level))
    if verbose > 1:
        print(pretty_max_values(n, k, dict(tblock=der_bounds_2)))

    if verbose > 0:
        print("Combining Results 2")
    bounds = np.zeros(len(hashes), dtype=np.float64)
    for i in range(len(der_types)):
        bounds[i] = compute_bound_for_alpha(n, input_dim, k, der_types[i], slp_bounds, der_bounds_2)
    der_bounds_3 = make_dbound_dict(hashes, adjust_bounds(bounds, der_types, by_level))
    if verbose > 1:
        print(pretty_max_values(n, k, dict(tblock=der_bounds_3)))

    if verbose > 0:
        print("Combining Results 3")
    bounds = np.zeros(len(hashes), dtype=np.float64)
    for i in range(len(der_types)):
        bounds[i] = compute_bound_for_alpha(n, output_dim, k, der_types[i], ln2_bounds, der_bounds_3)
    final_bounds = make_dbound_dict(hashes, adjust_bounds(bounds, der_types, by_level))
    return final_bounds, [
        get_max_vals(mha_bounds, n, k),
        get_max_vals(ln1_bounds, n, input_dim),
        get_max_vals(slp_bounds, n, input_dim),
        get_max_vals(ln2_bounds, n, output_dim),
        get_max_vals(final_bounds, n, k),
    ]
