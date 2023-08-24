# Adapted from https://github.com/themrzmaster/git-re-basin-pytorch/tree/main

from typing import NamedTuple, Dict
from collections import OrderedDict, defaultdict
import copy

import scipy
import torch
import torch.nn as nn


class PermutationSpec(NamedTuple):
    perm_to_axes: dict
    axes_to_perm: dict


def permutation_spec_from_axes_to_perm(axes_to_perm: Dict) -> PermutationSpec:
    """generates a permutation spec"""

    perm_to_axes = defaultdict(list)
    for wk, axis_perms in axes_to_perm.items():
        for axis, perm in enumerate(axis_perms):
            if perm is not None:
                perm_to_axes[perm].append((wk, axis))

    return PermutationSpec(perm_to_axes=dict(perm_to_axes), axes_to_perm=axes_to_perm)


def get_permuted_param(
    ps: PermutationSpec, perm: Dict, k: str, params: OrderedDict, except_axis=None
) -> torch.Tensor:
    """get parameter k from params, with the permutations applied"""

    w = params[k]

    # Apply the permutaion to the params
    for axis, p in enumerate(ps.axes_to_perm[k]):
        # skip the axis we're trying to permute.
        if axis == except_axis:
            continue

        # None indicates that there is no permutation relevant to that axis.
        if p is not None:
            w = torch.index_select(w, axis, perm[p].int())

    return w


def apply_permutation(
    ps: PermutationSpec, perm: Dict, params: OrderedDict
) -> OrderedDict:
    """apply a permuation to params, returns permuted params"""
    return {k: get_permuted_param(ps, perm, k, params) for k in params.keys()}


def weight_matching(
    ps: PermutationSpec,
    params_a: OrderedDict,
    params_b: OrderedDict,
    max_iter: int = 100,
    init_perm: Dict = None,
) -> Dict:
    """
    use weight matching to find a permutation of params_b to make them match params_a

    returns permutation
    """

    print("\nApplying weight matching to model_b")

    perm_sizes = {
        p: params_a[axes[0][0]].shape[axes[0][1]] for p, axes in ps.perm_to_axes.items()
    }

    perm = (
        {p: torch.arange(n) for p, n in perm_sizes.items()}
        if init_perm is None
        else init_perm
    )

    perm_names = list(perm.keys())

    for iteration in range(max_iter):
        progress = False

        for p_ix in torch.randperm(len(perm_names)):
            p = perm_names[p_ix]
            n = perm_sizes[p]
            A = torch.zeros((n, n))

            for wk, axis in ps.perm_to_axes[p]:
                w_a = params_a[wk]
                w_b = get_permuted_param(ps, perm, wk, params_b, except_axis=axis)
                w_a = torch.moveaxis(w_a, axis, 0).reshape((n, -1))
                w_b = torch.moveaxis(w_b, axis, 0).reshape((n, -1))

                A += w_a @ w_b.T

            ri, ci = scipy.optimize.linear_sum_assignment(
                A.detach().numpy(), maximize=True
            )
            assert (torch.tensor(ri) == torch.arange(len(ri))).all()
            oldL = torch.einsum("ij,ij->i", A, torch.eye(n)[perm[p].long()]).sum()
            newL = torch.einsum("ij,ij->i", A, torch.eye(n)[ci, :]).sum()

            print(f"iteration {iteration} {p}: progress {newL - oldL}")

            progress = progress or newL > oldL + 1e-12

            perm[p] = torch.Tensor(ci)

        if not progress:
            break

    return perm


def permute_model(model_a: nn.Module, model_b: nn.Module, max_iter: int) -> nn.Module:
    """
    permute model_b using weight matching wrt model_a

    returns permuted model
    """

    permuted_model = copy.deepcopy(model_b)

    permutation_spec = model_a.permutation_spec

    # Finds permuation using weight matching
    permutation = weight_matching(
        permutation_spec, model_a.state_dict(), model_b.state_dict(), max_iter=max_iter,
    )

    # Applies permutation
    permuted_params = apply_permutation(
        permutation_spec, permutation, model_b.state_dict()
    )

    permuted_model.load_state_dict(permuted_params)

    return permuted_model
