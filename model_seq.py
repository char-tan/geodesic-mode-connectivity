from copy import deepcopy
from typing import List, Tuple, Callable
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelSeq(nn.Module):
    """
    A ModelSeq is a sequnce of models wrapped into a single Module for ease of optimisation
    """

    def __init__(
        self,
        model_factory: Callable,
        n: int,
        state_dict_a: OrderedDict,
        state_dict_b: OrderedDict,
    ):
        """
        generates a sequence of n models between points in param space
        given by state_dict_a and state_dict_b
        """

        super().__init__()

        # Initalise models along path
        lerp_models = []
        for i in range(1, n - 1):
            model = model_factory()
            weights = self._lerp(i / (n + 1), state_dict_a, state_dict_b)
            model.load_state_dict(weights)
            lerp_models.append(model)

        # Initalise end point models, freeze their weights
        model_a = model_factory()
        model_a.load_state_dict(state_dict_a)
        model_a.requires_grad_(requires_grad=False)

        model_b = model_factory()
        model_b.load_state_dict(state_dict_b)
        model_b.requires_grad_(requires_grad=False)

        # All models into a ModuleList
        all_models = [model_a] + lerp_models + [model_b]
        self.models = nn.ModuleList(all_models)

    def _lerp(self, lam: int, state_dict_1: OrderedDict, state_dict_2: OrderedDict):
        """
        linearly interpolates between state_dict_1 and state_dict_2

        returns state_dict at point (1 - lam) * state_dict_1 + lam * state_dict_2
        """

        state_dict_3 = deepcopy(state_dict_2)

        for p in state_dict_1:
            state_dict_3[p] = (1 - lam) * state_dict_1[p] + lam * state_dict_2[p]

        return state_dict_3

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """returns list of outputs for each model on path for input x"""

        outputs = []
        for model in self.models:
            outputs.append(model(x.clone()))

        return outputs

    def euc_path_length(self) -> torch.Tensor:
        """computes euclidean path length (param space) of model seq"""

        total_length = 0

        params_a = self.models[0].parameters()
        model_vec_a = nn.utils.parameters_to_vector(params_a)

        for model in self.models[1:]:

            params_b = model.parameters()
            model_vec_b = nn.utils.parameters_to_vector(params_b)

            total_length += torch.sqrt(((model_vec_a - model_vec_b) ** 2).sum())

            model_vec_a = model_vec_b

        return total_length


def JSD_loss(logits_P: torch.Tensor, logits_Q: torch.Tensor) -> torch.Tensor:

    P = torch.softmax(logits_P, dim=-1)
    Q = torch.softmax(logits_Q, dim=-1)

    M = (P + Q) / 2

    logP, logQ, logM = torch.log(P), torch.log(Q), torch.log(M)

    # PT KL Div is reversed input to math notation
    JSD = (
        F.kl_div(logM, logP, log_target=True, reduction="batchmean")
        + F.kl_div(logM, logQ, log_target=True, reduction="batchmean")
    ) / 2

    return JSD


def metric_path_length(
    outputs: List[torch.Tensor], loss_metric=JSD_loss
) -> Tuple[torch.Tensor]:
    """
    computes distribution space path length using loss_metric

    returns total (sum between each pair of models) and sqrt_total (sum of sqrts)
    """

    total = 0
    sqrt_total = 0

    for i in range(0, len(outputs) - 1):

        length = loss_metric(outputs[i], outputs[i + 1])

        total += length
        sqrt_total += torch.sqrt(length)

    return total, sqrt_total
