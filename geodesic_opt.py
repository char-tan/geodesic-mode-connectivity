from typing import Tuple
import argparse

import numpy as np
import torch
import torch.nn as nn

from model_seq import ModelSeq, metric_path_length


@torch.no_grad()
def eval_path_loss(
    model_seq: ModelSeq, dataloader: torch.utils.data.DataLoader, device: str
) -> np.ndarray:
    """evaluates each model in model_seq, returns list of average losses"""

    loss_fn = torch.nn.CrossEntropyLoss()

    total_losses = []

    i = 0

    for inputs, targets in dataloader:

        i += 1

        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model_seq(inputs)

        # get loss of each model wrt to true target
        batch_losses = [
            loss_fn(single_output, targets).item() for single_output in outputs
        ]
        total_losses.append(batch_losses)

    # average over output
    total_losses = torch.tensor(total_losses).mean(axis=0)

    return total_losses.cpu().numpy()


def optimise_model_seq(
    model_seq: ModelSeq,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    lr: int,
    num_epochs: int,
    device: str,
) -> Tuple[np.ndarray]:
    """optimise a model_seq using data from dataloader"""

    p_space_eucs = []  # Param space euclidean distance of seq
    d_space_sqrt_JSDs = []  # Distribution space sqrt JSD of seq

    print("Optimising model sequence")

    for epoch_idx in range(num_epochs):

        for i, (inputs, targets) in enumerate(dataloader):

            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model_seq(inputs)

            # Measure d space JSD length (both normal and sqrt)
            d_space_JSD, d_space_sqrt_JSD = metric_path_length(outputs)
            d_space_sqrt_JSDs.append(d_space_sqrt_JSD.detach().cpu().numpy())

            # Measure p space euclidean distance
            p_space_euc = model_seq.euc_path_length()
            p_space_eucs.append(p_space_euc.detach().cpu().numpy())

            print(
                f"batch {i+1:03d} | p_space_euc {p_space_euc:.5f} | d_space_sqrt_JSD {d_space_sqrt_JSD:.5f}"
            )

            # Optimize wrt to non-squared JSD loss
            d_space_JSD.backward()
            optimizer.step()

        print(
            f"\nepoch {epoch_idx+1:03d} | p_space_euc {np.mean(p_space_eucs):.5f} | d_space_sqrt_JSD {np.mean(d_space_sqrt_JSDs):.5f}\n"
        )

    return d_space_sqrt_JSDs, p_space_eucs
