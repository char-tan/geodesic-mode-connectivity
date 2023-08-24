import argparse
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from data import get_dataloaders
from resnet import ResNet


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    opt: torch.optim.Optimizer,
    lr_sched: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    device: str,
) -> Tuple[np.ndarray]:
    """train model for one epoch, returns mean loss and accuracy"""

    model.train()

    train_loss, train_acc = [], []

    for inputs, targets in dataloader:

        inputs, targets = inputs.to(device), targets.to(device)

        opt.zero_grad(set_to_none=True)

        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        train_loss.append(loss.detach().item())
        train_acc.append((outputs.detach().argmax(-1) == targets).float().mean().item())

        loss.backward()
        opt.step()

        lr_sched.step()

    return np.mean(train_loss), np.mean(train_acc)


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    epoch: int,
    device: str,
) -> Tuple[np.ndarray]:
    """evaluate model on dataloader, returns mean loss and accuracy"""

    model.eval()

    eval_loss, eval_acc = [], []

    for inputs, targets in dataloader:

        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        eval_loss.append(loss.detach().item())
        eval_acc.append((outputs.detach().argmax(-1) == targets).float().mean().item())

    return np.mean(eval_loss), np.mean(eval_acc)


def train_experiment(args: argparse.Namespace, seed: int) -> nn.Module:
    """trains and returns a model"""

    torch.manual_seed(seed)

    train_loader, test_loader = get_dataloaders(
        args.crop_size, args.padding, args.base_batch_size
    )

    model = ResNet(wm=args.wm).to(args.device)

    opt = torch.optim.SGD(
        model.parameters(),
        args.base_lr,
        weight_decay=args.base_weight_decay,
        momentum=args.base_momentum,
        nesterov=True,
    )

    lr_sched = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=args.base_lr,
        total_steps=len(train_loader) * args.base_num_epochs,
        pct_start=args.base_pct_start,
        div_factor=args.base_div_factor,
        final_div_factor=args.base_final_div_factor,
    )

    loss_fn = nn.CrossEntropyLoss()

    print(f"\nTraining model with seed {seed}")

    for epoch in range(args.base_num_epochs):

        train_loss, train_acc = train_epoch(
            model, train_loader, loss_fn, opt, lr_sched, epoch, args.device
        )

        print(
            f"epoch {epoch:03d} | train_loss {train_loss:.5f} | train_acc {train_acc:.5f}"
        )

    eval_loss, eval_acc = eval_epoch(model, test_loader, loss_fn, epoch, args.device)

    print(f"eval      | eval_loss  {eval_loss:.5f} | eval_acc  {eval_acc:.5f}\n")

    return model.cpu()
