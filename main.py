import argparse

import numpy as np
import torch

from resnet import ResNet
from training import train_experiment
from model_permutation import permute_model
from geodesic_opt import optimise_model_seq, eval_path_loss
from plotting import plot_path_dist, plot_path_loss
from model_seq import ModelSeq
from data import get_dataloaders


def get_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument("--wm", type=int, default=4, help="resnet width multiplier")

    # Data
    parser.add_argument("--crop_size", type=int, default=32, help="image crop size")
    parser.add_argument(
        "--padding", type=int, default=4, help="image pre-crop padding amount"
    )

    # Base Training
    parser.add_argument(
        "--base_training",
        action="store_true",
        help="train new base models from initalisation",
    )
    parser.add_argument(
        "--base_num_epochs",
        type=int,
        default=100,
        help="number of epochs (base training)",
    )
    parser.add_argument(
        "--base_batch_size", type=int, default=128, help="batch size (base training)"
    )
    parser.add_argument(
        "--base_lr", type=float, default=1e-1, help="learning rate (base training)"
    )
    parser.add_argument(
        "--base_div_factor",
        type=float,
        default=1e9,
        help="start div factor for OneCycleLR (base training)",
    )
    parser.add_argument(
        "--base_final_div_factor",
        type=float,
        default=1e6,
        help="end div factor for OneCycleLR (base training)",
    )
    parser.add_argument(
        "--base_pct_start",
        type=float,
        default=0.1,
        help="pct_start for OneCycleLR (base training)",
    )
    parser.add_argument(
        "--base_momentum",
        type=float,
        default=0.9,
        help="SGD momentum value (base training)",
    )
    parser.add_argument(
        "--base_weight_decay",
        type=float,
        default=1e-4,
        help="weight decay (base training)",
    )

    # Geodesic Optimisation
    parser.add_argument(
        "--geodesic_N", type=int, default=25, help="number of models in model sequence"
    )
    parser.add_argument(
        "--geodesic_num_epochs",
        type=int,
        default=20,
        help="number of epochs (geodesic optimisation)",
    )
    parser.add_argument(
        "--geodesic_batch_size",
        type=int,
        default=256,
        help="batch size (geodesic optimisation",
    )
    parser.add_argument(
        "--geodesic_lr",
        type=float,
        default=1e-1,
        help="learning rate (geodesic optimisation",
    )

    # Device
    use_cuda = torch.cuda.is_available()
    default_device = "cuda" if use_cuda else "cpu"
    parser.add_argument("--device", type=str, default=default_device)

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = get_args()

    if args.base_training:

        # Train 2 models
        model_a = train_experiment(args, 0)
        model_b = train_experiment(args, 1)

        # Perform weight matching and permute model b
        model_b_p = permute_model(model_a, model_b, max_iter=30)

    else:

        # Initalise models
        model_a = ResNet(args.wm)
        model_b_p = ResNet(args.wm)

        # Load saved weights
        model_a.load_state_dict(torch.load("model_files/resnet_wm4_a.pt"))
        model_b_p.load_state_dict(torch.load("model_files/resnet_wm4_b_permuted.pt"))

    torch.manual_seed(2)

    # Generate model sequence
    model_seq = ModelSeq(
        lambda: ResNet(args.wm),
        args.geodesic_N,
        model_a.state_dict(),
        model_b_p.state_dict(),
    ).to(args.device)

    train_loader, test_loader = get_dataloaders(
        args.crop_size, args.padding, args.geodesic_batch_size
    )

    # Evaluate linear path
    print("\nPath evaluation on train data")
    start_losses_train = eval_path_loss(model_seq, train_loader, args.device)
    print("Path evaluation on test data\n")
    start_losses_test = eval_path_loss(model_seq, test_loader, args.device)

    # perform geodesic optimisation
    d_space_sqrt_JSDs, p_space_eucs = optimise_model_seq(
        model_seq,
        train_loader,
        torch.optim.SGD(model_seq.parameters(), args.geodesic_lr),
        args.geodesic_lr,
        args.geodesic_num_epochs,
        args.device,
    )

    # Evaluate geodesic path
    print("path evaluation on train data")
    end_losses_train = eval_path_loss(model_seq, train_loader, args.device)
    print("path evaluation on test data\n")
    end_losses_test = eval_path_loss(model_seq, test_loader, args.device)

    # Plotting
    plot_path_dist(d_space_sqrt_JSDs, p_space_eucs)
    plot_path_loss(
        start_losses_train, start_losses_test, end_losses_train, end_losses_test
    )
