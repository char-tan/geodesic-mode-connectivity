import matplotlib.pyplot as plt
import numpy as np


def plot_path_dist(d_space_sqrt_JSDs: np.ndarray, p_space_eucs: np.ndarray):

    fig, ax = plt.subplots()

    ax.plot(d_space_sqrt_JSDs, color="red")
    ax.set_xlabel("ITERATION", fontsize=14)
    ax.set_ylabel("SUM SQRT JSD", color="red", fontsize=14)

    ax2 = ax.twinx()
    ax2.plot(p_space_eucs, color="blue", linestyle="dashed")
    ax2.set_ylabel("EUCLIDEAN DISTANCE", color="blue", fontsize=14)
    plt.savefig("path_dist.png")


def plot_path_loss(
    start_losses_train: np.ndarray,
    start_losses_test: np.ndarray,
    end_losses_train: np.ndarray,
    end_losses_test: np.ndarray,
):

    fig, ax = plt.subplots()

    plt.plot(
        end_losses_train,
        linewidth=2,
        color="orange",
        marker="o",
        label="train (geodesic)",
    )
    plt.plot(
        end_losses_test,
        linewidth=2,
        color="orange",
        marker="|",
        label="test (geodesic)",
    )
    plt.plot(
        start_losses_train,
        linewidth=2,
        color="grey",
        linestyle="dashed",
        label="train (linear)",
    )
    plt.plot(
        start_losses_test,
        linewidth=2,
        color="grey",
        linestyle="dotted",
        label="test (linear)",
    )

    ax.set_xlabel("MODEL INDEX", fontsize=14)
    ax.set_ylabel("CROSS ENTROPY LOSS", fontsize=14)

    ax.legend()

    plt.savefig("path_loss.png")
