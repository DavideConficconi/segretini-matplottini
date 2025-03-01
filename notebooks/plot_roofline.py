from pathlib import Path
from typing import Any

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from segretini_matplottini.plot.roofline import MARKERS, PALETTE, roofline
from segretini_matplottini.utils.plot_utils import save_plot

##############################
# Setup ######################
##############################

PLOT_DIR = (Path(__file__).parent.parent / "plots").resolve()

##############################
# Load data and plot #########
##############################


def load_data_1() -> dict[str, Any]:
    performance = 0.4 * 10**9
    peak_bandwidth = 140.8 * 10**9
    peak_performance = 666 * 10**9
    operational_intensity = 1 / 12
    return dict(
        performance=performance,
        peak_bandwidth=peak_bandwidth,
        peak_performance=peak_performance,
        operational_intensity=operational_intensity,
    )


def plot_1(data_dict: dict[str, Any]) -> tuple[plt.Figure, plt.Axes]:
    return roofline(
        data_dict["performance"],
        data_dict["operational_intensity"],
        data_dict["peak_performance"],
        data_dict["peak_bandwidth"],
        add_legend=True,
        legend_labels="CPU",
    )


def load_data_2() -> dict[str, Any]:
    packet_size = 15
    bandwidth_single_core = 13.2 * 10**9
    max_cores = 64
    clock = 225 * 10**6
    packet_sizes = [packet_size] * 4
    num_cores = [1, 8, 16, 32]
    peak_performance = [p * clock * max_cores for p in packet_sizes]
    peak_bandwidth = [bandwidth_single_core * c for c in num_cores]
    operational_intensity = [p / (512 / 8) for p in packet_sizes]
    exec_times_fpga = [108, 12, 5, 3]
    performance = [20 * 10**7 / (p / 1000) for p in exec_times_fpga]
    num_cores = [1, 8, 16, 32]
    return dict(
        performance=performance,
        peak_bandwidth=peak_bandwidth,
        peak_performance=peak_performance,
        operational_intensity=operational_intensity,
        num_cores=num_cores,
    )


def plot_2(data_dict: dict[str, Any]) -> tuple[plt.Figure, plt.Axes]:
    return roofline(
        data_dict["performance"],
        data_dict["operational_intensity"],
        data_dict["peak_performance"],
        data_dict["peak_bandwidth"],
        palette=PALETTE,
        markers=MARKERS,
        performance_unit="FLOPS",
        xmin=0.01,
        xmax=20,
        add_legend=True,
        legend_labels=[f"{c} Cores" for c in data_dict["num_cores"]],
    )


def load_data_3() -> tuple[dict[str, Any], dict[str, Any]]:
    packet_size = 15
    max_cores = 64
    clock = 225 * 10**6
    performance = [0.4 * 10**9, 27 * 10**9, 20 * 10**7 / (3 / 1000)]
    operational_intensity = [1 / 12, 1 / 12, 0.23]
    peak_performance = [666 * 10**9, 3100 * 10**9, packet_size * clock * max_cores]
    peak_bandwidth = [140.8 * 10**9, 549 * 10**9, 422.4 * 10**9]
    return load_data_2(), dict(
        performance=performance,
        peak_bandwidth=peak_bandwidth,
        peak_performance=peak_performance,
        operational_intensity=operational_intensity,
    )


def plot_3(data_dict_1: dict[str, Any], data_dict_2: dict[str, Any]) -> tuple[plt.Figure, plt.Axes]:
    num_col = 2
    num_row = 1
    fig = plt.figure(figsize=(2.2 * num_col, 1.9 * num_row))
    gs = gridspec.GridSpec(num_row, num_col)
    plt.subplots_adjust(top=0.95, bottom=0.2, left=0.1, right=0.92, hspace=0, wspace=0.4)
    # First roofline
    ax = fig.add_subplot(gs[0, 0])
    ax = roofline(
        data_dict_1["performance"],
        data_dict_1["operational_intensity"],
        data_dict_1["peak_performance"],
        data_dict_1["peak_bandwidth"],
        palette=PALETTE,
        markers=MARKERS,
        ax=ax,
        performance_unit="FLOPS",
        xmin=0.01,
        xmax=20,
        add_legend=True,
        legend_labels=[f"{c} Cores" for c in data_dict_1["num_cores"]],
    )
    # Second roofline
    ax = fig.add_subplot(gs[0, 1])
    ax = roofline(
        data_dict_2["performance"],
        data_dict_2["operational_intensity"],
        data_dict_2["peak_performance"],
        data_dict_2["peak_bandwidth"],
        palette=PALETTE,
        markers=MARKERS,
        ax=ax,
        performance_unit="FLOPS",
        xmin=0.01,
        xmax=20,
        ylabel="",
        add_legend=True,
        legend_labels=["CPU", "GPU", "FPGA"],
        add_bandwidth_label=False,
        add_operational_intensity_label=False,
    )
    return fig, ax


##############################
# Main #######################
##############################

if __name__ == "__main__":

    # Create a single Roofline model;
    data_dict = load_data_1()
    fig, ax = plot_1(data_dict)
    save_plot(PLOT_DIR, "roofline.{}")

    # Create a fancier Roofline model, with multiple lines and custom settings;
    data_dict = load_data_2()
    fig, ax = plot_2(data_dict)
    save_plot(PLOT_DIR, "roofline_stacked.{}")

    # Create a single plot composed of 2 separate Rooflines;
    data_dict_1, data_dict_2 = load_data_3()
    fig, ax = plot_3(data_dict_1, data_dict_2)
    save_plot(PLOT_DIR, "roofline_double.{}")
