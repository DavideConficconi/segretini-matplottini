# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 09:53:24 2021

@author: albyr
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib import colors
from matplotlib.dates import YearLocator, MonthLocator, num2date
from datetime import datetime
from matplotlib.ticker import FuncFormatter
from sklearn import linear_model
import matplotlib.ticker as ticker

import sys
sys.path.append("..")
from plot_utils import *

##############################
##############################

# # Color palette used for plotting;
PALETTE_RD_CMP = ["#D01C8B", "#F1B6DA"]
MARKERS_RD_CMP = ["o", "X"]
MARKERS_WR_CMP = ["o", "X"]
PALETTE_WR_CMP = ["#4DAC26", "#B8E186"]
#["#543005","#8c510a","#bf812d","#dfc27d","#f6e8c3","#c7eae5","#80cdc1","#35978f","#01665e","#003c30"]
#["#8e0152","#c51b7d","#de77ae","#f1b6da","#fde0ef","#e6f5d0","#b8e186","#7fbc41","#4d9221","#276419"]
#["#a50026","#d73027","#f46d43","#fdae61","#fee08b","#d9ef8b","#a6d96a","#66bd63","#1a9850","#006837"]
#["#67001f","#b2182b","#d6604d","#f4a582","#fddbc7","#e0e0e0","#bababa","#878787","#4d4d4d","#1a1a1a"]
#["#8e0152","#c51b7d","#de77ae","#f1b6da","#fde0ef","#e6f5d0","#b8e186","#7fbc41","#4d9221","#276419"]
#["#40004b","#762a83","#9970ab","#c2a5cf","#e7d4e8","#d9f0d3","#a6dba0","#5aae61","#1b7837","#00441b"]
PALETTE_RD = ["#40004b","#762a83","#9970ab","#c2a5cf","#e7d4e8","#f7f7f7","#d9f0d3","#a6dba0","#5aae61","#1b7837","#00441b","#40004b","#762a83"]
#PALETTE_RD = sns.color_palette("colorblind",13)
MARKERS_RD = [".","8", "p","*","h","H","P","X","D","d","<","v","s" ]
# ["#67001f","#b2182b","#d6604d","#f4a582","#fddbc7","#e0e0e0","#bababa","#878787","#4d4d4d","#1a1a1a"]
PALETTE_WR = ["#40004b","#762a83","#9970ab","#c2a5cf","#e7d4e8","#f7f7f7","#d9f0d3","#a6dba0","#5aae61","#1b7837","#00441b","#40004b","#762a83"]
#MARKERS_WR = ["o","|","1","2","v","^","+","x","3","4"]
#MARKERS_WR = [".","8", "p","*","h","H","P","X","D","d" ]
MARKERS_WR = [".","8", "p","*","h","H","P","X","D","d","<","v","s" ]
#MARKERS_WR = ["_","|","1","x","4",0,3,"3","+","2"]

# # Axes limits used in the plot, change them accordingy to your data;
X_LIMITS = (2,34)
Y_LIMITS = (0, 85)
X_LIMITS_SINGLE = (0,34)
legend_fontsize=6
ax_fontsize=8
markerscale=0.8
##############################
##############################

def performance_scaling(data: pd.DataFrame,
                        set_axes_limits: bool=True,
                        plot_regression: bool=True) -> (plt.Figure, plt.Axes):
    """
    Parameters
    ----------
    data : pd.DataFrame with 6 columns:
        "year",
        "performance",
        "kind" ∈ ["compute", "memory", "interconnect"],
        "name" (label shown in the plot, it can be empty),
        "base" (base value used for speedup, it can be empty),
        "comment" (e.g. data source or non-used label, it can be empty).

    Returns
    -------
    fig : matplotlib figure containing the plot
    ax : matplotlib axis containing the plot
    """
    ##############
    # Plot setup #
    ##############
    
    # Reset matplotlib settings;
    plt.rcdefaults()
    # Setup general plotting settings;
    sns.set_style("white", {"ytick.left": True, "xtick.bottom": True})
    plt.rcParams["font.family"] = ["Latin Modern Roman Demi"]
    plt.rcParams['axes.labelpad'] = 0  # Padding between axis and axis label;
    plt.rcParams['xtick.major.pad'] = 1  # Padding between axis ticks and tick labels;
    plt.rcParams['ytick.major.pad'] = 1  # Padding between axis ticks and tick labels;
    plt.rcParams['axes.linewidth'] = 0.8  # Line width of the axis borders;
    
    # Create a figure for the plot, and adjust margins;
    fig = plt.figure(figsize=(6, 3))
    gs = gridspec.GridSpec(1, 1)
    plt.subplots_adjust(top=0.98,
                        bottom=0.1,
                        left=0.12,
                        right=0.99)  
    ax = fig.add_subplot(gs[0, 0])
    
    # Set axes limits;        
    if set_axes_limits:
        ax.set_xlim(X_LIMITS)
        ax.set_ylim(Y_LIMITS)
    
    #################
    # Main plot #####
    #################    

    # Measure performance increase over 20 and 2 years;
    kind_increase = {}      
    
    # Add a scatterplot for individual elements of the dataset, and change color based on hardware type;
    ax = sns.scatterplot(x="PowOfTwoBytes", y="AVG_RD_BW[Gbit/s]", hue="TestVersion", style="TestVersion", palette=PALETTE_RD_CMP, markers=MARKERS_RD_CMP, s=15,
                      data=data, ax=ax, edgecolor="#2f2f2f", linewidth=0.5, zorder=3)
    ax = sns.scatterplot(x="PowOfTwoBytes", y="AVG_WR_BW[Gbit/s]", hue="TestVersion", style="TestVersion", palette=PALETTE_WR_CMP, markers=MARKERS_WR_CMP, s=15,
                      data=data, ax=ax, edgecolor="#2f2f2f", linewidth=0.5, zorder=10)

    # Add a regression plot to highlight the correlation between variables, with 95% confidence intervals;
    # if plot_regression:
    #     for i, (kind, g) in enumerate(data.groupby("version", sort=False)):            
    #         data_tmp = g.copy()
    #         # We fit a straight line on the log of the relative performance, as the scaling is exponential.
    #         # Then, the real prediction is 10**prediction;
    #         regr = linear_model.LinearRegression()
    #         regr.fit(data_tmp["bytes_power"].values.reshape(-1, 1), np.log10(data_tmp["AVG_RD_BW[Gbit/s]"].values.reshape(-1, 1)))
    #         data_tmp["prediction"] = np.power(10, regr.predict(data_tmp["bytes_power"].values.astype(float).reshape(-1, 1)))
    #         ax = sns.lineplot(x=[data_tmp["bytes_power"].iloc[0], data_tmp["bytes_power"].iloc[-1]],
    #                           y=[data_tmp["prediction"].iloc[0], data_tmp["prediction"].iloc[-1]],
    #                           color=PALETTE_RD[i], ax=ax, alpha=0.5, linewidth=6)
            
    #         # Use the regression line to obtain the slope over 2 and 10 years;
    #         slope = (np.log10(data_tmp["prediction"].iloc[-1]) - np.log10(data_tmp["prediction"].iloc[0])) / ((data_tmp["bytes_power"].iloc[-1] - data_tmp["year"].iloc[0]).days / 365)
    #         slope_2_years = 10**(slope * 2)
    #         slope_20_years = 10**(slope * 20)
    #         kind_increase[kind] = (slope_2_years, slope_20_years)
    # ax.legend_.remove()  # Hack to remove legend;

    #####################
    # Add labels ########
    #####################
    
    # Associate a color to each kind of hardware (compute, memory, interconnection)
    # def get_color(c):  # Make the color darker, to use it for text;
    #     hue, saturation, brightness = colors.rgb_to_hsv(colors.to_rgb(c))
    #     return sns.set_hls_values(c, l=brightness * 0.6, s=saturation * 0.7)
    #kind_to_col = {k: get_color(PALETTE_RD[i]) for i, k in enumerate(data["Brst_size[#beats]"].unique())}
    
    #data["name"] = data["name"].fillna("")
    #for i, row in data.iterrows():
    #    label = row["name"]
        # Label-specific adjustments;
        # if label:
        #     if label ==  "Pentium II Xeon":
        #         xytext = (5, -9)
        #     elif label ==  "PCIe 4.0":
        #         xytext = (5, -9)
        #     elif label ==  "Radeon Fiji":
        #         xytext = (-7, 5)
        #     elif label ==  "TPUv2":
        #         xytext = (-7, 5)
        #     elif row["kind"] == "interconnect":
        #         xytext = (0, -9)
        #     else:
    #        xytext = (0, 5)
    #        ax.annotate(label, xy=(row["bytes_power"], row["performance"]), size=7, xytext=xytext,
    #                    textcoords="offset points", ha="center", color=kind_to_col[row["version"]])
    
    #####################
    # Style fine-tuning #
    #####################
    
    # Log-scale y-axis;
    #plt.yscale("log")
    
    # Turn on the grid;
    ax.yaxis.grid(True, linewidth=0.3)
    ax.xaxis.grid(True, linewidth=0.3)
    
    # Set tick number and parameters on x and y axes;
    def year_formatter(x, pos=None):
        d = num2date(x)
        if (d.year - X_LIMITS[0].year) % 3 != 0:
            return ""
        else:
            return d.year
    tick_spacing = 2
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    #ax.xaxis.set_minor_locator(MonthLocator(interval=3))
    #ax.xaxis.set_major_formatter(FuncFormatter(year_formatter))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.tick_params(axis="x", direction="out", which="both", bottom=True, top=False, labelsize=7, width=0.5, size=5)
    ax.tick_params(axis="x", direction="out", which="minor", size=2)  # Update size of minor ticks;
    ax.tick_params(axis="y", direction="out", which="both", left=True, right=False, labelsize=7, width=0.5, size=5)
    ax.tick_params(axis="y", direction="out", which="minor", size=2)  # Update size of minor ticks;
    
    # Ticks, showing relative performance;
    def format_speedup(l):
        if l >= 1:
            return str(int(l))
        else:
            return f"{l:.1f}"
    #ax.set_yticklabels(labels=[format_speedup(l) + r"$\mathdefault{\times}$" for l in ax.get_yticks()], ha="right", fontsize=7)
 
    # Add a fake legend with summary data.
    # We don't use a real legend as we need rows with different colors and we don't want patches on the left.
    # Also, we want the text to look justified.
    def get_kind_label(k):
        kind_name = ""
        if k == "compute":
            kind_name = "HW FLOPS"
        elif k == "memory":
            kind_name = "DRAM BW"
        else:
            kind_name = "Interconnect BW"
        return kind_name
    # Create a rectangle used as background;
    #rectangle = {"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "#B8B8B8", "linewidth": 0.5, "pad": 0.5}
    #for i, (k, v) in enumerate(kind_increase.items()):
       # pad = " " * 48 + "\n\n"  # Add padding to first label, to create a large rectangle that covers other labels; 
        # Use two annotations, to make the text look justified;
        #ax.annotate(get_kind_label(k) + ":" + (pad if i == 0 else ""), xy=(0.023, 0.94 - 0.05 * i),
        #            xycoords="axes fraction", fontsize=7, color=kind_to_col[k], ha="left", va="top", bbox=rectangle if i == 0 else None)
        #ax.annotate(f"{v[1]:.0f}" + r"$\mathdefault{\times}$" + f"/20 years ({v[0]:.1f}" + r"$\mathdefault{\times}$"+ "/2 years)",
        #            xy=(0.43, 0.941 - 0.05 * i), xycoords="axes fraction", fontsize=7, color=kind_to_col[k], ha="right", va="top")
        
    # Add axes labels;
    plt.ylabel("Performance Scaling [Gbit/s]", fontsize=ax_fontsize)
    plt.xlabel("Byte Size (2^x)", fontsize=ax_fontsize)
    plt.legend(title="Version",title_fontsize=legend_fontsize+1,loc="lower right", frameon=True, fontsize=legend_fontsize, markerscale=markerscale)
    
    return fig, ax
##############################
##############################

def mem_bw_read(data: pd.DataFrame,
                        set_axes_limits: bool=True,
                        plot_regression: bool=True) -> (plt.Figure, plt.Axes):
    """
    Parameters
    ----------
    data : pd.DataFrame with 6 columns:
        "year",
        "performance",
        "kind" ∈ ["compute", "memory", "interconnect"],
        "name" (label shown in the plot, it can be empty),
        "base" (base value used for speedup, it can be empty),
        "comment" (e.g. data source or non-used label, it can be empty).

    Returns
    -------
    fig : matplotlib figure containing the plot
    ax : matplotlib axis containing the plot
    """
    data.rename(columns={'Brst_size[#beats]': 'Burst Size'}, inplace=True)
    
    ##############
    # Plot setup #
    ##############
    
    # Reset matplotlib settings;
    plt.rcdefaults()
    # Setup general plotting settings;
    sns.set_style("white", {"ytick.left": True, "xtick.bottom": True})
    plt.rcParams["font.family"] = ["Latin Modern Roman Demi"]
    plt.rcParams['axes.labelpad'] = 0  # Padding between axis and axis label;
    plt.rcParams['xtick.major.pad'] = 1  # Padding between axis ticks and tick labels;
    plt.rcParams['ytick.major.pad'] = 1  # Padding between axis ticks and tick labels;
    plt.rcParams['axes.linewidth'] = 0.8  # Line width of the axis borders;
    
    # Create a figure for the plot, and adjust margins;
    fig = plt.figure(figsize=(6, 3))
    gs = gridspec.GridSpec(1, 1)
    plt.subplots_adjust(top=0.98,
                        bottom=0.1,
                        left=0.12,
                        right=0.99)  
    ax = fig.add_subplot(gs[0, 0])
    
    # Set axes limits;        
    if set_axes_limits:
        ax.set_xlim(X_LIMITS_SINGLE)
        ax.set_ylim(Y_LIMITS)
    
    #################
    # Main plot #####
    #################    

    # Measure performance increase over 20 and 2 years;
    kind_increase = {}      
    
    # Add a scatterplot for individual elements of the dataset, and change color based on hardware type;
    ax = sns.scatterplot(x="PowOfTwoBytes", y="AVG_RD_BW[Gbit/s]", hue="Burst Size", style="Burst Size", palette=PALETTE_RD, markers=MARKERS_RD, s=15,
                      data=data, ax=ax, edgecolor="#2f2f2f", linewidth=0.5, zorder=3)
    
    #####################
    # Style fine-tuning #
    #####################
    
    # Log-scale y-axis;
    #plt.yscale("log")
    
    # Turn on the grid;
    ax.yaxis.grid(True, linewidth=0.3)
    ax.xaxis.grid(True, linewidth=0.3)
    
    # Set tick number and parameters on x and y axes;
    def year_formatter(x, pos=None):
        d = num2date(x)
        if (d.year - X_LIMITS[0].year) % 3 != 0:
            return ""
        else:
            return d.year
    ax.tick_params(axis="x", direction="out", which="both", bottom=True, top=False, labelsize=7, width=0.5, size=5)
    ax.tick_params(axis="x", direction="out", which="minor", size=2)  # Update size of minor ticks;
    ax.tick_params(axis="y", direction="out", which="both", left=True, right=False, labelsize=7, width=0.5, size=5)
    ax.tick_params(axis="y", direction="out", which="minor", size=2)  # Update size of minor ticks;

    plt.legend(title="Burst size",title_fontsize=legend_fontsize+1,loc="upper left", frameon=True, fontsize=legend_fontsize, markerscale=markerscale)

    # Add axes labels;
    plt.ylabel("Read Bandwidth Scaling [Gbit/s]", fontsize=ax_fontsize)
    plt.xlabel("Byte Size (2^x)", fontsize=ax_fontsize)
    return fig, ax

##############################
##############################
def mem_bw_write(data: pd.DataFrame,
                        set_axes_limits: bool=True,
                        plot_regression: bool=True) -> (plt.Figure, plt.Axes):
    """
    Parameters
    ----------
    data : pd.DataFrame with 6 columns:
        "year",
        "performance",
        "kind" ∈ ["compute", "memory", "interconnect"],
        "name" (label shown in the plot, it can be empty),
        "base" (base value used for speedup, it can be empty),
        "comment" (e.g. data source or non-used label, it can be empty).

    Returns
    -------
    fig : matplotlib figure containing the plot
    ax : matplotlib axis containing the plot
    """
    #done in the read
    #data.rename(columns={'Brst_size[#beats]': 'Burst Size'}, inplace=True)
    
    ##############
    # Plot setup #
    ##############
    
    # Reset matplotlib settings;
    plt.rcdefaults()
    # Setup general plotting settings;
    sns.set_style("white", {"ytick.left": True, "xtick.bottom": True})
    plt.rcParams["font.family"] = ["Latin Modern Roman Demi"]
    plt.rcParams['axes.labelpad'] = 0  # Padding between axis and axis label;
    plt.rcParams['xtick.major.pad'] = 1  # Padding between axis ticks and tick labels;
    plt.rcParams['ytick.major.pad'] = 1  # Padding between axis ticks and tick labels;
    plt.rcParams['axes.linewidth'] = 0.8  # Line width of the axis borders;
    
    # Create a figure for the plot, and adjust margins;
    fig = plt.figure(figsize=(6, 3))
    gs = gridspec.GridSpec(1, 1)
    plt.subplots_adjust(top=0.98,
                        bottom=0.1,
                        left=0.12,
                        right=0.99)  
    ax = fig.add_subplot(gs[0, 0])
    
    # Set axes limits;        
    if set_axes_limits:
        ax.set_xlim(X_LIMITS_SINGLE)
        ax.set_ylim(Y_LIMITS)
    
    #################
    # Main plot #####
    #################    
    
    
    # Add a scatterplot for individual elements of the dataset, and change color based on hardware type;
    ax = sns.scatterplot(x="PowOfTwoBytes", y="AVG_WR_BW[Gbit/s]", hue="Burst Size", style="Burst Size", palette=PALETTE_WR, markers=MARKERS_WR, s=15,
                      data=data, ax=ax, edgecolor="#2f2f2f", linewidth=0.5, zorder=4)
    
    #####################
    # Style fine-tuning #
    #####################
   
    # Turn on the grid;
    ax.yaxis.grid(True, linewidth=0.3)
    ax.xaxis.grid(True, linewidth=0.3)
    ax.tick_params(axis="x", direction="out", which="both", bottom=True, top=False, labelsize=7, width=0.5, size=5)
    ax.tick_params(axis="x", direction="out", which="minor", size=2)  # Update size of minor ticks;
    ax.tick_params(axis="y", direction="out", which="both", left=True, right=False, labelsize=7, width=0.5, size=5)
    ax.tick_params(axis="y", direction="out", which="minor", size=2)  # Update size of minor ticks;
    
    plt.legend(title="Burst size", title_fontsize=legend_fontsize+1, loc="upper left", frameon=True, fontsize=legend_fontsize, markerscale=markerscale)

    # Add axes labels;
    plt.ylabel("Write Bandwidth Scaling [Gbit/s]", fontsize=ax_fontsize)
    plt.xlabel("Byte Size (2^x)", fontsize=ax_fontsize)
    
    return fig, ax

##############################
##############################

if __name__ == "__main__":
    
    # Load data;
    data = pd.read_csv("../../data/memtest_smpl_vs_cmplx.csv")   
    # Convert date;
    #data["year"] = pd.to_datetime(data["year"], format='%Y-%m')

    # Create the plot;
    fig, ax = performance_scaling(data)   
    
    # Save the plot;
    save_plot("../../plots", "membw_performance_scaling_new.{}")  
    
    data = pd.read_csv("../../data/memtest_plot_complex.csv")   
    
    # Create the plot;
    fig, ax = mem_bw_read(data)   
    
    # Save the plot;
    save_plot("../../plots", "membw_read.{}")  

    # Create the plot;
    fig, ax = mem_bw_write(data)   
    
    # Save the plot;
    save_plot("../../plots", "membw_write.{}")  
