# -*- coding: utf-8 -*-
"""
Created on Mon Feb 7 18:50:00 2022

@author: davideconficconi
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
PALETTE_RD_5 = ["#40004b","#762a83","#9970ab","#c2a5cf","#e7d4e8"]
#PALETTE_RD = sns.color_palette("colorblind",13)
MARKERS_RD = [".","8", "p","*","h","H","P","X","D","d","<","v","s" ]
MARKERS_RD_5 = [".","8", "p","*","X" ]
# ["#67001f","#b2182b","#d6604d","#f4a582","#fddbc7","#e0e0e0","#bababa","#878787","#4d4d4d","#1a1a1a"]
PALETTE_WR = ["#40004b","#762a83","#9970ab","#c2a5cf","#e7d4e8","#f7f7f7","#d9f0d3","#a6dba0","#5aae61","#1b7837","#00441b","#40004b","#762a83"]
#MARKERS_WR = ["o","|","1","2","v","^","+","x","3","4"]
#MARKERS_WR = [".","8", "p","*","h","H","P","X","D","d" ]
MARKERS_WR = [".","8", "p","*","h","H","P","X","D","d","<","v","s" ]
#MARKERS_WR = ["_","|","1","x","4",0,3,"3","+","2"]

# # Axes limits used in the plot, change them accordingy to your data;
X_LIMITS = (2,34)
Y_LIMITS = (0, 25000)
X_LIMITS_SINGLE = (0,11)
legend_fontsize=11
ax_fontsize=12
markerscale=1
major_lbl_size=10
marker_size=50

##############################

##############################
##############################

def cf_plot(data: pd.DataFrame,
            set_axes_limits: bool=True,
            plot_regression: bool=True) -> (plt.Figure, plt.Axes):
    """
    Parameters
    ----------
    data : pd.DataFrame with 3 columns:
        "thr[#]",
        "thread[ms]",
        "openmp[ms]"

    Returns
    -------
    fig : matplotlib figure containing the plot
    ax : matplotlib axis containing the plot
    """
    # data.rename(columns={'Brst_size[#beats]': 'Burst Size'}, inplace=True)
    
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
    fig = plt.figure(figsize=(10, 5))
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
    ax = sns.scatterplot(x="thr[#]", y="openmp[ms]", hue="img_dim", style="img_dim", palette=PALETTE_RD_5, markers=MARKERS_RD_5, s=marker_size,
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
    ax.tick_params(axis="x", direction="out", which="both", bottom=True, top=False, labelsize=major_lbl_size, width=0.5, size=5)
    ax.tick_params(axis="x", direction="out", which="minor", size=2)  # Update size of minor ticks;
    ax.tick_params(axis="y", direction="out", which="both", left=True, right=False, labelsize=major_lbl_size, width=0.5, size=5)
    ax.tick_params(axis="y", direction="out", which="minor", size=2)  # Update size of minor ticks;

    plt.legend(title="Scaling IMG vs #FPGAs",title_fontsize=legend_fontsize+1,loc="upper right", frameon=True, fontsize=legend_fontsize, markerscale=markerscale)

    # Add axes labels;
    plt.ylabel("Execution Time [ms]", fontsize=ax_fontsize)
    plt.xlabel("Thread Number", fontsize=ax_fontsize)
    return fig, ax

##############################
##############################

if __name__ == "__main__":
    
    # Load data;
    dataudp8 = pd.read_csv("../../data/cf_wax_res/cf_udp_8_5p.csv")
    dataudp128 = pd.read_csv("../../data/cf_wax_res/cf_udp_128.csv")
    dataudp256 = pd.read_csv("../../data/cf_wax_res/cf_udp_256.csv")
    dataudp512 = pd.read_csv("../../data/cf_wax_res/cf_udp_512_5p.csv")
    dataudp1k = pd.read_csv("../../data/cf_wax_res/cf_udp_1024.csv")
    # Convert date;
    #data["year"] = pd.to_datetime(data["year"], format='%Y-%m')

    # Create the plot;
        # fig, ax = performance_scaling(data)   
        
        # # Save the plot;
        # save_plot("../../plots", "membw_performance_scaling_new.{}")  
        
        # data = pd.read_csv("../../data/memtest_plot_complex.csv")   
    
    # Create the plot;
    # frames = [dataudp8, dataudp128,dataudp256,dataudp512,dataudp1k]

    data = pd.concat( [dataudp8, dataudp128,dataudp256,dataudp512,dataudp1k])
    fig, ax = cf_plot(data)   
    
    # Save the plot;
    save_plot("../../plots", "cf_wax_scaling.{}")  

    # Create the plot;
    # fig, ax = mem_bw_write(data)   
    
    # # Save the plot;
    # save_plot("../../plots", "membw_write.{}")  
