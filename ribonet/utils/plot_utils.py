
import seaborn as sns
import matplotlib.pyplot as plt
import h5py as h5
import pandas as pd
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from collections import Counter
from typing import Dict

color_map = LinearSegmentedColormap.from_list("", ["#762a83","#f7f7f7","#1b7837"])
plt.rcParams['pdf.fonttype'] = 42


def plot_pos_seqlets(fhfile, length):
    '''
    Plot heatmap showing positions of seqlets identified by TF-MoDISco.

    Parameters
    ----------
    fhfile : str
        Path to HDF5 file containing TF-MoDISco results.
    length : int
        Total sequence length for the heatmap.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing heatmaps for positive and negative patterns.
    '''
    fh = h5.File(fhfile)
    type_patterns = [p for p in ["pos_patterns","neg_patterns"] if fh[p].keys()]
    if not type_patterns:
        fh.close()
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.text(0.5, 0.5, "No patterns found", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return fig
    fig, ax = plt.subplots(nrows=len(type_patterns), figsize=(10, 5*len(type_patterns)))
    for ax_n, type_pattern in enumerate(type_patterns):
        subax = ax[ax_n] if len(type_patterns)>1 else ax
        if type_pattern in fh.keys():
            heatparts = []
            seps = []
            patterns = fh[type_pattern].keys()
            for pattern in [p for _,p in sorted(zip([int(s.split("_")[-1]) for s in patterns],patterns))]:
                is_revcomps = np.array(fh[type_pattern][pattern]["seqlets"]["is_revcomp"])
                starts = np.array(fh[type_pattern][pattern]["seqlets"]["start"])
                ends = np.array(fh[type_pattern][pattern]["seqlets"]["end"])
                cwm = list(fh[type_pattern][pattern]["seqlets"]["contrib_scores"])
                tempdf = pd.DataFrame(np.nan, index = np.arange(len(starts)), columns=np.arange(length))
                for n, data in enumerate(zip(starts, ends, cwm, is_revcomps)):
                    if data[3]==False:
                        tempdf.loc[n,data[0]:data[1]-1] = np.sum(np.abs(data[2]),axis=1)
                    else:
                        tempdf.loc[n,data[0]:data[1]-1] = np.sum(np.abs(data[2]),axis=1)[::-1]
                heatparts.append(tempdf.loc[np.argsort(starts)])
                seps.append(sum(len(p) for p in heatparts))
            heatdf = pd.concat(heatparts, axis=0, ignore_index=True)
            cmap = "rocket" if type_pattern=="pos_patterns" else "mako"
            sns.heatmap(heatdf, yticklabels=False, cmap = cmap, ax= subax, robust=True)
            for s in seps:
                subax.axhline(s, color="gray", linewidth=0.5)
        else: 
            fig.delaxes(subax)
    fh.close()
    fig.tight_layout()
    return fig


def generate_offset_figure(counts_by_len: Dict[int, Counter], readlengths: list, window: int):
    """
    Build matplotlib Figure showing counts per length (2x3 grid).

    Parameters
    ----------
    counts_by_len : dict
        length -> Counter(relative_pos -> counts)
    readlengths : list
        List of read lengths to plot
    window : int
        +/- window to plot

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the subplots.
    """
    x = list(range(-window, window + 1))
    n_columns = 3
    n_rows = int(np.ceil(len(readlengths) / n_columns))

    xticks = list(range(-window, window + 1, 5))

    fig, axes = plt.subplots(n_rows, n_columns, figsize=(12, 3*n_rows))

    for ax, L in zip(axes.flat, readlengths):
        y = [counts_by_len.get(L, Counter()).get(xx, 0) for xx in x]
        ax.bar(x, y, color="#4b7b6a")
        ax.axvline(0, linestyle='--', linewidth=1)
        ax.set_title(f"{L} nt (n={sum(y)})")

        ax.set_xlabel("Position relative to CDS start (nt)")
        ax.set_ylabel("5' read starts")
        ax.set_xlim(-window - 0.5, window + 0.5)
        ax.set_xticks(xticks)
        ax.tick_params(axis='x', labelrotation=90)
        ax.set_xlim(-window - 0.5, window + 0.5)

    for j in range(len(readlengths), n_rows * n_columns):
        axes.flat[j].axis('off')
    
    plt.tight_layout()
    return fig


def plot_length_distribution(read_lengths: pd.DataFrame):
    """
    Generate read length distribution plots.

    Parameters
    ----------
    read_lengths : pd.DataFrame
        DataFrame containing columns "bam" and "Length".

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the read length distribution plots.
    """
    bams_unique = read_lengths["bam"].unique()

    read_min = read_lengths["Length"].min()
    read_max = min(40, read_lengths["Length"].max())

    ncols = 3
    nrows = (len(bams_unique) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3 * nrows))
    axes = axes.flatten()

    for ax, bam_name in zip(axes, bams_unique):

        subset = read_lengths[read_lengths["bam"] == bam_name]

        counts = subset["Length"].value_counts().sort_index()

        ax.bar(counts.index, counts.values, color="#4b7b6a")

        ax.set_title(bam_name, fontsize=9)
        ax.set_xlim(read_min, read_max)
        ax.set_xlabel("Read Length")
        ax.set_ylabel("Count")

    # Remove unused panels
    for ax in axes[len(bams_unique):]:
        ax.remove()

    fig.suptitle("Read Length Distribution")
    plt.tight_layout()

    return fig