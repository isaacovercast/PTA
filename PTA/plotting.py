

import matplotlib
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    matplotlib.use("agg")
import math
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import shutil
import subprocess

from sklearn.decomposition import PCA
from sklearn.preprocessing import PowerTransformer, StandardScaler

from PTA.inference import default_targets
import PTA.util


## A dictionary mapping mModel parameters in params/SIMOUT file format
## to unicode/prettier versions for plotting
##
## Helpful matplotlib info for plotting unicode characters:
## https://matplotlib.org/users/mathtext.html#subscripts-and-superscripts
target_labels = {"zeta":"\u03B6",\
                "psi":"\u03A8",\
                "S_m":r"$S_M$",\
                "J_m":r"$J_M$",\
                "trait_rate_meta":r"$\sigma^2_M",\
                "ecological_strength":r"$s_E$",\
                "J":r"$J$",\
                "m":r"$m$"}


def _filter_sims(sims,\
                    feature_set='',\
                    nsims=1000,\
                    select='',\
                    tol='',\
                    verbose=False):
    """
    Load simulation data and perform filtering and downsampling that's common
    across many plotting routines. Normally you won't call this directly.

    :param str sims: 
    :param list feature_set:
    :param int nsims:
    :param int/float select: 
    :param bool verbose: Whether to print progress messages.

    :return: Returns a tuple of pd.DataFrame containing the community assembly
        model class labels for retained simulations and a pd.DataFrame of 
        filtered and pruned simulation summary statistics.
    """
    ## Load the simulations
    sim_df = PTA.util._load_sims(sims, sep=" ")

    ## Wether to select only specific timepoints bracketed by `tol` or
    ## just plot everything.
    if select is '':
        pass
    else:
        pass

    ## If feature_set is unspecified assume we're using all features.
    ## Leftover from mess, not sure if it's useful
    if not feature_set:
        pass

    ## Prune the simulations based on selected features and number of
    ## simulations to retain.
    sim_df = sim_df[:nsims]

    ## Remove invariant targets (save time)
    ## TODO: Removing invariant targets doesn't remove the keys from
    ## the janky 'default_targets' global thing we're doing rn. It's
    ## a convenience, so doesn't matter that much.
    #sim_df = sim_df.loc[:, (sim_df != sim_df.iloc[0]).any()]
    #retained = list(sim_df.columns)
    #if verbose: print("Removed invariant targets. Retained: {}".format(list(retained)))

    return sim_df


def plot_simulations_hist(sims,\
                        ax='',\
                        figsize=(12, 6),\
                        feature_set='',\
                        nsims=1000,\
                        bins=20,\
                        alpha=0.6,\
                        select='',\
                        tol='',\
                        title='',\
                        outfile='',\
                        verbose=False):
    """
    Simple histogram for each summary statistic. Useful for inspecting model
    performance. Invariant summary statistics will be removed.

    :param str sims: 
    :param tuple figsize:
    :param list feature_set:
    :param int nsims:
    :param int bins: The number of bins per histogram.
    :param float alpha: Set alpha value to determine transparency [0-1], larger
        values increase opacity.
    :param int/float select: 
    :param int/float tol:
    :param str title:
    :param str outfile:
    :param bool verbose:

    :return: Return a list of `matplotlib.pyplot.axis` on which the simulated
        summary statistics have been plotted. This list can be _long_ depending
        on how many statistics you plot.
    """

    ## Filter and downsample the simulations
    sim_df = _filter_sims(sims,\
                            feature_set=feature_set,\
                            nsims=nsims,\
                            select=select,\
                            tol=tol,\
                            verbose=verbose)

    target_df = sim_df[default_targets]
    axs = target_df.hist(figsize=figsize, alpha=alpha, bins=bins, grid=False)

    plt.tight_layout()
    return axs


def plot_simulations_pca(sims, ax='',\
                            figsize=(8, 8),\
                            target='',\
                            feature_set='',\
                            loadings=False,\
                            nsims=1000,\
                            select='',\
                            tol='',\
                            title='',\
                            outfile='',\
                            colorbar=True,\
                            verbose=False):
    """
    Plot summary statistics for simulations projected into PC space.

    :param str sims: 
    :param matplotlib.pyplot.axis ax:
    :param tuple figsize:
    :param str target:
    :param list feature_set:
    :param bool loadings: BROKEN! Whether to plot the loadings in the figure.
    :param int nsims:
    :param int/float select: 
    :param int/float tol:
    :param str title:
    :param str outfile:
    :param bool verbose:

    :return: Return the `matplotlib.pyplot.axis` on which the simulations are
        plotted.
    """
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)

    ## Filter and downsample the simulations
    sim_df = _filter_sims(sims,\
                            feature_set=feature_set,\
                            nsims=nsims,\
                            select=select,\
                            tol=tol,\
                            verbose=verbose)

    ## Have to retain the targets because we drop them prior to PCA    
    target_df = sim_df[default_targets]
    sim_df = sim_df.drop(default_targets, axis=1)

    ## These are also left over from mess and not sure they are needed.
    # sim_df = StandardScaler().fit_transform(sim_df)
    sim_df = PowerTransformer(method='yeo-johnson').fit_transform(sim_df)

    pca = PCA(n_components=2)
    dat = pca.fit_transform(sim_df)

    if not target:
        target = "zeta"
    target_values = target_df[target].values
    sc = ax.scatter(dat[:, 0], dat[:, 1], label=target_df[target], c=target_values)

    if colorbar
        plt.colorbar(sc)

    ## Remove a bunch of visual noise
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.tick_params(top='off', bottom='off', left='off', right='off')

    var_expl = pca.explained_variance_ratio_
    ax.set_xlabel("Variance explained {:.3}%".format(var_expl[0]*100), fontsize=15)
    ax.set_ylabel("Variance explained {:.3}%".format(var_expl[1]*100), fontsize=15)

    if title:
        ax.set_title(title)

    ## TODO: Doesn't work how I'd like.
    ##print("Explained variance", pca.explained_variance_ratio_)
    ##if loadings:
    ##    for i, comp in enumerate(pca.components_.T):
    ##        plt.arrow(0, 0, pca.components_.T[i,0], pca.components_.T[i,1], color = 'r',alpha = 0.5)
    ##        plt.text(pca.components_.T[i,0]* 1.5, pca.components_.T[i,1] * 1.5, dat[i+2], color = 'black', ha = 'center', va = 'center')

    ## If writing to file then don't plot to screen.
    if outfile:
        try:
            plt.savefig(outfile)
            if verbose: print("Wrote figure to: {}".format(outfile))
        except Exception as inst:
            raise Exception("Failed saving figure: {}".format(inst))
        plt.close()

    return ax


## This is cool, but it's not getting used for anything rn.
def _make_animated_gif(datadir, outfile, delay=50):
    """
    This function will take all png files in a directory and make them
    into an animated gif. The inputs are the directory with all the images
    and the full path including filename of the file to write out

    :param str datadir: Directory that contains all the component files
        for the animation. These should be .png, and should be alpha-sorted
        in the order of the animation.
    :param str outfile: The name of the file to write the animated gif to.
    :param int delay: Time delay between frame changes in 1/100 second
        increments.
    """

    ## Do the imagemagick conversion, if possible
    ## `convert -delay 100 outdir/* anim.gif`
    ## TODO: Define the total time to be 10 seconds, total available
    ## timeslots is * 100 bcz convert flag is in 1/100 seconds
    ## Default half second intervals
    cmd = "convert -delay {} ".format(delay)\
            + datadir + "/*.png "\
            + outfile
    try:
        res = subprocess.check_output(cmd, shell=True)
    except Exception as inst:
        print("Trouble creating abundances through time animated gif - {}".format(inst))
        print("You probably don't have imagemagick installed")


REQUIRE_IMAGEMAGICK_ERROR = """
The plots_through_time() function requires the image-magick graphics
processing package which may be installed with conda:

    conda install -c conda-forge imagemagick -y
"""


if __name__ == "__main__":
    pass
