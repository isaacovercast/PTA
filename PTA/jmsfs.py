import glob
import matplotlib
import momi
import numpy as np
import os
import pandas as pd
import warnings
from collections import OrderedDict
from scipy.stats import entropy, kurtosis, iqr, skew
from skbio.stats import composition as skb

## RuntimeWarning skew/kurtosis raise this for constant data and
## return 'nan', this is a change from scipy >= 1.9:
## https://github.com/scipy/scipy/issues/16765
warnings.simplefilter('ignore', RuntimeWarning)

np.set_printoptions(suppress=True)

class JointMultiSFS(object):
    def __init__(self, sfs_list, sort=False, proportions=False, mask_corners=True, clr=False):
        '''
        sfs_list: A list of dadi sfs files containing 2D sfs
        '''
        self.ntaxa = len(sfs_list)

        # sfs_list may be either a list of file names or an np.array
        if isinstance(sfs_list[0], np.ndarray):
            # This will take both a list of np.arrays or a properly formatted 3d array
            self.jsfs_shape = sfs_list[0].shape
            self.jMSFS = np.array(sfs_list)

        elif isinstance(sfs_list[0], str):
            ## The shape of all incoming jSFS (must all be the same shape)
            self.jsfs_shape = self._get_jsfs_shape(sfs_list[0])

            ## Load the data for each 2D sfs into a temp list
            tmpjMSFS = []
            for f in sfs_list:
                tmp_jsfs_shape = self._get_jsfs_shape(f)
                if not tmp_jsfs_shape == self.jsfs_shape:
                    raise Exception("All 2D-SFS must have the same dimensions")

                dat = open(f).readlines()
                sfs_dat = np.array(dat[1].split(), dtype=float)
                sfs_dat = sfs_dat.reshape(self.jsfs_shape)
                tmpjMSFS.append(sfs_dat)

            self.jMSFS = np.array(tmpjMSFS)

        ## A mask to remove bins above the diagonal for folded jsfs
        ## This will allow to remove zero-bins and also to mask
        ## the jsfs for calculation of clr (if requested)
        ## For 2d sfs, the diag to mask is a bit tricky to figure
        ## but here it is
        self._mask_diag = -int(round((self.jsfs_shape[0]-self.jsfs_shape[1])/2))+1
        self.mask = np.triu(np.ones(self.jsfs_shape), self._mask_diag)[::-1, :].astype(bool)

        ## Mask monomorphic bins (ancestral/ derived in all)
        if mask_corners:
            for idx, _ in enumerate(self.jMSFS):
                self.jMSFS[idx][0, 0]  = 0
                self.jMSFS[idx][-1, -1]  = 0

        ## Rescale sfs bins to proportions
        if proportions:
            for idx, _ in enumerate(self.jMSFS):
                ## Test for all empty sfs bins and set to 0 to avoid divide by
                ## zero and a jMSFS full of nan
                if self.jMSFS[idx].sum() == 0:
                    self.jMSFS[idx] = np.zeros(self.jsfs_shape)
                else:
                    self.jMSFS[idx] = self.jMSFS[idx]/self.jMSFS[idx].sum()

                    ## Center log ratio transform to make better use of the compositional
                    ## nature of the proportions. We only do this if we are doing proportional
                    ## sfs bins, and only if there is some data (skip sfs with 0 snps).
                    if clr:

                        tmp_jsfs = np.ma.array(self.jMSFS[idx], mask=self.mask)
                        tmp_jsfs = skb.clr(tmp_jsfs+1e-5)
                        # Put the zeros back
                        self.jMSFS[idx] = tmp_jsfs.filled(0)

        ## Calc proportions within sfs before sorting across sfs to better
        ## control for differences in # of snps for a given species pair
        if sort:
            self.jMSFS = np.sort(self.jMSFS, axis=0)[::-1]


        self.shape = self.jMSFS.shape


    def _get_jsfs_shape(self, sfs_file):
        try:
            # Set the shape of the 2D multiSFS.
            samps = open(sfs_file).readlines()[0].split()[:2]
            jsfs_shape = (int(samps[0]), int(samps[1]))
        except:
            raise Exception(f"Malformed sfs file, should be dadi format: {sfs_file}")
        return jsfs_shape


    def __repr__(self):
        msg = f"""
    jMSFS: ntaxa={self.ntaxa} - shape={self.jsfs_shape}

{self.to_string()}
            """
        return msg


    def to_string(self):
        return np.round(self.jMSFS, decimals=4)


    def to_dataframe(self):
        jmsfs_dat = pd.DataFrame(self.jMSFS.flatten()).T

        try:
            jmsfs_dat = self.stats.merge(jmsfs_dat, how="cross")
        except:
            ## empirical data won't have stats, so don't write it
            pass

        return jmsfs_dat


    def dump(self, outdir="", outfile="", full=False):
        """
        """
        raise Exception("JointMultiSFS.dump not implemented")
        if not outfile:
            outfile = "output.msfs"
        if not outdir:
            outdir = "./"
        if full: outdir = os.path.join(outdir, "sfs_dir")

        if not os.path.exists(outdir):
            os.mkdir(outdir)

        if full:
            for sfs in self.sfslist:
                sfs.dump(os.path.join(outdir, "{}.sfs".format(sfs.populations[0])))
        else:
            self.df.to_csv(os.path.join(outdir, outfile))


    ## TODO: Do stuff here.
    @staticmethod
    def load(insfs):
        """
        Would be convenient to have a load method as well.
        """
        raise Exception("JointMultiSFS.load not implemented")
        if os.path.isdir(insfs):
            if indir:
                sfss = glob.glob(simsfs + "/*")
                sfss = [momi.Sfs.load(x) for x in sfss]
                msfs = PTA.msfs.multiSFS(sfss, sort=sort, proportions=proportions)
        else:
            try:
                msfs = pd.read_csv(insfs, index_col=0)
            except:
                raise PTAError("Malformed msfs file: {}".format(insfs))
        return msfs


    ## What's coming in is pd.Series([zeta, zeta_e, r_modern_mu, r_modern_sigma,
    ##                                r_modern_alpha, r_moderns, Ne_anc]
    ## zeta_e is 'effective zeta', the # of populations co-expanding
    def set_params(self, params):
        self._full_params = params

        ## Convenience apparatus to make calculating the moments easier
        moments = OrderedDict({})
        for name, func in zip(["mean", "std", "skewness", "kurtosis", "median", "iqr"],\
                            [np.mean, np.std, skew, kurtosis, np.median, iqr]):
            moments[name] = func

        # zeta and zeta_e are fixed for a simulation
        stat_dict = OrderedDict({"zeta":params["zeta"],
                                "zeta_e":params["zeta_e"],
                                "r_modern_mu":params["r_modern_mu"],
                                "r_modern_sigma":params["r_modern_sigma"],
                                "r_modern_alpha":params["r_modern_alpha"],
                                })

        ## For each list of values, rip through and calculate stats
        for label, dat in zip(["r_moderns", "Ne_s"],
                                [self._full_params.r_moderns,\
                                 self._full_params.Ne_anc]):

            ## I'm going to just mask out the pops_per_tau stats for now
            ## as for the initial stage of development we'll focus on the
            ## pipe_master model, of one pulse of co-expansion.
            if label == "pops_per_tau":
                continue

            ## RuntimeWarning skew/kurtosis raise this for constant data and
            ## return 'nan', this is a change from scipy >= 1.9:
            ## https://github.com/scipy/scipy/issues/16765
            for func_name, func in list(moments.items()):
                stat_dict["{}_{}".format(label, func_name)] = func(dat)
        self.stat_dict = stat_dict
        self.stats = pd.DataFrame(stat_dict, index=["0"])


    def plot_2d_sfs(self, sfs_idx=None, vmin=None, vmax=None, ax=None, 
                           pop_ids=None, extend='neither', colorbar=True,
                           plot_residuals=False, cmap=matplotlib.pyplot.cm.viridis_r):
        """
        Heatmap of single 2d SFS. Extensively borrowed from DADI package plotting
        functions.

        sfs_idx: The index of the individual 2D-sfs to plot. If no value is given
                 plot the mean across all jSFS per bin
        """

        # If no sfs_idx is passed in then take the average value across all sfs bins
        if sfs_idx is None:
            sfs = self.jMSFS.mean(axis=0)
        else:
            sfs = self.jMSFS[sfs_idx]

        if plot_residuals:
            # This may only work for square matrices atm
            u = np.tril(sfs) - np.triu(sfs).T
            l = np.triu(sfs) - np.tril(sfs).T
            sfs = u+l
            cmap = matplotlib.pyplot.cm.Reds

        if ax is None:
            ax = matplotlib.pyplot.gca()
    
        if vmin is None:
            vmin = sfs[sfs > 0].min()
        if vmax is None:
            vmax = sfs.max()
    
        if vmax / vmin > 10:
            # Under matplotlib 1.0.1, default LogFormatter omits some tick lines.
            # This works more consistently.
            norm = matplotlib.colors.LogNorm(vmin=vmin*(1-1e-3), vmax=vmax*(1+1e-3))
            format = matplotlib.ticker.LogFormatterMathtext()
        else:
            norm = matplotlib.colors.Normalize(vmin=vmin*(1-1e-3), 
                                               vmax=vmax*(1+1e-3))
            format = None
        mappable=ax.pcolor(np.ma.masked_where(sfs<vmin, sfs), 
                           cmap=cmap, edgecolors='none',
                           norm=norm)
        cb = ax.figure.colorbar(mappable, extend=extend, format=format)
    
        ax.plot([0,sfs.shape[1]],[0, sfs.shape[0]], '-k', lw=0.2)
    
        if pop_ids is None:
            pop_ids = ['pop0','pop1']
        ax.set_ylabel(pop_ids[0])
        ax.set_xlabel(pop_ids[1])
    
        for tick in ax.xaxis.get_ticklines() + ax.yaxis.get_ticklines():
            tick.set_visible(False)
    
        ax.set_xlim(0, sfs.shape[1])
        ax.set_ylim(0, sfs.shape[0])
    
        
        matplotlib.pyplot.xticks(np.linspace(0.5, sfs.shape[1]-0.5, sfs.shape[1]),
                                labels=range(0,sfs.shape[1]))
        matplotlib.pyplot.yticks(np.linspace(0.5, sfs.shape[0]-0.5, sfs.shape[0]),
                                labels=range(0,sfs.shape[0]))
        
        return cb


    def _header(self, sep=" "): 
        raise Exception("JointMultiSFS._header not implemented")
        dat = self.to_dataframe().columns.tolist()
        return sep.join(dat)
