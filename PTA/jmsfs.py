import glob
import matplotlib
import momi
import numpy as np
import os
import pandas as pd
import warnings
from scipy.stats import entropy, kurtosis, iqr, skew
from collections import OrderedDict

## RuntimeWarning skew/kurtosis raise this for constant data and
## return 'nan', this is a change from scipy >= 1.9:
## https://github.com/scipy/scipy/issues/16765
warnings.simplefilter('ignore', RuntimeWarning)

np.set_printoptions(suppress=True)

class JointMultiSFS(object):
    def __init__(self, sfs_list, sort=False, proportions=False, mask_corners=True):
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

        ## Mask monomorphic bins (ancestral/ derived in all)
        if mask_corners:
            for idx, _ in enumerate(self.jMSFS):
                self.jMSFS[idx][0, 0]  = 0
                self.jMSFS[idx][-1, -1]  = 0

        ## Rescale sfs bins to proportions
        if proportions:
            for idx, _ in enumerate(self.jMSFS):
                self.jMSFS[idx] = self.jMSFS[idx]/self.jMSFS[idx].sum()

        ## Calc proportions within sfs before sorting across sfs to better
        ## control for differences in # of snps for a given species pair
        if sort:
            self.jMSFS = np.sort(self.jMSFS, axis=0)[::-1]


    def _get_jsfs_shape(self, sfs_file):
        try:
            # Set the shape of the 2D multiSFS.
            samps = open(sfs_file).readlines()[0].split()[:2]
            jsfs_shape = (int(samps[0]), int(samps[1]))
        except:
            raise Exception("Malformed sfs file, should be dadi format")
        return jsfs_shape


    def __repr__(self):
        msg = f"""
    jMSFS: ntaxa={self.ntaxa} - shape={self.jsfs_shape}

{self.to_string()}
            """
        return msg


    def to_string(self):
        return np.round(self.jMSFS, decimals=4)


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


    ## What's coming in is pd.Series([zeta, zeta_e, psi, pops_per_tau, taus, epsilons, N_es]
    ## zeta_e is 'effective zeta', the # of populations co-expanding
    def set_params(self, params):
        raise Exception("JointMultiSFS.set_params not implemented")
        self._full_params = params

        ## Convenience apparatus to make calculating the moments easier
        moments = OrderedDict({})
        for name, func in zip(["mean", "std", "skewness", "kurtosis", "median", "iqr"],\
                            [np.mean, np.std, skew, kurtosis, np.median, iqr]):
            moments[name] = func

        stat_dict = OrderedDict({"zeta":params["zeta"],\
                                "zeta_e":params["zeta_e"],\
                                "psi":params["psi"],\
                                "t_s":params["t_s"]})

        ## Handle taus with no variance and set omega to 0
        if not params["taus"].var():
            omega = 0
        else:
            omega = params["taus"].var()/params["taus"].mean()
        stat_dict["omega"] = omega

        ## For each list of values, rip through and calculate stats
        for label, dat in zip(["pops_per_tau", "taus", "epsilons", "Ne_s"],
                                [self._full_params.pops_per_tau,\
                                    self._full_params.taus,\
                                    self._full_params.epsilons,\
                                    self._full_params.N_es]):

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


    def plot_2d_sfs(self, sfs_idx=0, vmin=None, vmax=None, ax=None, 
                           pop_ids=None, extend='neither', colorbar=True,
                           cmap=matplotlib.pyplot.cm.viridis_r):
        """
        Heatmap of single 2d SFS. Extensively borrowed from DADI package plotting
        functions.

        sfs_idx: The index of the individual 2D-sfs to plot
        """

        sfs = self.jMSFS[sfs_idx]

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
