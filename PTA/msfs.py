import glob
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

class multiSFS(object):
    def __init__(self, sfs_list, sort=False, proportions=False):
        self.length = sfs_list[0].length
        self.ntaxa = len(sfs_list)
        self.df = multiSFS.to_df(sfs_list, sort=sort, proportions=proportions)
        self.config_array = self.df.index.values
        self.loc_counts = np.array([x.loc_counts for x in sfs_list])
        self.stats = pd.Series(dtype=object)

        # For debugging, you can remove this after msfs works good.
        self.sfslist = sfs_list


    ## This piece of trash method cost me several hours to figure out.
    @staticmethod
    def to_df(sfslist, sort=False, proportions=False):
        """
        The primary method for generating a multiSFS. It gets passed a list
        of momi style sfs objects and generates the PTA.multiSFS format.

        :param sort bool: Whether to sort the bins of the msfs. Default is False.
        :param proportions bool: Whether to rescale the bins so the sum to 1.
        """
        dtype = np.uint32
        sfs_dict = {}
        for i, sfs in enumerate(sfslist):
            tmp_lc = sfs.loc_counts[0]
            if proportions:
                tmp_lc = tmp_lc/sum(tmp_lc)
                dtype = np.float32
            tmp_ca = sfs.config_array
            sfs_dict["pop{}".format(i)] = {np.array2string(x).replace(" ", "_"):y for x, y in zip(tmp_ca, tmp_lc)}
        msfs = pd.DataFrame(sfs_dict).fillna(0).astype(dtype)
        if sort:
            ## The `result_type` argument allows to return as a DF retaining
            ## original column and index values, otherwise msfs will be a
            ## returned as a Series. Annoying.
            msfs = msfs.apply(sorted, reverse=True, axis=1, result_type='broadcast')
        return msfs


    ## Series will be alpha-sorted, so the order of the bins will be:
    ## pop1-cfg1 pop1-cfg2 pop1-cfg3 pop2-cfg1 pop2-cfg2 pop2-cfg3 ...
    def to_dataframe(self):
        msfs_dat = pd.DataFrame({"{}-{}".format(x, y):self.df[x][y] for x in self.df.columns\
                                                        for y in self.df.index}, index=["0"])
        if not self.stats.empty:
            ## empirical data won't have stats, so don't write it
            msfs_dat = pd.concat([self.stats, msfs_dat], axis=1)

        return msfs_dat


    def to_string(self, sep=" "):
        dat = self.to_dataframe().to_csv(header=False, index=False, sep=sep, line_terminator="")
        return dat


    def dump(self, outdir="", outfile="", full=False):
        """
        """
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


    def _header(self, sep=" "):
        dat = self.to_dataframe().columns.tolist()
        return sep.join(dat)
