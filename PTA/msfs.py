import numpy as np
import os
import pandas as pd
from scipy.stats import entropy, kurtosis, iqr, skew
from collections import OrderedDict


class multiSFS(object):
    def __init__(self, sfs_list, sort=False, proportions=False):
        self.length = sfs_list[0].length
        self.ntaxa = len(sfs_list)
        self.df = multiSFS.to_df(sfs_list, sort=sort, proportions=proportions)
        self.config_array = self.df.index.values
        self.loc_counts = np.array([x.loc_counts for x in sfs_list])
        self.stats = pd.Series()

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
        return pd.concat([self.stats, msfs_dat], axis=1)


    def to_string(self, sep=" "):
        dat = self.to_dataframe().to_csv(header=False, index=False, sep=sep, line_terminator="")
        return dat


    def dump(self, outdir="", out="", full=False):
        """
        """
        if not outdir:
            outdir = "./"
        if not out:
            out = "output.msfs"

        if not os.path.exists(outdir):
            os.mkdir(outdir)

        with open(os.path.join(outdir, out), 'a+') as outfile:
            outfile.write(self.to_string())

        if full:
            outdir = os.path.join(outdir, "sfs_dir")
            if not os.path.exists(outdir):
                os.mkdir(outdir)
            for sfs in self.sfslist:
                sfs.dump(os.path.join(outdir, "{}.sfs".format(sfs.populations[0])))


    ## TODO: Do stuff here.
    @staticmethod
    def load(file):
        """
        Would be convenient to have a load method as well.
        """
        pass


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
            omega = params["taus"].mean()/params["taus"].var()
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

            for func_name, func in list(moments.items()):
                stat_dict["{}_{}".format(label, func_name)] = func(dat)
        self.stat_dict = stat_dict
        self.stats = pd.DataFrame(stat_dict, index=["0"])


    def _header(self, sep=" "):
        dat = self.to_dataframe().columns.tolist()
        return sep.join(dat)
