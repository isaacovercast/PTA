import numpy as np
import pandas as pd
from scipy.stats import entropy, kurtosis, iqr, skew


class multiSFS(object):
    def __init__(self, sfs_list, proportions=False):
        self.length = sfs_list[0].length
        self.ntaxa = len(sfs_list)
        self.df = multiSFS.to_df(sfs_list, proportions=proportions)
        self.config_array = self.df.index.values
        self.loc_counts = np.array([x.loc_counts for x in sfs_list])
        self.params = []

        # For debugging, you can remove this after msfs works good.
        self.sfslist = sfs_list


    ## This piece of trash method cost me several hours to figure out.
    @staticmethod
    def to_df(sfslist, proportions=False):
        dtype = np.uint32
        sfs_dict = {}
        for i, sfs in enumerate(sfslist):
            tmp_lc = sfs.loc_counts[0]
            if proportions:
                tmp_lc = tmp_lc/sum(tmp_lc)
                dtype = np.float32
            tmp_ca = sfs.config_array
            sfs_dict["pop{}".format(i)] = {np.array2string(x).replace(" ", "_"):y for x, y in zip(tmp_ca, tmp_lc)}
        return pd.DataFrame(sfs_dict).fillna(0).astype(dtype)


    ## Series will be alpha-sorted, so the order of the bins will be:
    ## pop1-cfg1 pop1-cfg2 pop1-cfg3 pop2-cfg1 pop2-cfg2 pop2-cfg3 ...
    def to_series(self):
        msfs_dat = pd.Series({"{}-{}".format(x, y):self.df[x][y] for x in self.df.columns\
                                                        for y in self.df.index})
        params_dat = self.params[:2]
        return pd.concat([params_dat, msfs_dat])


    def to_string(self, sep=" "):
        dat = self.to_series().tolist()
        dat = map(str, dat)
        return sep.join(dat)


    def dump(self, file):
        with open(file, 'a+') as outfile:
            outfile.write(self.to_string())


    ## What's coming in is pd.Series([zeta, psi, pops_per_tau, taus, epsilons]
    def set_params(self, params):
        self._full_params = params

        ## Convenience apparatus to make calculating the moments easier
        moments = {}
        for name, func in zip(["mean", "std", "skewness", "kurtosis", "median", "iqr"],\
                            [np.mean, np.std, skew, kurtosis, np.median, iqr]):
            moments[name] = func

        stat_dict = {}
        ## For each list of values, rip through and calculate stats
        for label, dat in zip(["pops_per_tau", "taus", "epsilons"],
                                [self._full_params.pops_per_tau,\
                                    self._full_params.taus,\
                                    self._full_param.epsilons]):

            for func_name, func in list(moments.items()):
                stat_dict["{}_{}".format(label, func_name)] = func(dat)

        self.params = params


    def _header(self, sep=" "):
        dat = self.to_series().index.tolist()
        return sep.join(dat)
