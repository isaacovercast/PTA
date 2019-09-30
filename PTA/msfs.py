import numpy as np
import pandas as pd

class multiSFS(object):
    def __init__(self, sfs_list):
        self.length = sfs_list[0].length
        self.ntaxa = len(sfs_list)
        self.df = multiSFS.to_df(sfs_list)
        self.config_array = self.df.index.values
        self.loc_counts = np.array([x.loc_counts for x in sfs_list])
        self.params = []

        # For debugging, you can remove this after msfs works good.
        self.sfslist = sfs_list


    ## This piece of trash method cost me several hours to figure out.
    @staticmethod
    def to_df(sfslist):
        sfs_dict = {}
        for i, sfs in enumerate(sfslist):
            tmp_lc = sfs.loc_counts[0]
            tmp_ca = sfs.config_array
            sfs_dict["pop{}".format(i)] = {np.array2string(x):y for x, y in zip(tmp_ca, tmp_lc)}
        return pd.DataFrame(sfs_dict).fillna(0)


    ## Series will be alpha-sorted, so the order of the bins will be:
    ## pop1-cfg1 pop1-cfg2 pop1-cfg3 pop2-cfg1 pop2-cfg2 pop2-cfg3 ...
    def to_series(self):
        return pd.Series({"{}-{}".format(x, y):self.df[x][y] for x in self.df.columns\
                                                        for y in self.df.index})


    def to_string(self, sep=" "):
        msfs_dat = self.to_series().tolist()
        params_dat = self.params.tolist()[:2]
        dat = map(str, params_dat + msfs_dat)
        return sep.join(dat)


    def dump(self, file):
        with open(file, 'a+') as outfile:
            outfile.write(self.to_string())


    def set_params(self, params):
        self.params = params


    def _header(self, sep=" "):
        dat = self.params.index.tolist()[:2] + self.to_series().index.tolist()
        ## Configs by default look like this `popx-[[2 2]]`. Get rid of the space.
        dat = map(lambda x: x.replace(" ", "_"), dat)
        return sep.join(dat)
