import numpy as np

class multiSFS(object):
    def __init__(self, sfs_list):
        self.length = sfs_list[0].length
        self.ntaxa = len(sfs_list)
        self.config_array = sfs_list[0].config_array
        self.loc_counts = np.array([x.loc_counts for x in sfs_list])

    def to_string(self, sep=" "):
        dat = map(str, self.flatten()[0].tolist())
        return sep.join(dat)

    def flatten(self):
        return np.concatenate(self.loc_counts, axis=1)

    def dump(self, file):
        with open(file, 'a+') as outfile:
            outfile.write(self.to_string())
