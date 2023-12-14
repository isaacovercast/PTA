# PTA - Phylogeographic Temporal Analysis
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/isaacovercast/PTA/master)


Whole genome or SNP based comparative phylogeographic analysis for co-distributed
assemblages of communities. The result of aggregating a bunch of momis and dadis.

## Quick installation
**PTA requires python 3.9 at the moment.**

* Install [miniconda](https://conda.io/miniconda.html) and activate conda
* Install PTA inside a clean conda environment:
```
conda create -n pta python=3.9
conda activate pta
conda install -c conda-forge -c bioconda -c PTA pta
```

### Stuck solving environment
If the conda install takes a very long time you can try changing the solver to libmamba. If you are impatient you should just do this right away because it works.

```
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
```

Full installation and usage information is available on the [PTA rtfd site](https://pta.readthedocs.io)
