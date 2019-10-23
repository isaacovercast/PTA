.. _sec-installation:

============
Installation
============

``PTA`` requires Python >= 3.5. Installation is facilitated by the conda package
management system.

1. Download `miniconda <https://conda.io/miniconda.html>`_ and run the installer: ``bash Miniconda*``
2. Create a separate `conda environment <https://conda.io/docs/user-guide/tasks/manage-environments.html>`_ to install PTA into:

.. code:: bash

    conda create -n PTA
    conda activate PTA

3. Install:

.. code:: bash

    conda install -c conda-forge -c bioconda -c PTA PTA

4. Test:

.. code:: bash

   PTA -v

Installation issues can be reported on the `PTA github <https://github.com/isaacovercast/PTA>`_.
