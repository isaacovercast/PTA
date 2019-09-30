import datetime
import logging
import momi
import numpy as np
import os
import string
import time
import tempfile

from collections import OrderedDict

import PTA
from PTA.util import *
from PTA.msfs import *

LOGGER = logging.getLogger(__name__)

class DemographicModel(object):
    """
    The PTA object

    :param str name: The name of this PTA simulation. This is used for
        creating output files.
    :param bool quiet: Don't print anything ever.
    :param bool verbose: Print more progress info.
    """

    def __init__(self, name, quiet=False, verbose=False):
        if not name:
            raise PTAError(REQUIRE_NAME)

        ## Do some checking here to make sure the name doesn't have
        ## special characters, spaces, or path delimiters. Allow _ and -.
        ## This will raise an error immediately if there are bad chars in name.
        self._check_name(name)

        self._version = PTA.__version__

        ## stores default ipcluster launch info
        self._ipcluster = {
            "cluster_id" : "",
            "profile" : "default",
            "engines" : "Local",
            "quiet" : 0,
            "timeout" : 120,
            "cores" : 0, #detect_cpus(),
            "threads" : 2,
            "pids": {},
            }

        ## the default params dict
        ## If you add a parameter to this dictionary you need
        ## to also add a short description to the PARAMS dict
        ## at the end of this file
        ##
        ## Also be sure to add it to _paramschecker so the type gets set correctly
        self.paramsdict = OrderedDict([
                       ("simulation_name", name),
                       ("project_dir", "./default_PTA"),
                       ("npops", 10),
                       ("nsamps", 4),
                       ("N_e", 10000),
                       ("tau", 20000),
                       ("epsilon", 10),
                       ("length", 1000),
                       ("num_replicates", 100),
                       ("recoms_per_gen", 1e-9),
                       ("muts_per_gen", 1e-8)
        ])

        ## A dictionary for holding prior ranges for values we're interested in
        self._priors = dict([
                        ("N_e", []),
                        ("tau", []),
                        ("epsilon", []),
        ])

        ## elite hackers only internal dictionary, normally you shouldn't mess with this
        ##  * sorted_sfs: Whether or not to sort the bins of the msfs
        ##  * allow_psi>1: Whether to allow multiple co-expansion events per simulation
        ##      or to fix it to 1. This is the msbayes vs pipemaster flag.
        self._hackersonly = dict([
                       ("sorted_sfs", False),
                       ("allow_psi>1", True), 
        ])


    #########################
    ## Housekeeping functions
    #########################
    def __str__(self):
        return "<PTA.DemographicModel: {}>".format(self.paramsdict["simulation_name"])


    ## Test assembly name is valid and raise if it contains any special characters
    def _check_name(self, name):
        invalid_chars = string.punctuation.replace("_", "")\
                                          .replace("-", "")+ " "
        if any(char in invalid_chars for char in name):
            raise PTAError(BAD_PTA_NAME.format(name))


    def _get_simulation_outdir(self, prefix=""):
        """
        Construct an output directory for a simulation run.
        Make output directory formatted like <output_dir>/<name>-<timestamp><random 2 digit #>
        This will _mostly_ avoid output directory collisions, right?

        :param string prefix: The directory within which to create the
            simulation output directory.
        """

        dirname = prefix + self.paramsdict["simulation_name"]
        outdir = os.path.join(self.paramsdict["project_dir"],\
                              dirname\
                              + "-" + str(time.time()).replace(".", "")[-7:]\
                              + str(np.random.randint(100)))
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        return outdir


    def _paramschecker(self, param, newvalue, quiet=True):
        """
        Check and set parameters. Raises exceptions when params are set to 
        values they should not be.

        :param string param: The parameter to set.
        :param newvalue: The value of the parameter.
        :param bool quiet: Whether to print info.
        """
        ## TODO: This should actually check the values and make sure they make sense
        try:
            ## Cast params to correct types
            if param == "project_dir":
                ## If it already exists then just inform the user that we'll be adding
                ## more simulations to the current project directory
                self.paramsdict[param] = newvalue
                if not os.path.exists(self.paramsdict["project_dir"]):
                    os.mkdir(self.paramsdict["project_dir"])
                else:
                    if not quiet:
                        print("  Project directory exists. Additional simulations will be appended.")
            
            elif param in ["N_e", "tau", "epsilon"]:
                tup = tuplecheck(newvalue, dtype=int)
                if isinstance(tup, tuple):
                    self._priors[param] = tup
                    self.paramsdict[param] = sample_param_range(tup)[0]
                else:
                    self.paramsdict[param] = tup

            elif param in ["npops", "nsamsp", "length", "num_replicates"]:
                self.paramsdict[param] = int(newvalue)
            elif param in ["recoms_per_gen", "muts_per_gen"]:
                self.paramsdict[param] = float(newvalue)
            else:
                self.paramsdict[param] = newvalue
        except Exception as inst:
            ## Do something intelligent here?
            raise


    ## Getting parameters header and parameters carves off
    ## the simulation name and the project directory
    def _get_params_header(self):
        return list(self.paramsdict.keys())[2:]


    def _get_params_values(self):
        return list(self.paramsdict.values())[2:]


    def set_param(self, param, value, quiet=True):
        """
        A convenience function for setting parameters in the API mode, which
        turns out to be a little annoying if you don't provide this. With
        the set_param method you can set parameters on the Region, the
        Metacommunity, or the LocalCommunity. Simply pass the parameter
        name and the value, and this method identifies the appropriate target
        parameter.

        :param string param: The name of the parameter to set.
        :param value: The value of the parameter to set.
        :param bool quiet: Whether to print info to the console.
        """
        try:
            self = set_params(self, param, value, quiet)
        except:
            raise PTAError("Bad param/value {}/{}".format(param, value))


    def get_params(self):
        """
        A convenience function for getting nicely formatted params in API mode.

        :return: A string of all the params ready to be printed.
        """
        tf = tempfile.NamedTemporaryFile()
        self.write_params(outfile=tf.name, full=True, force=True)
        dat = open(tf.name).read()
        return dat


    def write_params(self, outfile=None, outdir=None, full=False, force=False):
        """
        Write out the parameters of this model to a file properly formatted as
        input for the PTA CLI. A good and simple way to share/archive 
        parameter settings for simulations. This is also the function that's
        used by __main__ to generate default params.txt files for `PTA -n`.

        :param string outfile: The name of the params file to generate. If not
            specified this will default to `params-<Region.name>.txt`.
        :param string outdir: The directory to write the params file to. If not
            specified this will default to the project_dir.
        :param bool full: Whether to write out only the parameters of the
            specific parameter values of this Region, or to write out the
            parameters including prior ranges for parameter values..
        :param bool force: Whether to overwrite if a file already exists.
        """
        if outfile is None:
            outfile = "params-"+self.paramsdict["simulation_name"]+".txt"

        ## If outdir is blank then default to writing to the project dir
        if outdir is None:
            outdir = self.paramsdict["project_dir"]
        elif not os.path.exists(outdir):
            raise PTAError(NO_OUTDIR).format(outdir)

        outfile = os.path.join(outdir, outfile)
        ## Test if params file already exists?
        ## If not forcing, test for file and bail out if it exists
        if not force:
            if os.path.isfile(outfile):
                raise PTAError(PARAMS_EXISTS.format(outfile))

        with open(outfile, 'w') as paramsfile:
            ## Write the header. Format to 80 columns
            header = "------- PTA params file (v.{})".format(PTA.__version__)
            header += ("-"*(80-len(header)))
            paramsfile.write(header)

            ## Whip through the current paramsdict and write out the current
            ## param value, the ordered dict index number. Also,
            ## get the short description from paramsinfo. Make it look pretty,
            ## pad nicely if at all possible.
            for key, val in self.paramsdict.items():
                paramvalue = str(val)

                ## If it's one of the params with a prior, and if the prior is not
                ## empty and if writing out full, then write the prior, and not
                ## the sampled value
                if full:
                    if key in list(self._priors.keys()):
                        paramvalue = "-".join([str(i) for i in self._priors[key]])

                padding = (" "*(20-len(paramvalue)))
                paramkey = list(self.paramsdict.keys()).index(key)
                paramindex = " ## [{}] ".format(paramkey)
                LOGGER.debug("{} {} {}".format(key, val, paramindex))
                name = "[{}]: ".format(key)
                description = PARAMS[key]
                paramsfile.write("\n" + paramvalue + padding + \
                                        paramindex + name + description)

            paramsfile.write("\n")


    ########################
    ## Model functions/API
    ########################
    def _sample_tau(self, ntaus=1):
        return np.random.randint(1000, 50000, ntaus)


    def _sample_epsilon(self, ntaus=1):
        return np.random.randint(1, 20, ntaus)


    def _sample_zeta(self):
        return np.random.uniform()


    def _sample_Ne(self):
        N_e = 0
        if isinstance(self.paramsdict["N_e"], tuple):
            min_Ne, max_Ne = self.paramsdict["N_e"]
            N_e = np.random.randint(min_Ne, max_Ne+1)
        else:
            N_e = self.paramsdict["N_e"]
        return N_e


    def get_pops_per_tau(self, zeta):

        # Get effective # of coexpanding taxa
        n_sync = int(np.round(zeta * self.paramsdict["npops"]))
    
        # There needs to be at least 1 coexpansion event and coexpansion events must
        # include at least 2 taxa
        if n_sync > 1:
            try:
                if self._hackersonly["allow_psi>1"]:
                    psi = np.random.randint(1, (n_sync+1)/2)
                else:
                    psi = 1
            except ValueError:
                # If n_sync + 1 / 2 = 1 then psi = 1
                psi = 1
            n_async = self.paramsdict["npops"] - n_sync
    
            # Have to allocate all pops to a table in the restaurant
            # Each table has to have at least 2 customers or its not a real table
            pops_per_tau = np.array([2] * psi, dtype=int)
    
            # Allocate any remaining pops to a table with uniform probability
            unallocated_pops = n_sync - np.sum(pops_per_tau)
            unallocated_pops = np.random.multinomial(unallocated_pops, [1/psi]*psi)
    
            pops_per_tau = pops_per_tau + unallocated_pops
            pops_per_tau = np.concatenate([pops_per_tau, [1]*n_async])
        else:
            # If no coexpansion events then everyone gets their own table
            psi = 0
            n_sync = 0
            n_async = self.paramsdict["npops"]
            pops_per_tau = np.array([1] * self.paramsdict["npops"])
    
        return psi, pops_per_tau.astype(int)

    
    def parallel_simulate(self, ipyclient, nsims=1, quiet=False, verbose=False):
        npops = self.paramsdict["npops"]
        parallel_jobs = {}
        _ipcluster = {}
        ## store ipyclient engine pids to the Assembly so we can
        ## hard-interrupt them later if assembly is interrupted.
        ## Only stores pids of engines that aren't busy at this moment,
        ## otherwise it would block here while waiting to find their pids.
        _ipcluster["pids"] = {}
        for eid in ipyclient.ids:
            engine = ipyclient[eid]
            if not engine.outstanding:
                pid = engine.apply(os.getpid).get()
                _ipcluster["pids"][eid] = pid

        lbview = ipyclient.load_balanced_view()
        for i in range(nsims):
            ## Call do_serial sims args are: nsims, quiet, verbose
            parallel_jobs[i] = lbview.apply(serial_simulate, self, 1, True, False)

        ## Wait for all jobs to finish
        start = time.time()
        printstr = " Performing Simulations    | {} |"
        while 1:
            try:
                fin = [i.ready() for i in parallel_jobs.values()]
                elapsed = datetime.timedelta(seconds=int(time.time()-start))
                if not quiet: progressbar(len(fin), sum(fin), printstr.format(elapsed))
                time.sleep(0.1)
                if len(fin) == sum(fin):
                    print("")
                    break
            except KeyboardInterrupt as inst:
                print("\n    Cancelling remaining simulations.")
                break
        if not quiet: progressbar(100, 100, " Finished {} simulations in   {}\n".format(i+1, elapsed))

        faildict = {}
        passdict = {}
        param_df = pd.DataFrame()
        msfs_list = []
        ## Gather results
        for result in parallel_jobs:
            try:
                if not parallel_jobs[result].successful():
                    faildict[result] = parallel_jobs[result].metadata.error
                else:
                    passdict[result] = parallel_jobs[result].result()
                    m_list = passdict[result]
                    #param_df = pd.concat([param_df, p_df], axis=1)
                    msfs_list.extend(m_list)
            except Exception as inst:
                LOGGER.error("Caught a failed simulation - {}".format(inst))
                ## Don't let one bad apple spoin the bunch,
                ## so keep trying through the rest of the asyncs
        LOGGER.debug(faildict)

        return msfs_list
    
    
    def serial_simulate(self, nsims=1, quiet=False, verbose=False):
        import pandas as pd
        npops = self.paramsdict["npops"]
    
        param_df = pd.DataFrame([], columns=["zeta", "psi", "pops_per_tau", "taus", "epsilons"])
        msfs_list = []

        printstr = " Performing Simulations    | {} |"
        for i in range(nsims):
            start = time.time()
            try:
                elapsed = datetime.timedelta(seconds=int(time.time()-start))
                if not quiet: progressbar(nsims, i, printstr.format(elapsed))

                zeta = self._sample_zeta()
                psi, pops_per_tau = self.get_pops_per_tau(zeta)
                LOGGER.debug("sim {} - zeta {} - psi {} - pops_per_tau{}".format(i, zeta, psi, pops_per_tau))
                taus = self._sample_tau(ntaus=len(pops_per_tau))
                epsilons = self._sample_epsilon(len(pops_per_tau))
                param_df.loc[i] = [zeta, psi, pops_per_tau, taus, epsilons]
                sfs_list = []
                for tidx, tau_pops in enumerate(pops_per_tau):
                    for pidx in range(tau_pops):
                        name = "pop{}-{}".format(tidx, pidx)
                        sfs_list.append(self.get_sfs(name,
                                                N_e=self._sample_Ne(),
                                                tau=taus[tidx],
                                                epsilon=epsilons[tidx]))
                msfs = multiSFS(sfs_list)
                msfs.set_params(pd.Series([zeta, psi, pops_per_tau, taus, epsilons],\
                                        index=["zeta", "psi", "pops_per_tau", "taus", "epsilons"]))
                msfs_list.append(msfs)

            except KeyboardInterrupt as inst:
                print("\n    Cancelling remaining simulations")
                break
            except Exception as inst:
                LOGGER.debug("Simulation failed: {}".format(inst))
                raise PTAError("Failed inside serial_simulate: {}".format(inst))

        if not quiet: progressbar(100, 100, " Finished {} simulations in   {}\n".format(i+1, elapsed))

        return msfs_list


    def get_sfs(self, name, N_e=1e6, tau=20000, epsilon=10, verbose=False):
        model = momi.DemographicModel(N_e=N_e)
        model.add_leaf(name)
        model.set_size(name, t=tau, N=N_e/epsilon)
        sampled_n_dict={name:4}
        if verbose: print(sampled_n_dict)
        ac = model.simulate_data(length=self.paramsdict["length"],
                                num_replicates=self.paramsdict["num_replicates"],
                                recoms_per_gen=self.paramsdict["recoms_per_gen"],
                                muts_per_gen=self.paramsdict["muts_per_gen"],
                                sampled_n_dict=sampled_n_dict)
        return ac.extract_sfs(n_blocks=1)

    
    def simulate(self, nsims=1, ipyclient=None, quiet=False, verbose=False, force=False):
        """
        Do the heavy lifting here. 

        :param int nsims: The number of PTA codemographic simulations to
            perform.
        :param ipyparallel.Client ipyclient: If specified use this ipyparallel
            client to parallelize simulation runs. If not specified simulations
            will be run serially.
        :para bool quiet: Whether to display progress of these simulations.
        :para bool verbose: Display a bit more progress information.
        :param bool force: Whether to append to or overwrite results from
            previous simulations. Setting `force` to ``True`` will overwrite
            any previously generated simulation in the `project_dir/SIMOUT.txt`
            file.
        """
        param_df = pd.DataFrame([], columns=["zeta", "psi", "pops_per_tau", "taus", "epsilons"])

        if not quiet: print("    Generating {} simulation(s).".format(nsims))

        if not os.path.exists(self.paramsdict["project_dir"]):
            os.mkdir(self.paramsdict["project_dir"])

        simfile = os.path.join(self.paramsdict["project_dir"], "SIMOUT.txt")    
        ## Open output file. If force then overwrite existing, otherwise just append.
        io_mode = 'a'
        if force:
            io_mode = 'w'
            ## Prevent from shooting yourself in the foot with -f
            try:
                os.rename(simfile, simfile+".bak")
            except FileNotFoundError:
                ## If the simfile doesn't exist catch the error and move on
                pass

        if ipyclient:
            msfs_list = self.parallel_simulate(ipyclient, nsims=nsims, quiet=quiet, verbose=verbose)
        else:
            # Run simulations serially
            msfs_list = self.serial_simulate(nsims=nsims, quiet=quiet, verbose=verbose)
    
        ## Decide whether to print the header, if stuff is already in there then
        ## don't print the header, unless you're doing force because this opens
        ## in overwrite mode.
        header = msfs_list[0]._header() + "\n"
        if os.path.exists(simfile) and not force:
            header = ""

        with open(simfile, io_mode) as outfile:
            outfile.write(header)

            for msfs in msfs_list:
                try:
                    outfile.write(msfs.to_string() + "\n")
                except Exception as inst:
                    print("Writing output failed. See pta_log.txt.")
                    LOGGER.error("Malformed msfs: {}\n{}\n{}".format(inst, msfs.to_string()))


def serial_simulate(model, nsims=1, quiet=False, verbose=False):
    import os
    LOGGER.debug("Entering sim - {} on pid {}\n{}".format(model, os.getpid(), model.paramsdict))
    res = model.serial_simulate(nsims, quiet=quiet, verbose=verbose)
    LOGGER.debug("Leaving sim - {} on pid {}".format(model, os.getpid()))
    return res


#############################
## Model Parameter Info Dicts
#############################
PARAMS = {
    "simulation_name" : "The name of this simulation scenario",\
    "project_dir" : "Where to save files",\
    "npops" : "Number of populations undergoing co-demographic processes",\
    "nsamps" : "Numbers of samples for each populations",\
    "N_e" : "Effective population size of the ancestral population",\
    "tau" : "Time of demographic change",\
    "epsilon" : "Magnitude of demographic change",\
    "length" : "Length in bp of each indpendent genomic region to simulate",
    "num_replicates" : "Number of genomic regions to simulate",
    "recoms_per_gen" : "Recombination rate within independent regions scaled per base per generation",
    "muts_per_gen" : "Mutation rate scaled per base per generation",\
}


#############################
## Global error messages
#############################
BAD_PTA_NAME = """\
    No spaces or special characters of any kind are allowed in the simulation
    name. Special characters include all punctuation except dash '-' and
    underscore '_'. A good practice is to replace spaces with underscores '_'.
    An example of a good simulation name is: hawaiian_arthropods

    Here's what you put:
    {}
    """

PARAMS_EXISTS = """
    Error: Params file already exists: {}
    Use force argument to overwrite.
    """

NO_OUTDIR = """
    Error: Attempting to write params to a directory that doesn't exist - {}
    """

REQUIRE_NAME = """\
    Simulation scenario name _must_ be set. This is the first parameter in the
    params.txt file, and will be used as a prefix for output files. It should be a
    short string with no special characters, i.e., not a path (no \"/\" characters).
    If you need a suggestion, name it after the location you're working on.
    """

if __name__ == "__main__":
    pass
