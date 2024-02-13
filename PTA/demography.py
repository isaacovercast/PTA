import datetime
import itertools
import json
import logging
import momi
import msprime
import numpy as np
import os
import string
import time
import tempfile

from collections import OrderedDict

import PTA
from PTA.util import *
from PTA.msfs import *
from PTA.jmsfs import *

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
        self.name = name

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
                       ("zeta", 0),
                       ("length", 1000),
                       ("num_replicates", 100),
                       ("generation_time", 1),
                       ("recoms_per_gen", 1e-9),
                       ("muts_per_gen", 1e-8),
        ])

        ## Separator to use for reading/writing files
        self._sep = " "

        ## A dictionary for storing taxon specific information. This dictionary
        ## is populated when empirical data is loaded. If no per taxon info
        ## is present then we sample from the priors as specified in the params
        ## file.
        self.taxa = {}

        ## elite hackers only internal dictionary, normally you shouldn't mess with this
        ##  * sorted_sfs: Whether or not to sort the bins of the msfs
        ##  * allow_psi>1: Whether to allow multiple co-expansion events per simulation
        ##      or to fix it to 1. This is the msbayes vs pipemaster flag.
        ##  * proportional_msfs: Scale counts within an sfs bin per population to sum to 1.
        ##  * mu_variance: If this parameter is > 0, then mu will be sampled from a zero-
        ##      truncated normal distribution with mean `muts_per_gen` and variance
        ##      `mu_variance`. If 0, then `muts_per_gen` is a fixed global mutation rate.
        ##  * Ne_loguniform: Whether to sample Ne values from uniform or loguniform distributions
        ##  * scale_tau_to_coaltime: Whether to scale tau values to coalescent time.
        ##      Default (False) returns taus in generations.
        ##  * tau_buffer: Time (in generations) of buffer around tau values. If 0 then
        ##      no buffer is used.
        self._hackersonly = dict([
                       ("sorted_sfs", True),
                       ("allow_psi>1", False), 
                       ("proportional_msfs", False),
                       ("mu_variance", 0),
                       ("Ne_loguniform", False),
                       ("scale_tau_to_coaltime", False),
                       ("tau_buffer", 0),
                       ("sfs_dim", 1),
                       ("fix_ts", 0),
        ])

        ## Ne_ave, the expected value of the Ne parameter given a unifrom prior
        ## This will actually be set inside _paramschecker
        self._Ne_ave = self.paramsdict["N_e"]

        ## The empirical msfs
        self.empirical_msfs = ""


    #########################
    ## Housekeeping functions
    #########################
    def __repr__(self):
        return self.get_params()


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
                if " " in newvalue:
                    raise PTAError("`project_dir` may not contain spaces. You put:\n{}".format(newvalue))
                self.paramsdict[param] = os.path.realpath(os.path.expanduser(newvalue))

                if not os.path.exists(self.paramsdict["project_dir"]):
                    os.mkdir(self.paramsdict["project_dir"])
            
            elif param in ["N_e", "tau", "epsilon"]:
                dtype = int
                if param == "epsilon":
                    dtype = float
                tup = tuplecheck(newvalue, dtype=dtype)
                if isinstance(tup, tuple):
                    self.paramsdict[param] = tup
                    if tup[0] <= 0:
                        raise PTAError("{} values must be strictly > 0. You put {}".format(param, tup))
                else:
                    self.paramsdict[param] = tup
                    if tup <= 0:
                        raise PTAError("{} values must be strictly > 0. You put {}".format(param, tup))
                if param == "N_e":
                    # Calculate Ne_ave, the expected value of the uniform prior on Nes
                    self._Ne_ave = np.mean(tup)

            elif param in ["num_replicates", "generation_time"]:
                ## num_replicates must be a list that is the same length
                ## as the number of populations, and should contain the
                ## numbers of observed loci for each sample in the data.
                ## If you pass in a single integer value then we just use
                ## the same num_replicates for all pops.
                ## islist will properly handle setting with a list in API
                ## mode and will do nothing in CLI mode.
                if param == "num_replicates":
                    dtype = int
                elif param == "generation_time":
                    dtype = float
                newvalue = tuplecheck(newvalue, islist=True, dtype=dtype)
                if isinstance(newvalue, dtype):
                    newvalue = [newvalue] * self.paramsdict["npops"]
                self.paramsdict[param] = newvalue
                if not len(newvalue) == self.paramsdict["npops"]:
                    raise PTAError(BAD_NUM_REPLICATES.format(len(newvalue),\
                                                             self.paramsdict["npops"]))

            elif param in ["nsamps"]:
                tup = tuplecheck(newvalue, islist=True, dtype=int)
                if isinstance(tup, tuple) or isinstance(tup, list):
                    if len(tup) > 2:
                        raise PTAError("{} limited to 2-dimension. You put {}".format(param, tup))
                    elif len(tup) == 2:
                        self._hackersonly["sfs_dim"] = 2
                    else:
                        self._hackersonly["sfs_dim"] = 1
                self.paramsdict[param] = tup

            elif param in ["npops", "length"]:
                self.paramsdict[param] = int(newvalue)

            elif param in ["recoms_per_gen", "muts_per_gen", "zeta"]:
                self.paramsdict[param] = float(newvalue)

            else:
                try:
                    self.paramsdict[param] = int(newvalue)
                except:
                    try:
                        self.paramsdict[param] = float(newvalue)
                    except:
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
        turns out to be a little annoying if you don't provide this.

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
        self.write_params(outfile=tf.name, force=True)
        dat = open(tf.name).read()
        return dat


    def write_params(self, outfile=None, outdir=None, force=False):
        """
        Write out the parameters of this model to a file properly formatted as
        input for the PTA CLI. A good and simple way to share/archive 
        parameter settings for simulations. This is also the function that's
        used by __main__ to generate default params.txt files for `PTA -n`.

        :param string outfile: The name of the params file to generate. If not
            specified this will default to `params-<Region.name>.txt`.
        :param string outdir: The directory to write the params file to. If not
            specified this will default to the project_dir.
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
            header = "------- PTA params - {} - (v.{}) ".format(\
                                    str(type(self)).split('.')[-1].split('\'')[0],\
                                    PTA.__version__)
            header += ("-"*(80-len(header)))
            paramsfile.write(header)

            ## Whip through the current paramsdict and write out the current
            ## param value, the ordered dict index number. Also,
            ## get the short description from paramsinfo. Make it look pretty,
            ## pad nicely if at all possible.
            for key, val in self.paramsdict.items():
                if isinstance(val, tuple):
                    paramvalue = "-".join(map(str, val))
                elif isinstance(val, list):
                    paramvalue = ",".join(map(str, val))
                else:
                    paramvalue = str(val)

                padding = (" "*(20-len(paramvalue)))
                paramkey = list(self.paramsdict.keys()).index(key)
                paramindex = " ## [{}] ".format(paramkey)
                LOGGER.debug("{} {} {}".format(key, val, paramindex))
                name = "[{}]: ".format(key)
                description = PARAMS[key]
                paramsfile.write("\n" + paramvalue + padding + \
                                        paramindex + name + description)

            paramsfile.write("\n")


    def save(self):
        _save_json(self)


    @staticmethod
    def load(json_path, quiet=False):
        """
        Load a json serialized object and ensure it matches to the current model format.
        """
        # expand HOME in JSON path name
        json_path = json_path.replace("~", os.path.expanduser("~"))

        # raise error if JSON not found
        if not os.path.exists(json_path):
            raise PTAError("""
                Could not find saved model file (.json) in expected location.
                Checks in: [project_dir]/[assembly_name].json
                Checked: {}
                """.format(json_path))

        # load JSON file
        with open(json_path, 'rb') as infile:
            fullj = json.loads(infile.read(), object_hook=_tup_and_byte)

        # get name and project_dir from loaded JSON
        oldname = fullj["model"].pop("name")
        olddir = fullj["model"]["paramsdict"]["project_dir"]
        oldpath = os.path.join(olddir, os.path.splitext(oldname)[0] + ".json")

        # create a fresh new Assembly
        null = PTA.DemographicModel(oldname, quiet=True)

        # print Loading message with shortened path
        if not quiet:
            oldpath = oldpath.replace(os.path.expanduser("~"), "~")
            print("  loading DemographicModel: {}".format(oldname))
            print("  from saved path: {}".format(oldpath))

        ## get the taxa. Create empty taxa dict of correct length
        taxa = fullj["model"].pop("taxa")
        null.taxa = taxa

        ## Set params
        oldparams = fullj["model"].pop("paramsdict")
        for param, value in oldparams.items():
            null.set_param(param, value)

        oldhackersonly = fullj["model"].pop("_hackersonly")
        null._hackersonly = oldhackersonly

        null._sep = fullj["model"].pop("_sep")
        null.empirical_msfs = fullj["model"].pop("empirical_msfs")

        taxon_names = list(fullj["taxa"].keys())

        return null


    def load_empirical(inpath):
        """
        Load in empirical data, either from a stored msfs or from a directory
        of individual momi-style sfs which have been dumped to file.
        """
        pass

    ########################
    ## Model functions/API
    ########################

    ## For sampling from priors, the draw comes from the half open interval
    ## so for tau we add 1 to account for most people not thinking of this.
    def _sample_tau(self, pops_per_tau):
        """
        pops_per_tau - A list of number of populations per coexpansion event.
                       In the simplest case of one coexpansion it will look
                       like this [6, 1, 1, 1], indictating one coexpansion of
                       6 taxa and independent expansions of 3 other taxa (9 total).
        returns a list of expansion times of length npops
        """
        tau = self.paramsdict["tau"]
        if isinstance(tau, tuple):
            tau = (tau[0], tau[1]+1)

            if not self._hackersonly["tau_buffer"]:
                # No buffering, sample random tau values
                taus = [[np.random.randint(tau[0], tau[1], 1)[0]] * x for x in pops_per_tau]
            else:
                # Sample times with buffer
                tau_min, tau_max = (tau[0], tau[1])
                buffmax = np.floor((tau_max - tau_min) / (len(pops_per_tau)+1) / 2)
                if not buffmax < self._hackersonly["tau_buffer"]:
                    # If tau_buffer less than buffmax use user specified buffer value
                    # otherwise enforce buffmax to allow all pops to get viable taus
                    buffmax = self._hackersonly["tau_buffer"]

                domain = range(int(tau_min), int(tau_max), 100)
                taus = []
                for i in range(len(pops_per_tau)):
                    tau = np.random.choice(domain)
                    domain = [x for x in domain if x <= tau - buffmax or x >= tau + buffmax]
                    taus.append(tau)

                taus = [[y] * x for y, x in zip(taus, pops_per_tau)]

            # Collapse the list of lists to a single list with ts per population
            taus = np.array(list(itertools.chain.from_iterable(taus)))

        else:
            # Tau invariable, so fix all times to the same value
            taus = np.array([tau] * self.paramsdict["npops"])

        if self._hackersonly["fix_ts"]:
            zeta_e = pops_per_tau[0]
            taus = np.array([self._hackersonly["fix_ts"]] * zeta_e + taus[zeta_e:].tolist())

        # Scale years to generations
        taus = taus/self.paramsdict["generation_time"]

        return taus

    ## Here we can either have all co-expanding taxa with the same epsilon
    ## or we can have one random epsilon per taxon. As it stands it's the first way
    def _sample_epsilon(self, pops_per_tau):
        eps = self.paramsdict["epsilon"]
        if isinstance(eps, tuple):
            eps = [[np.random.uniform(eps[0], eps[1], 1)[0]] * x for x in pops_per_tau]
            eps = np.array(list(itertools.chain.from_iterable(eps)))
        else:
            eps = np.array([eps] * self.paramsdict["npops"])
        return eps


    def _sample_zeta(self):
        """
        If zeta is specified in the params dict, then just use this value. If zeta
        is zero (the default), then sample a random value between [0, 1).
        """
        zeta = self.paramsdict["zeta"]
        ## If tau is fixed then zeta is 1 by definition
        if not isinstance(self.paramsdict["tau"], tuple):
            zeta = 1
        ## If zeta is 0 sample uniform [0, 1)
        elif not zeta:
            zeta = np.random.uniform()
        return zeta


    def _sample_Ne(self, nsamps=1):
        N_e = self.paramsdict["N_e"]
        if isinstance(N_e, tuple):
            if self._hackersonly["Ne_loguniform"]:
                N_e = np.exp(np.random.uniform(np.log(N_e[0]), np.log(N_e[1]+1), nsamps))
            else:
                N_e = np.random.randint(N_e[0], N_e[1]+1, nsamps)
        else:
            N_e = np.array([N_e] * nsamps)
        return N_e


    def _check_numreplicates(self):
        """
        Ensure num_replicates is the correct length. API mode doesn't
        automatically expand an individual integer to an int list, and
        also, if one chooses to specify the number of loci per pop, but
        this doesn't agree with the number of npops this is a problem.
        """
        num_replicates = self.paramsdict["num_replicates"]
        if isinstance(num_replicates, list):
            if not len(num_replicates) == self.paramsdict["npops"]:
                raise PTAError(BAD_NUM_REPLICATES.format(len(num_replicates),\
                                                         self.paramsdict["npops"]))
        elif isinstance(num_replicates, int):
            num_replicates = [num_replicates] * self.paramsdict["npops"]
        else:
            raise PTAError("num_replicates param must be int or list: {}".format(num_replicates))
        return num_replicates


    def _check_gentimes(self):
        """
        Ensure generation_time is the correct length.
        """
        gentimes = self.paramsdict["generation_time"]
        if isinstance(gentimes, list):
            if not len(gentimes) == self.paramsdict["npops"]:
                raise PTAError(BAD_GENTIMES.format(len(gentimes),\
                                                         self.paramsdict["npops"]))
        elif isinstance(gentimes, float) or isinstance(gentimes, int):
            gentimes = [gentimes] * self.paramsdict["npops"]
        else:
            raise PTAError("generation_time param must be float or list: {}".format(gentimes))
        return gentimes 


    def _sample_mu(self):
        """
        Sample mu from a zero-truncated normal distribution, if mu_var is
        specified, otherwise use fixed, global mu. If you set the mu_variance
        too high and you get a very small value, or zero, then the simulation
        will die and it'll raise a PTAError.
        """
        mu = self.paramsdict["muts_per_gen"]
        mu_var = self._hackersonly["mu_variance"]
        if mu_var:
            mu = np.random.normal(mu, mu_var)
            if mu < 0:
                mu = 0
        return mu


    def get_pops_per_tau(self, n_sync):
    
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
                    break
            except KeyboardInterrupt as inst:
                print("\n    Cancelling remaining simulations.")
                break
        if not quiet: progressbar(100, 100, " Finished {} simulations in   {}\n".format(i+1, elapsed))

        faildict = {}
        param_df = pd.DataFrame()
        msfs_list = []
        ## Gather results
        for result in parallel_jobs:
            try:
                if not parallel_jobs[result].successful():
                    faildict[result] = parallel_jobs[result].metadata.error
                else:
                    msfs_list.extend(parallel_jobs[result].result())
            except Exception as inst:
                LOGGER.error("Caught a failed simulation - {}".format(inst))
                ## Don't let one bad apple spoin the bunch,
                ## so keep trying through the rest of the asyncs
        LOGGER.debug(faildict)

        return msfs_list
    
    
    def serial_simulate(self, nsims=1, quiet=False, verbose=False):
        import pandas as pd
        npops = self.paramsdict["npops"]
    
        msfs_list = []

        printstr = " Performing Simulations    | {} |"
        start = time.time()
        for i in range(nsims):
            try:
                elapsed = datetime.timedelta(seconds=int(time.time()-start))
                if not quiet: progressbar(nsims, i, printstr.format(elapsed))

                zeta = self._sample_zeta()
                # Get effective # of coexpanding taxa
                zeta_e = int(np.ceil(zeta * self.paramsdict["npops"]))
                psi, pops_per_tau = self.get_pops_per_tau(n_sync=zeta_e)

                LOGGER.debug("sim {} - zeta {} - zeta_e {} - psi {} - pops_per_tau{}".format(i, zeta, zeta_e, psi, pops_per_tau))
                # All taus, epsilons, and N_es will be the length of npops
                # taus here will be in generations not years
                taus = self._sample_tau(pops_per_tau)
                epsilons = self._sample_epsilon(pops_per_tau)
                N_es = self._sample_Ne(self.paramsdict["npops"])
                num_replicates = self._check_numreplicates()
                sfs_list = []
                idx = 0
                for tidx, tau_pops in enumerate(pops_per_tau):
                    for pidx in range(tau_pops):
                        name = "pop{}-{}".format(tidx, pidx)
                        ## FIXME: Here the co-expanding pops all receive the same
                        ## epsilon. Probably not the best way to do it.
                        sfs_list.append(self._simulate(name,
                                                N_e=N_es[idx],
                                                tau=taus[idx],
                                                epsilon=epsilons[idx],
                                                num_replicates=num_replicates[idx]))
                        idx += 1
                msfs = multiSFS(sfs_list,\
                                sort=self._hackersonly["sorted_sfs"],\
                                proportions=self._hackersonly["proportional_msfs"])

                if self._hackersonly["scale_tau_to_coaltime"]:
                    ## Scale time to coalescent units
                    ## Here taus in generations already, so scale to coalescent units
                    ## assuming diploid so 2 * 2Ne
                    taus = taus/(4*self._Ne_ave)

                ## In the pipe_master model the first tau in the list is the co-expansion time
                ## If/when you get around to doing the msbayes model of multiple coexpansion
                ## pulses, then this will have to change 
                msfs.set_params(pd.Series([zeta, zeta_e, psi, taus[0], pops_per_tau, taus, epsilons, N_es],\
                                        index=["zeta", "zeta_e", "psi", "t_s", "pops_per_tau", "taus", "epsilons", "N_es"]))
                msfs_list.append(msfs)

            except KeyboardInterrupt as inst:
                print("\n    Cancelling remaining simulations")
                break
            except Exception as inst:
                LOGGER.debug("Simulation failed: {}".format(inst))
                raise PTAError("Failed inside serial_simulate: {}".format(inst))

        if not quiet: progressbar(100, 100, " Finished {} simulations in   {}\n".format(i+1, elapsed))

        return msfs_list


    def _simulate(self,
                    name,
                    N_e=1e6,
                    tau=20000,
                    epsilon=10,
                    num_replicates=100,
                    verbose=False):

        model = momi.DemographicModel(N_e=N_e)
        model.add_leaf(name)
        ## epsilon > 1 is bottleneck backwards in time
        ## epsilon < 1 is expansion
        ## epsilon == 1 is constant size
        model.set_size(name, t=tau, N=N_e*epsilon)

        sampled_n_dict={name:self.paramsdict["nsamps"]}
        if verbose: print(sampled_n_dict)
        ac = model.simulate_data(length=self.paramsdict["length"],
                                num_replicates=num_replicates,
                                recoms_per_gen=self.paramsdict["recoms_per_gen"],
                                muts_per_gen=self._sample_mu(),
                                sampled_n_dict=sampled_n_dict)
        try:
            sfs = ac.extract_sfs(n_blocks=1)
            ## TODO: Issue #12 <- Allow unfolded SFS.
            sfs = sfs.fold()
        except ValueError:
            ## If _sample_mu() returns zero, or a very small value with respect to
            ## sequence length, Ne, and tau, then you can get a case where there
            ## are no snps in the data, and constructing the sfs freaks.
            raise PTAError("Can't extract SFS from a simulation with no variation. Check that muts_per_gen looks reasonable.")

        return sfs

    
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
            any previously generated simulation in the `project_dir/{name}-SIMOUT.txt`
            file.
        """
        param_df = pd.DataFrame([], columns=["zeta", "psi", "pops_per_tau", "taus", "epsilons"])

        if not quiet: print("    Generating {} simulation(s).".format(nsims))

        if not os.path.exists(self.paramsdict["project_dir"]):
            os.mkdir(self.paramsdict["project_dir"])

        if ipyclient:
            msfs_list = self.parallel_simulate(ipyclient, nsims=nsims, quiet=quiet, verbose=verbose)
        else:
            msfs_list = self.serial_simulate(nsims=nsims, quiet=quiet, verbose=verbose)

        self._write_df(msfs_list, force=force)


    ## Save the results to the output DataFrame
    def _write_df(self, msfs_list, force=False):

        simfile = os.path.join(self.paramsdict["project_dir"], "{}-SIMOUT.csv".format(self.name))
        ## Open output file. If force then overwrite existing, otherwise just append.
        if force:
            ## Prevent from shooting yourself in the foot with -f
            try:
                os.rename(simfile, simfile+".bak")
            except FileNotFoundError:
                ## If the simfile doesn't exist catch the error and move on
                pass
        try:
            dat = pd.read_csv(simfile, sep=self._sep)
        except FileNotFoundError:
            dat = pd.DataFrame()

        ## sort=False suppresses a warning about non-concatenation index if
        ## SIMOUT is empty
        msfs_df = pd.DataFrame(pd.concat([x.to_dataframe() for x in msfs_list], sort=False)).fillna(0)
        # Map sfs bin column names to str to prevent confusion between int/str column names
        msfs_df.columns = msfs_df.columns.map(str)

        dat = pd.concat([dat, msfs_df], sort=False)
        dat.to_csv(simfile, header=True, index=False, sep=self._sep, float_format='%.3f')



    def load_simulations(self, nrows=None):
        """
        Load in the simulation data, if it exists.
        """
        simfile = os.path.join(self.paramsdict["project_dir"], "{}-SIMOUT.csv".format(self.name))
        if not os.path.exists(simfile):
            raise PTAError("No simulations exist for {}".format(self.name))

        dat = pd.read_csv(simfile, sep=" ")

        if nrows == None:
            nrows = len(dat)

        # Split the params and jsfs data
        # Get all the columns with 'pop' in the column title, only sfs bins
        pidxs = [x for x in dat.columns if 'pop' not in x]
        sidxs = [x for x in dat.columns if 'pop' in x]

        params = dat[pidxs]
        msfs = dat[sidxs]

        return params.iloc[:nrows, :], msfs.iloc[:nrows, :]


    def plot_sims_PCA(self, color_by="zeta"):
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import PowerTransformer

        fig, ax = plt.subplots(figsize=(7, 5))

        params, msfs = self.load_simulations()

        dat = PowerTransformer(method='yeo-johnson').fit_transform(msfs)

        pca = PCA(n_components=2)
        pcs = pca.fit_transform(dat)

        g = ax.scatter(pcs[:, 0], pcs[:, 1], c=params[color_by])
        cbar = fig.colorbar(g)
        return ax


    def _write_simout(self, msfs_list, force=False):
        """
        This is the somewhat old-fashioned way of doing it where you just open
        the file and start writing out the simulations. This will _bite_ you if
        the msfs are sparse because then the lengths of lines will be different.
        I'm keeping it for now because I wrote it and it works and I'm precious
        about it. You can probably just remove, as the _write_df call generates
        essentially the exact same output, but is more reliable.
        """
        simfile = os.path.join(self.paramsdict["project_dir"], "{}-SIMOUT.txt".format(self.name))
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

        ## Decide whether to print the header, if stuff is already in there then
        ## don't print the header, unless you're doing force because this opens
        ## in overwrite mode.
        header = msfs_list[0]._header(sep=self._sep) + "\n"
        if os.path.exists(simfile) and not force:
            header = ""

        with open(simfile, io_mode) as outfile:
            outfile.write(header)

            for msfs in msfs_list:
                try:
                    outfile.write(msfs.to_string(sep=self._sep) + "\n")
                except Exception as inst:
                    print("Writing output failed. See pta_log.txt.")
                    LOGGER.error("Malformed msfs: {}\n{}\n{}".format(inst, msfs.to_string(self._sep)))


def serial_simulate(model, nsims=1, quiet=False, verbose=False):
    import os
    LOGGER.debug("Entering sim - {} on pid {}\n{}".format(model, os.getpid(), model.paramsdict))
    res = model.serial_simulate(nsims, quiet=quiet, verbose=verbose)
    LOGGER.debug("Leaving sim - {} on pid {}".format(model, os.getpid()))
    return res


##########################################
## Saving functions to dump model to json
## This is all ripped directly from ipyrad
## with minor modifications.
##########################################

class _Encoder(json.JSONEncoder):
    """
    Save JSON string with tuples embedded as described in stackoverflow
    thread. Modified here to include dictionary values as tuples.
    link: http://stackoverflow.com/questions/15721363/

    This Encoder Class is used as the 'cls' argument to json.dumps()
    """
    def encode(self, obj):
        """ function to encode json string"""
        def hint_tuples(item):
            """ embeds __tuple__ hinter in json strings """
            if isinstance(item, tuple):
                return {'__tuple__': True, 'items': item}
            if isinstance(item, list):
                return [hint_tuples(e) for e in item]
            if isinstance(item, dict):
                return {
                    key: hint_tuples(val) for key, val in item.items()
                }
            else:
                return item
        return super(_Encoder, self).encode(hint_tuples(obj))


def _default(o):
    print(o)
    # https://stackoverflow.com/questions/11942364/
    # typeerror-integer-is-not-json-serializable-when-
    # serializing-json-in-python?utm_medium=organic&utm_
    # source=google_rich_qa&utm_campaign=google_rich_qa
    if isinstance(o, np.int64):
        return int(o)
    raise TypeError


def _tup_and_byte(obj):
    """ this is used in loading """

    # convert all strings to bytes
    if isinstance(obj, (bytes)):
        return obj.decode()  # encode('utf-8')
        #return obj.encode('utf-8')

    # if this is a list of values, return list of byteified values
    if isinstance(obj, list):
        return [_tup_and_byte(item) for item in obj]

    # if this is a dictionary, return dictionary of byteified keys and values
    # but only if we haven't already byteified it
    if isinstance(obj, dict):
        if "__tuple__" in obj:
            return tuple(_tup_and_byte(item) for item in obj["items"])
        else:
            return {
                _tup_and_byte(key): _tup_and_byte(val) for
                key, val in obj.items()
                }

    # if it's anything else, return it in its original form
    return obj


def _save_json(data, quiet=False):
    """
    Save assembly and samples as json
    ## data as dict
    #### samples save only keys
    """
    # store params without the reference to Assembly object in params
    paramsdict = {i: j for (i, j) in data.paramsdict.items() if i != "_data"}

    # store all other dicts
    datadict = OrderedDict([\
        ("name", data.name),\
        ("__version__", data._version),\
        ("_sep", data._sep),\
        ("paramsdict", paramsdict),\
        ("taxa", list(data.taxa.keys())),\
        ("_hackersonly", data._hackersonly),\
        ("empirical_msfs", data.empirical_msfs)\
    ])

    ## save taxat
    taxadict = OrderedDict([])
    for key, taxon in data.taxa.items():
        taxadict[key] = taxon._to_fulldict()

    ## json format it using cumstom Encoder class
    fulldumps = json.dumps({
        "model": datadict,
        "taxa": taxadict
    },
        cls=_Encoder,
        sort_keys=False, indent=4, separators=(",", ":"),
        default=_default,
    )

    ## save to file
    modelpath = os.path.join(data.paramsdict["project_dir"],\
                                data.name + ".json")
    if not os.path.exists(data.paramsdict["project_dir"]):
        os.mkdir(data.paramsdict["project_dir"])

    ## protect save from interruption
    done = 0
    if not quiet: print("  Saving DemographicModel to {}".format(modelpath))
    while not done:
        try:
            with open(modelpath, 'w') as jout:
                jout.write(fulldumps)
            done = 1
        except (KeyboardInterrupt, SystemExit):
            print('.')
            continue


#############################
## Model Parameter Info Dicts
#############################
PARAMS = {
    "simulation_name" : "The name of this simulation scenario",\
    "project_dir" : "Where to save files",\
    "npops" : "Number of populations undergoing co-demographic processes",\
    "nsamps" : "Numbers of samples for each populations",\
    "N_e" : "Effective population size of the contemporary population",\
    "tau" : "Time of demographic change",\
    "epsilon" : "Magnitude of demographic change",\
    "zeta" : "Proportion of coexpanding taxa. Default will sample U~(0, 1)",\
    "length" : "Length in bp of each independent genomic region to simulate",\
    "num_replicates" : "Number of genomic regions to simulate",\
    "generation_time" : "Generation time in years",\
    "recoms_per_gen" : "Recombination rate within independent regions scaled per base per generation",\
    "muts_per_gen" : "Mutation rate scaled per base per generation",\
    "t_recent_change": "Time of recent size change (years)",\
    "t_historic_samp": "Time the historical sample was taken (years)",\
    "t_ancestral_change": "Time of ancestral size change (years)",\
    "ne_ancestral": "Ancestral Ne",\
    "r_modern": "Growth rate between time 0 and t_recent_change",\
    "r_ancestral": "Growth rate between t_ancestral_change and t_recent_change",\
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

BAD_NUM_REPLICATES = """
    `num_replicates` parameter must be either a single integer value, which
    will be interpreted as all populations having this number of loci, or it
    must be a list of integer values that is of length `npops`.

    len(num_replicates) = {}
    npops =               {}
    """

BAD_GENTIMES = """
    `generation_time` parameter must be either a single float value, which
    will be interpreted as all populations having this generation time, or it
    must be a list of float values that is of length `npops`.

    len(generation_time) = {}
    npops =               {}
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
