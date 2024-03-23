import datetime
import msprime
import numpy as np
import os
import pandas as pd
import time

import PTA
from PTA.util import *
from PTA.jmsfs import *

class DemographicModel_2D_Temporal(PTA.DemographicModel):
    def __init__(self, name, quiet=False, verbose=False):
        super().__init__(name, quiet, verbose)

        self.paramsdict = OrderedDict([
               ("simulation_name", name),
               ("project_dir", "./default_PTA"),
               ("npops", 10),
               ("nsamps", [4, 4]),
               ("zeta", 0),
               ("length", 1000),
               ("num_replicates", 100),
               ("generation_time", 1),
               ("body_size", 1),
               ("recoms_per_gen", 1e-9),
               ("muts_per_gen", 1e-8),
               ("t_recent_change", 80),
               ("t_historic_samp", 110),
               ("t_ancestral_change", 15000),
               ("ne_ancestral", 100000),
               ("r_modern_mu", -0.1),
               ("r_modern_sigma", 0),
               ("r_modern_alpha", 0),
               ("r_ancestral_mu", 0),
               ("r_ancestral_sigma", 0),
    ])


    def _paramschecker(self, param, newvalue, quiet=True):
        """
        Check and set parameters. Raises exceptions when params are set to
        values they should not be.

        :param string param: The parameter to set.
        :param newvalue: The value of the parameter.
        :param bool quiet: Whether to print info.
        """
        super()._paramschecker(param, newvalue, quiet)

        ## TODO: This should actually check the values and make sure they make sense
        try:
            if param in ["ne_ancestral"]:
                dtype = int
                tup = tuplecheck(newvalue, dtype=dtype)
                if isinstance(tup, tuple):
                    self.paramsdict[param] = tup
                    if tup[0] <= 0:
                        raise PTAError("{} values must be strictly > 0. You put {}".format(param, tup))
                else:
                    self.paramsdict[param] = tup
            elif param in ["r_modern_mu", "r_modern_sigma",\
                            "r_ancestral_mu", "r_ancestral_sigma"]:
                dtype = float
                newvalue = tuplecheck(newvalue, dtype=dtype)
                if isinstance(newvalue, tuple):
                    self.paramsdict[param] = newvalue
                self.paramsdict[param] = newvalue
            elif param == "body_size":
                dtype = float
                newvalue = tuplecheck(newvalue, islist=True, dtype=dtype)
                if isinstance(newvalue, dtype):
                    newvalue = [newvalue] * self.paramsdict["npops"]
                self.paramsdict[param] = newvalue
                if not len(newvalue) == self.paramsdict["npops"]:
                    raise PTAError(BAD_BODY_SIZE.format(len(newvalue),\
                                                             self.paramsdict["npops"]))
        except:
            ## Do something intelligent here?
            raise


    def _sample_Ne(self, nsamps=1):
        N_e = self.paramsdict["ne_ancestral"]
        if isinstance(N_e, tuple):
            if self._hackersonly["Ne_loguniform"]:
                N_e = np.exp(np.random.uniform(np.log(N_e[0]), np.log(N_e[1]+1), nsamps))
            else:
                N_e = np.random.randint(N_e[0], N_e[1]+1, nsamps)
        else:
            N_e = np.array([N_e] * nsamps)
        return N_e


    def _sample_zeta(self):
        """
        If zeta is specified in the params dict, then just use this value. If zeta
        is zero (the default), then sample a random value between [0, 1).
        """
        zeta = self.paramsdict["zeta"]
        zeta = np.random.uniform()
        return zeta


    def _sample_r(self, modern=True):
        """
        If r_modern_sigma is 0, then all species get identical r_modern, otherwise
        sample from a normal distribution centered on r_modern_mu

        param modern (bool) : Whether to sample modern or ancestral r
        """
        if modern:
            r_mu = self.paramsdict["r_modern_mu"]
            r_sigma = self.paramsdict["r_modern_sigma"]
        else:
            r_mu = self.paramsdict["r_ancestral_mu"]
            r_sigma = self.paramsdict["r_ancestral_sigma"]

        if isinstance(r_mu, list):
            # Sample the r_modern_mu uniformly if the r_modern param is a tuple
            # otherwise r_modern_mu is fixed
            r_mu = np.random.uniform(r_mu[0], r_mu[1])

        if isinstance(r_sigma, list):
            r_sigma = np.random.uniform(r_sigma[0], r_sigma[1])

        r_alpha = self.paramsdict["r_modern_alpha"]
        if isinstance(r_alpha, list):
            r_alpha = np.random.uniform(r_alpha[0], r_alpha[1])
        else:
            r_alpha = np.array(r_alpha)

        if modern:
            # Modern growth rate can be modified by alpha and body size following:
            # r_modern ~ N(r_modern_mu + r_modern_alpha * bodysize, r_modern_sigma)
            loc = r_mu + np.array(r_alpha) * self.paramsdict["body_size"]
        else:
            loc = r_mu

        return r_mu, r_sigma, r_alpha, np.random.normal(loc,
                                                        r_sigma,
                                                        self.paramsdict["npops"])


    def serial_simulate(self, nsims=1, quiet=False, verbose=False):
        import pandas as pd
        npops = self.paramsdict["npops"]

        jmsfs_list = []

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

                LOGGER.debug("sim {} - zeta {} - zeta_e {} - pops_per_tau {}".format(i, zeta, zeta_e, pops_per_tau))
                # All taus, epsilons, and N_es will be the length of npops
                # taus here will be in generations not years
                #taus = self._sample_tau(pops_per_tau)
                #epsilons = self._sample_epsilon(pops_per_tau)
                N_es = self._sample_Ne(self.paramsdict["npops"])
                num_replicates = self._check_numreplicates()
                gentimes = self._check_gentimes()

                # Sample r_modern per species
                r_mu, r_sig, r_alph, r_moderns = self._sample_r(modern=True)
                # Sample r_ancestrals, we are not estimating mu/sigma for
                # ancestrals, so we don't need to keep them.
                _, _, _, r_ancestrals = self._sample_r(modern=False)

                sfs_list = []
                idx = 0
                for tidx, tau_pops in enumerate(pops_per_tau):
                    for pidx in range(tau_pops):
                        sfs_list.append(self._simulate(
                                gentime=gentimes[idx],
                                t_recent_change=self.paramsdict["t_recent_change"],
                                t_historic_samp=self.paramsdict["t_historic_samp"],
                                t_ancestral_change=self.paramsdict["t_ancestral_change"],
                                ne_ancestral=N_es[idx],
                                r_modern=r_moderns[idx],
                                r_ancestral=r_ancestrals[idx],
                                ))
                        idx += 1
                jmsfs = JointMultiSFS(sfs_list,\
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
                jmsfs.set_params(pd.Series([zeta, zeta_e, r_mu, r_sig, r_alph, r_moderns, N_es],\
                                        index=["zeta", "zeta_e", "r_modern_mu", "r_modern_sigma",\
                                                "r_modern_alpha", "r_moderns", "Ne_anc"]))
                jmsfs_list.append(jmsfs)

            except KeyboardInterrupt as inst:
                print("\n    Cancelling remaining simulations")
                break
            except Exception as inst:
                LOGGER.debug("Simulation failed: {}".format(inst))
                raise PTAError("Failed inside serial_simulate: {}".format(inst))

        if not quiet: progressbar(100, 100, " Finished {} simulations in   {}\n".format(i+1, elapsed))

        return jmsfs_list

    
    def _simulate(self,
                    gentime=1,
                    t_recent_change=80,
                    t_historic_samp=110,
                    t_ancestral_change=15000,
                    ne_ancestral=100000,
                    r_modern=-0.1,
                    r_ancestral=0,
                    debug=False,
                    verbose=False):

        n_albatross = self.paramsdict["nsamps"][0]
        n_contemp = self.paramsdict["nsamps"][1]
        mu = self.paramsdict["muts_per_gen"]

        ne_historic=ne_ancestral/np.exp(-(r_ancestral)*((t_ancestral_change-t_historic_samp)/gentime))
        ne_contemp=ne_historic/np.exp(-(r_modern)*(t_recent_change/gentime))

        dem=msprime.Demography()
        dem.add_population(name="C",initial_size=ne_contemp)
        dem.add_population_parameters_change(time=0, growth_rate=r_modern)
        dem.add_population_parameters_change(time=(t_recent_change/gentime), growth_rate=0)
        dem.add_population_parameters_change(time=(t_historic_samp/gentime), growth_rate=r_ancestral)
        dem.add_population_parameters_change(time=(t_ancestral_change/gentime), growth_rate=0)

        if debug:
            history = msprime.DemographyDebugger(demography=dem)
            print(history)

        ## TODO: This doesn't handle different # of loci per population
        n_loci = self.paramsdict["num_replicates"]
        if isinstance(n_loci, list):
            n_loci = n_loci[0]
        length = self.paramsdict["length"]

        n_sites = n_loci * length-1
        rateseq = [0,0.5] * n_loci
        unlinkedloci_rates = rateseq[:-1]
        loci_startpoints = [x*length for x in list(range(n_loci))]
        loci_endpoints = [(x+1)*length-1 for x in list(range(n_loci))]
        loci_boundaries = sorted(loci_startpoints + loci_endpoints)
        rate_map = msprime.RateMap(position=loci_boundaries, rate=unlinkedloci_rates)

        albatross_sampset = msprime.SampleSet(n_albatross, time=round(t_historic_samp/gentime))
        contemporary_sampset = msprime.SampleSet(n_contemp)
        ts = msprime.sim_ancestry(
            samples=[contemporary_sampset,albatross_sampset],
            demography=dem,
            recombination_rate=rate_map,
            sequence_length=n_sites
        )
        mts=msprime.sim_mutations(ts, rate=mu)

        contemp_list=list(range(n_contemp*2))
        albatross_list=[x+n_contemp*2 for x in list(range(n_albatross*2))]
        jsfs = mts.allele_frequency_spectrum(sample_sets=[albatross_list, contemp_list],
                                             mode="site",
                                             span_normalise=False)

        return jsfs


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
        # The dataframe is formated so that the first bin of the jMSFS is 0
        idx = dat.columns.get_loc('0')
        params = dat.iloc[:nrows, :idx]
        jmsfs = dat.iloc[:nrows, idx:]
        nrows = min(len(jmsfs), nrows)
        jmsfs = jmsfs.values.reshape(nrows,
                            self.paramsdict["npops"],
                            self.paramsdict["nsamps"][0]*2+1,
                            self.paramsdict["nsamps"][1]*2+1)
        return params, jmsfs


    def plot_sims_PCA(self, color_by="Ne_s_mean"):
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import PowerTransformer

        fig, ax = plt.subplots(figsize=(7, 5))

        params, jmsfs = self.load_simulations()

        dat = PowerTransformer(method='yeo-johnson').fit_transform(\
                                        jmsfs.reshape(*jmsfs.shape[:-3], -1))

        pca = PCA(n_components=2)
        pcs = pca.fit_transform(dat)

        g = ax.scatter(pcs[:, 0], pcs[:, 1], c=params[color_by])
        cbar = fig.colorbar(g)
        return ax


BAD_BODY_SIZE = """
    `body_size` parameter must be either a single integer value, which
    will be interpreted as all populations having this body size, or it
    must be a list of integer values that is of length `npops`.

    len(body_size) =      {}
    npops =               {}
    """
