"""Analysis functions"""

import os, gc
import numpy as np
import matplotlib as mpl
from numpy.random import default_rng
from tqdm import tqdm

gc.enable()
mpl.rcParams['pdf.fonttype'] = 42
rng = default_rng()


class gibbs(object):
    """Gibbs sampler to estimate parameters of an exponential mixture for a set 
    of data. Results are stored in gibbs.results, which uses /home/ricky
    MDAnalysis.analysis.base.Results(). If 'results=None' the gibbs sampler has
    not been executed, which requires calling '.run()'
    
    """

    def __init__(self, times, residue, loc=0, ncomp=15, niter=50000):
        self.times, self.residue = times, residue
        self.niter, self.loc, self.ncomp = niter, loc, ncomp
        self.results, self.g = None, 100

        diff = (np.sort(times)[1:]-np.sort(times)[:-1])
        self.ts = diff[diff != 0][0]
        

    def __str__(self):
        return f'Gibbs sampler'


    def _prepare(self):
        self.t, self.s = get_s(self.times, self.ts)

        if not os.path.exists(f'{self.residue}'):
            os.mkdir(f'{self.residue}')

        # initialize arrays
        self.indicator = np.memmap(f'{self.residue}/.indicator_{self.niter}.npy',
                                   shape=((self.niter + 1) // g, x.shape[0]), mode='w+',
                                   dtype=np.uint8)
        self.mcweights = np.zeros(((self.niter + 1) // g, self.ncomp))
        self.mcrates = np.zeros(((self.niter + 1) // g, self.ncomp))

        # guess hyperparameters
        self.whypers = np.ones(self.ncomp) / [self.ncomp]
        self.rhypers = np.ones((self.ncomp, 2)) * [1, 3]

    def _run(self):
        # initialize weights and rates
        inrates = 0.5 * 10 ** np.arange(-self.ncomp + 2, 2, dtype=float)
        tmpw = 9 * 10 ** (-np.arange(1, self.ncomp + 1, dtype=float))
        weights, rates = tmpw / tmpw.sum(), inrates[::-1]

        # gibbs sampler
        for j in tqdm(range(1, self.niter+1),
                      desc=f'{self.residue}-K{self.ncomp}',
                      position=self.loc, leave=False):

            # compute probabilities
            tmp = weights*rates*np.exp(np.outer(-rates,x)).T
            z = (tmp.T/tmp.sum(axis=1)).T
        
            # sample indicator
            s = np.argmax(rng.multinomial(1, z), axis=1)
            
            # get indicator for each data point
            inds = [np.where(s == i)[0] for i in range(self.ncomp)]

            # compute total time and number of point for each component
            Ns = np.array([len(inds[i]) for i in range(self.ncomp)])
            Ts = np.array([self.times[inds[i]].sum() for i in range(self.ncomp)])

            # sample posteriors
            weights = rng.dirichlet(self.whypers+Ns)
            rates = rng.gamma(self.rhypers[:, 0]+Ns, 1/(self.rhypers[:, 1]+Ts))

            # save every g steps
            if j%g==0:
                ind = j//g-1
                self.mcweights[ind], self.mcrates[ind] = weights, rates
                self.indicator[ind] = s

        self._save()

    def _save(self):
        attrs = ["mcweights", "mcrates", "ncomp", "niter", "s", "t", "residue",
                 "times"]
        values = [self.mcweights, self.mcrates, self.ncomp, self.niter, self.s,
                  self.t, self.residue, self.times]
        
        r = save_results(attrs, values)
        self.results = r

