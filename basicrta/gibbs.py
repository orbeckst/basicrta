"""Analysis functions
"""

import os
import gc
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy.random import default_rng
from tqdm import tqdm
from scipy import stats
from sklearn.cluster import KMeans
from MDAnalysis.analysis.base import Results
from basicrta.util import confidence_interval

gc.enable()
mpl.rcParams['pdf.fonttype'] = 42
rng = default_rng()


class ProcessProtein(object):
    def __init__(self, niter):
        self.residues, self.niter = {}, niter


    def __getitem__(self, item):
        return getattr(self, item)


    def collect_results(self):
        from glob import glob
        if not os.getcwd().split('/')[-1][:8] == 'basicrta':
            raise NotImplementedError('navigate to basicrta-{cutoff} directory'
                                      'and rerun')
        dirs = np.array(glob('?[0-9]*'))
        sorted_inds = np.array([int(adir[1:]) for adir in dirs]).argsort()
        dirs = dirs[sorted_inds]
        for adir in tqdm(dirs):
            if os.path.exists(f'{adir}/gibbs_{self.niter}.pkl'):
                with open(f'{adir}/gibbs_{self.niter}.pkl', 'rb') as r:
                    self.residues[adir] = pickle.load(r)
            elif os.path.exists(f'{adir}/results_{self.niter}.pkl'):
                try:
                    self.residues[adir] = Gibbs().load_results(f'{adir}/results'
                                                           f'_{self.niter}.pkl')
                except ValueError:
                    print(f'{adir} does not contain a valid dataset')
                    continue
            else:
                print(f'results for {adir} do not exist')
                # raise FileNotFoundError(f'results for {adir} do not exist')


class ParallelGibbs(object):
    def __init__(self):
        print('unimplemented')
        # Add class to take in processed contacts and resids and
        # run gibbs samplers


class Gibbs(object):
    """Gibbs sampler to estimate parameters of an exponential mixture for a set 
    of data. Results are stored in gibbs.results, which uses /home/ricky
    MDAnalysis.analysis.base.Results(). If 'results=None' the gibbs sampler has
    not been executed, which requires calling '.run()'
    """

    def __init__(self, times=None, residue=None, loc=0, ncomp=15, niter=50000):
        self.times, self.residue = times, residue
        self.niter, self.loc, self.ncomp = niter, loc, ncomp
        self.g, self.burnin = 100, 10000
        self.processed_results = Results()

        if times:
            diff = (np.sort(times)[1:]-np.sort(times)[:-1])
            self.ts = diff[diff != 0][0]
        else:
            self.ts = None

        self.keys = {'times', 'residue', 'loc', 'ncomp', 'niter', 'g', 'burnin',
                     'processed_results', 'ts'}


    def __getitem__(self, item):
        return getattr(self, item)


    def _prepare(self):
        from basicrta.util import get_s
        self.t, self.s = get_s(self.times, self.ts)

        if not os.path.exists(f'{self.residue}'):
            os.mkdir(f'{self.residue}')

        # initialize arrays
        self.indicator = np.memmap(f'{self.residue}/.indicator_{self.niter}.'
                                   f'npy',
                                   shape=((self.niter + 1) // self.g,
                                          self.times.shape[0]),
                                   mode='w+', dtype=np.uint8)
        self.mcweights = np.zeros(((self.niter + 1) // self.g, self.ncomp))
        self.mcrates = np.zeros(((self.niter + 1) // self.g, self.ncomp))

        # guess hyperparameters
        self.whypers = np.ones(self.ncomp) / [self.ncomp]
        self.rhypers = np.ones((self.ncomp, 2)) * [1, 3]


    def run(self):
        # initialize weights and rates
        self._prepare()
        inrates = 0.5 * 10 ** np.arange(-self.ncomp + 2, 2, dtype=float)
        tmpw = 9 * 10 ** (-np.arange(1, self.ncomp + 1, dtype=float))
        weights, rates = tmpw / tmpw.sum(), inrates[::-1]

        # gibbs sampler
        for j in tqdm(range(1, self.niter+1),
                      desc=f'{self.residue}-K{self.ncomp}',
                      position=self.loc, leave=False):

            # compute probabilities
            tmp = weights*rates*np.exp(np.outer(-rates, self.times)).T
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
            if j%self.g == 0:
                ind = j//self.g-1
                self.mcweights[ind], self.mcrates[ind] = weights, rates
                self.indicator[ind] = s

        # attributes to save
        attrs = ["mcweights", "mcrates", "ncomp", "niter", "s", "t", "residue",
                 "times"]
        values = [self.mcweights, self.mcrates, self.ncomp, self.niter, self.s,
                  self.t, self.residue, self.times]
        
        self._save_results(attrs, values)
        self._process_gibbs()


    def _process_gibbs(self, cutoff=1e-4):
        burnin_ind = self.burnin // self.g

        inds = np.where(self.mcweights[burnin_ind:] > cutoff)
        indices = (np.arange(self.burnin, self.niter + 1, self.g)[inds[0]] //
                   self.g)
        lens = [len(row[row > cutoff]) for row in self.mcweights[burnin_ind:]]
        ncomp = stats.mode(lens, keepdims=False)[0]

        weights = self.mcweights[burnin_ind::][inds]
        rates = self.mcrates[burnin_ind::][inds]

        data = np.stack((weights, rates), axis=1)
        km = KMeans(n_clusters=ncomp, n_init=17).fit(np.log(data))
        Indicator = np.zeros((self.times.shape[0], ncomp))
        indicator = np.memmap(f'{self.residue}/.indicator_{self.niter}.npy',
                              shape=((self.niter + 1) // self.g,
                                     self.times.shape[0]),
                              mode='r', dtype=np.uint8)

        for j in np.unique(inds[0]):
            mapinds = km.labels_[inds[0] == j]
            for i, indx in enumerate(inds[1][inds[0] == j]):
                tmpind = np.where(indicator[j] == indx)[0]
                Indicator[tmpind, mapinds[i]] += 1

        Indicator = (Indicator.T / Indicator.sum(axis=1)).T

        attrs = ["weights", "rates", "ncomp", "residue", "indicator", "labels",
                 "iteration", "niter"]
        values = [weights, rates, ncomp, self.residue, Indicator,
                  km.labels_, indices, self.niter]
        self._save_results(attrs, values, processed=True)
        self._estimate_params()
        self._pickle_self()


    def _pickle_self(self):
        with open(f'{self.residue}/gibbs_{self.niter}.pkl', 'w+b') as f:
            pickle.dump(self, f)


    def _save_results(self, attrs, values, processed=False):
        if processed:
            r = self.processed_results
        else:
            r = self

        for attr, value in zip(attrs, values):
            setattr(r, attr, value)

        if processed:
            with open(f'{r.residue}/processed_results_{r.niter}.pkl',
                      'wb') as W:
                pickle.dump(r, W)
        else:
            with open(f'{r.residue}/results_{r.niter}.pkl', 'wb') as W:
                pickle.dump(r, W)


    def load_results(self, results, processed=False):
        if processed:
            with open(results, 'r+b') as f:
                r = pickle.load(f)

            for attr in list(r.keys()):
                setattr(self.processed_results, attr, r[f'{attr}'])
        else:
            with open(results, 'r+b') as f:
                r = pickle.load(f)

            for attr in list(r.keys()):
                setattr(self, attr, r[f'{attr}'])

            self._process_gibbs()
        return self


    def hist_results(self, scale=1.5, save=False):
        cmap = mpl.colormaps['tab20']
        rp = self.processed_results

        fig, ax = plt.subplots(1, 2, figsize=(4*scale, 3*scale))
        [ax[0].hist(rp.weights[rp.labels == i],
                     bins=np.exp(np.linspace(np.log(rp.weights[rp.labels == i]
                                                    .min()),
                                             np.log(rp.weights[rp.labels == i]
                                                    .max()), 50)),
                    label=f'{i+1}', alpha=0.5, color=cmap(i))
         for i in range(rp.ncomp)]
        [ax[1].hist(rp.rates[rp.labels == i],
                    bins=np.exp(np.linspace(np.log(rp.rates[rp.labels == i]
                                                   .min()),
                                            np.log(rp.rates[rp.labels == i]
                                                   .max()), 50)),
                    label=f'{i+1}', alpha=0.5, color=cmap(i))
         for i in range(rp.ncomp)]
        ax[0].set_xscale('log')
        ax[1].set_xscale('log')
        ax[0].legend(title='component')
        ax[1].legend(title='component')
        ax[0].set_xlabel(r'weight')
        ax[1].set_xlabel(r'rate ($ns^{-1}$)')
        ax[0].set_ylabel('count')
        ax[0].set_xlim(1e-4, 1)
        ax[1].set_xlim(1e-3, 10)
        plt.tight_layout()
        if save:
            plt.savefig('hist_results.png', bbox_inches='tight')
            plt.savefig('hist_results.pdf', bbox_inches='tight')
        plt.show()


    def plot_results(self, scale=1.5, sparse=1, save=False):
            cmap = mpl.colormaps['tab20']
            rp = self.processed_results

            fig, ax = plt.subplots(2, figsize=(4*scale, 3*scale), sharex=True)
            [ax[0].plot(rp.iteration[rp.labels == i][::sparse],
                        rp.weights[rp.labels == i][::sparse], '.',
                        label=f'{i+1}', color=cmap(i))
             for i in range(rp.ncomp)]
            ax[0].set_yscale('log')
            ax[0].set_ylabel(r'weight')
            [ax[1].plot(rp.iteration[rp.labels == i][::sparse],
                        rp.rates[rp.labels == i][::sparse], '.', label=f'{i+1}',
                        color=cmap(i)) for i in range(rp.ncomp)]
            ax[1].set_yscale('log')
            ax[1].set_ylabel(r'rate ($ns^{-1}$)')
            ax[1].set_xlabel('sample')
            ax[0].legend(title='component')
            ax[1].legend(title='component')
            plt.tight_layout()
            if save:
                plt.savefig('plot_results.png', bbox_inches='tight')
                plt.savefig('plot_results.pdf', bbox_inches='tight')
            plt.show()


    def _estimate_params(self):
        rp = self.processed_results

        ds = [rp.rates[rp.labels == i] for i in range(rp.ncomp)]
        bounds = np.array([confidence_interval(d) for d in ds])
        H = [np.histogram(rp.rates[rp.labels == i], bins=100) for i in
             range(rp.ncomp)]
        params = np.zeros(len(H))
        for i, hist in enumerate(H):
            ind = np.where(hist[0] == hist[0].max())[0]
            val = 0.5 * (hist[1][:-1][ind] + hist[1][1:][ind])
            try:
                params[i] = val
            except ValueError:
                params[i] = np.min(val)
                print(f'May have underestimated {self.residue} param for '
                      f'component with label {i}')

        setattr(rp, 'parameters', params)
        setattr(rp, 'intervals', bounds)


    def estimate_tau(self):
        from basicrta.util import get_bars
        rp = self.processed_results
        index = np.argmin(rp.parameters)
        taus = 1/rp.rates[rp.labels == index]
        ci = confidence_interval(taus)
        H = np.histogram(taus, bins=15)
        indmax = np.where(H[0] == H[0].max())[0]
        val = 0.5 * (H[1][:-1][indmax] + H[1][1:][indmax])[0]
        bars = get_bars([ci[0], val, ci[1]])
        return [val, *bars]


if __name__ == '__main__':
    print('do nothing')