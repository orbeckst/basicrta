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
from MDAnalysis.analysis.base import Results
from basicrta.util import confidence_interval
from multiprocessing import Pool, Lock
import MDAnalysis as mda
from basicrta import istarmap

gc.enable()
mpl.rcParams['pdf.fonttype'] = 42
rng = default_rng()


class ProcessProtein(object):
    def __init__(self, niter, prot):
        self.residues = {}
        self.niter = niter
        self.prot = prot

    def __getitem__(self, item):
        return getattr(self, item)

    def _collect_res(self, adir):
        print(adir)
        if os.path.exists(f'{adir}/gibbs_{self.niter}.pkl'):
            with open(f'{adir}/gibbs_{self.niter}.pkl', 'rb') as r:
                self.residues[adir] = pickle.load(r)
        elif os.path.exists(f'{adir}/results_{self.niter}.pkl'):
            try:
                self.residues[adir] = Gibbs().load_results(f'{adir}/results'
                                                       f'_{self.niter}.pkl')
            except ValueError:
                print(f'{adir} does not contain a valid dataset')
        else:
            print(f'results for {adir} do not exist')
                # raise FileNotFoundError(f'results for {adir} do not exist')

    def collect_results(self, nproc=1):
        from glob import glob

        if not os.getcwd().split('/')[-1][:8] == 'basicrta':
            raise NotImplementedError('navigate to basicrta-{cutoff} directory'
                                      'and rerun')
        dirs = np.array(glob('?[0-9]*'))
        sorted_inds = np.array([int(adir[1:]) for adir in dirs]).argsort()
        dirs = dirs[sorted_inds]
        with Pool(nproc, initializer=tqdm.set_lock, initargs=(Lock(),)) as p:
            try:
                for _ in tqdm(p.imap(self._collect_res, dirs),
                              total=len(dirs), position=0,
                              desc='overall progress'):
                    pass
            except KeyboardInterrupt:
                    pass
        for adir in dirs:
            self._collect_res(adir)

    def _get_taus(self):
        from basicrta.util import get_bars

        taus = []
        for res in self.residues:
            gib = self.residues[res]
            taus.append(gib.estimate_tau())
        taus = np.array(taus)
        print(taus.shape)
        bars = get_bars(taus)
        return taus[:, 1], bars

    def plot_protein(self):
        from basicrta.util import plot_protein
        taus, bars = self._get_taus()
        residues = list(self.residues.keys())
        plot_protein(residues, taus, bars, self.prot)


class ParallelGibbs(object):
    """
    A module to take a contact map and run Gibbs samplers for each residue
    """

    def __init__(self, contacts, nproc=1, ncomp=15, niter=110000):
        self.cutoff = float(contacts.strip('.pkl').split('/')[-1].split('_')
                            [-1])
        self.niter = niter
        self.nproc = nproc
        self.ncomp = ncomp
        self.contacts = contacts

    def run(self, run_resids=None):
        from basicrta.util import run_residue

        with open(self.contacts, 'r+b') as f:
            contacts = pickle.load(f)

        protids = np.unique(contacts[:, 0])
        if not run_resids:
            run_resids = protids

        if not isinstance(run_resids, (list, np.ndarray)):
            run_resids = [run_resids]

        rg = contacts.dtype.metadata['ag1'].residues
        resids = rg.resids
        reslets = np.array([mda.lib.util.convert_aa_code(name) for name in
                            rg.resnames])
        residues = np.array([f'{reslet}{resid}' for reslet, resid in
                             zip(reslets, resids)])
        times = [contacts[contacts[:, 0] == i][:, 3] for i in
                 run_resids]
        inds = np.array([np.where(resids == resid)[0][0] for resid in
                         run_resids])
        residues = residues[inds]
        input_list = [[residues[i], times[i].copy(), i % self.nproc,
                       self.ncomp, self.niter] for i in range(len(residues))]

        del contacts, times
        gc.collect()

        with (Pool(self.nproc, initializer=tqdm.set_lock,
                   initargs=(Lock(),)) as p):
            try:
                for _ in tqdm(p.istarmap(run_residue, input_list),
                              total=len(residues), position=0,
                              desc='overall progress'):
                    pass
            except KeyboardInterrupt:
                    pass


class Gibbs(object):
    """Gibbs sampler to estimate parameters of an exponential mixture for a set
    of data. Results are stored in gibbs.results, which uses /home/ricky
    MDAnalysis.analysis.base.Results(). If 'results=None' the gibbs sampler has
    not been executed, which requires calling '.run()'
    """

    def __init__(self, times=None, residue=None, loc=0, ncomp=15, niter=50000):
        self.times = times
        self.residue = residue
        self.niter = niter
        self.loc = loc
        self.ncomp = ncomp
        self.g = 100
        self.burnin = 10000
        self.processed_results = Results()

        if times is not None:
            diff = (np.sort(times)[1:]-np.sort(times)[:-1])
            try:
                self.ts = diff[diff != 0][0]
            except IndexError:
                self.ts = times.min()
        else:
            self.ts = None

        self.keys = {'times', 'residue', 'loc', 'ncomp', 'niter', 'g', 'burnin',
                     'processed_results', 'ts', 'mcweights', 'mcrates'}

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

    def _process_gibbs(self):
        from basicrta.util import mixture_and_plot
        from scipy import stats

        data_len = len(self.times)
        wcutoff = 10/data_len
        burnin_ind = self.burnin//self.g
        inds = np.where(self.mcweights[burnin_ind:] > wcutoff)
        indices = (np.arange(self.burnin, self.niter + 1, self.g)[inds[0]] //
                   self.g)
        weights, rates = self.mcweights[burnin_ind:], self.mcrates[burnin_ind:]
        fweights, frates = weights[inds], rates[inds]

        lens = [len(row[row > wcutoff]) for row in self.mcweights[burnin_ind:]]
        lmin, lmode, lmax = np.min(lens), stats.mode(lens).mode, np.max(lens)
        train_param = lmode

        pindicator = np.zeros((self.times.shape[0], train_param))
        indicator = self._sample_indicator()
        labels = mixture_and_plot(self, 'GaussianMixture', n_init=17,
                                  n_components=lmode,
                                  covariance_type='spherical')
        for j in np.unique(inds[0]):
            mapinds = labels[inds[0] == j]
            for i, indx in enumerate(inds[1][inds[0] == j]):
                tmpind = np.where(indicator[j] == indx)[0]
                pindicator[tmpind, mapinds[i]] += 1

        pindicator = (pindicator.T / pindicator.sum(axis=1)).T

        attrs = ["weights", "rates", "ncomp", "residue", "indicator", "labels",
                 "iteration", "niter"]
        values = [fweights, frates, lmode, self.residue, pindicator,
                  labels, indices, self.niter]
        self._save_results(attrs, values, processed=True)
        self._estimate_params()
        self._pickle_self()

    def _sample_indicator(self):
        indicator = np.zeros(((self.niter+1)//self.g, self.times.shape[0]),
                             dtype=np.uint8)
        burnin_ind = self.burnin//self.g
        for i, (w, r) in enumerate(zip(self.mcweights, self.mcrates)):
            # compute probabilities
            probs = w*r*np.exp(np.outer(-r, self.times)).T
            z = (probs.T/probs.sum(axis=1)).T

            # sample indicator
            s = np.argmax(rng.multinomial(1, z), axis=1)
            indicator[i] = s
        return indicator[burnin_ind:]

    def _pickle_self(self):
        with open(f'{self.residue}/gibbs_{self.niter}.pkl', 'w+b') as f:
            pickle.dump(self, f)

    def load_self(self, filename):
        with open(filename, 'r+b') as f:
            g = pickle.load(f)
        return g

    def _save_results(self, attrs, values, processed=False):
        if processed:
            r = self.processed_results
        else:
            r = self

        for attr, value in zip(attrs, values):
            setattr(r, attr, value)

        if processed:
            with (open(f'{r.residue}/processed_results_{r.niter}.pkl', 'wb')
                  as W):
                pickle.dump(r, W)
        else:
            with open(f'{r.residue}/results_{r.niter}.pkl', 'wb') as W:
                pickle.dump(r, W)

    def load_results(self, results, processed=False):
        if processed:
            with open(results, 'r+b') as f:
                r = pickle.load(f)

            for attr in list(r.keys):
                setattr(self.processed_results, attr, r[f'{attr}'])
        else:
            with open(results, 'r+b') as f:
                r = pickle.load(f)

            for attr in list(r.keys):
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
                    label=f'{i}', alpha=0.5, color=cmap(i))
         for i in range(rp.ncomp)]
        [ax[1].hist(rp.rates[rp.labels == i],
                    bins=np.exp(np.linspace(np.log(rp.rates[rp.labels == i]
                                                   .min()),
                                            np.log(rp.rates[rp.labels == i]
                                                   .max()), 50)),
                    label=f'{i}', alpha=0.5, color=cmap(i))
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
            cmap = mpl.colormaps['tab10']
            rp = self.processed_results

            fig, ax = plt.subplots(2, figsize=(4*scale, 3*scale), sharex=True)
            [ax[0].plot(rp.iteration[rp.labels == i][::sparse],
                        rp.weights[rp.labels == i][::sparse], '.',
                        label=f'{i}', color=cmap(i))
             for i in np.unique(rp.labels)]
            ax[0].set_yscale('log')
            ax[0].set_ylabel(r'weight')
            [ax[1].plot(rp.iteration[rp.labels == i][::sparse],
                        rp.rates[rp.labels == i][::sparse], '.', label=f'{i}',
                        color=cmap(i)) for i in np.unique(rp.labels)]
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

        rs = [rp.rates[rp.labels == i] for i in range(rp.ncomp)]
        ws = [rp.weights[rp.labels == i] for i in range(rp.ncomp)]
        bounds = np.array([confidence_interval(d) for d in rs])
        params = np.array([[np.mean(w), np.mean(r)] for w,r in zip(ws, rs)])

        setattr(rp, 'parameters', params)
        setattr(rp, 'intervals', bounds)

    def estimate_tau(self):
        rp = self.processed_results
        index = np.argmin(rp.parameters[:, 1])
        taus = 1 / rp.rates[rp.labels == index]
        ci = confidence_interval(taus)
        bins = np.exp(np.linspace(np.log(taus.min()), np.log(taus.max()), 100))
        H = np.histogram(taus, bins=bins)
        indmax = np.where(H[0] == H[0].max())[0]
        val = 0.5 * (H[1][:-1][indmax] + H[1][1:][indmax])[0]
        return [ci[0], val, ci[1]]

    def plot_surv(self, scale=1.5, sparse=1, save=False):
        cmap = mpl.colormaps['tab10']
        rp = self.processed_results

        ws, rs = rp.parameters[:, 0], rp.parameters[:, 1]
        fig, ax = plt.subplots(1, figsize=(4 * scale, 3 * scale))
        ax.plot(self.t, self.s, '.')
        [ax.plot(self.t, ws[i]*np.exp(-rs[i]*self.t), label=f'{i}',
                 color=cmap(i)) for i in np.unique(rp.labels)]
        ax.set_ylim(1e-6, 5)
        ax.set_yscale('log')
        ax.set_ylabel('s').set_rotation(0)
        ax.set_xlabel(r't ($ns$)')
        ax.legend(title='component')
        plt.tight_layout()
        if save:
            plt.savefig('s_vs_t.png', bbox_inches='tight')
            plt.savefig('s_vs_t.pdf', bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--contacts')
    parser.add_argument('--resid', type=int)
    args = parser.parse_args()

    contact_path = os.path.abspath(args.contacts)
    cutoff = args.contacts.split('/')[-1].strip('.pkl').split('_')[-1]
    os.chdir(f'basicrta-{cutoff}')

    ParallelGibbs(contact_path).run(run_resids=args.resid)
