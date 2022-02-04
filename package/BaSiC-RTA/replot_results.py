import multiprocessing

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from setup_L import *
from numpy.random import default_rng
import os
import pymbar.timeseries as pmts
from MDAnalysis.analysis.base import Results
import MDAnalysis as mda
import pickle
import gc
from glob import glob
import seaborn as sns
from multiprocessing import Pool, Lock
from tqdm.contrib.concurrent import process_map
from multiprocessing import Manager
from functools import partial
from p_tqdm import p_map
mpl.rcParams['pdf.fonttype'] = 42

class gibbs(object):
    def __init__(self, times, residue, loc, niter=10000):
        self.times, self.residue = times, residue
        self.niter, self.loc = niter, loc

        # self.ncomp, self.niter, self.data = ncomp, niter, data
        # self.rhypers = np.ones((ncomp, 2))*[2, 1]  # guess hyperparameters
        # self.whypers = np.ones(ncomp)*[2]  # guess hyperparameters
        # self.weights, self.rates = [], []
        # self.mcweights = np.zeros((self.niter+1, self.ncomp))
        # self.mcrates = np.zeros((self.niter+1, self.ncomp))
        # self.name = name
        # self.complete = False

    # def __repr__(self):
    #     return f'Gibbs sampler with N_comp={self.ncomp}'

    # def __str__(self):
    #     return f'Gibbs sampler with N_comp={self.ncomp}'

    def run(self):
        x, residue = self.times, self.residue
        t, s = get_s(x)
        for ncomp in range(2, 8):
            inrates = 10**(np.linspace(-3, 1, ncomp))
            mcweights = np.zeros((self.niter+1, ncomp))
            mcrates = np.zeros((self.niter+1, ncomp))
            mcweights[0], mcrates[0] = inrates/sum(inrates), inrates
            whypers, rhypers = np.ones(ncomp)*[2], np.ones((ncomp, 2))*[2, 1]  # guess hyperparameters
            weights, rates = [], []
            for i in tqdm(range(5000), desc=f'{residue}-K{ncomp}', position=self.loc, leave=False):
                tmp = mcweights[i]*norm_exp(x, mcrates[i]).T
                z = tmp.T / tmp.sum(axis=1)
                Ns = z.sum(axis=1)
                mcweights[i + 1] = rng.dirichlet(whypers + Ns)
                mcrates[i + 1] = rng.gamma(rhypers[:, 0] + Ns, 1 / (rhypers[:, 1] + np.dot(z, x)))
            # self.z = s

            uniq_rts = unique_rates(ncomp, mcrates, first_check=True)
            if uniq_rts != ncomp:
                break
            else:
                for i in tqdm(range(5000, self.niter), initial=5000, total=self.niter, desc=f'{residue}-K{ncomp}',
                              position=self.loc, leave=False):
                    tmp = mcweights[i]*norm_exp(x, mcrates[i]).T
                    z = tmp.T / tmp.sum(axis=1)
                    Ns = z.sum(axis=1)
                    mcweights[i + 1] = rng.dirichlet(whypers + Ns)
                    mcrates[i + 1] = rng.gamma(rhypers[:, 0] + Ns, 1 / (rhypers[:, 1] + np.dot(z, x)))
                # self.z = s

                uniq_rts = unique_rates(ncomp, mcrates)
                if uniq_rts == ncomp:
                    for i in range(ncomp):
                        start = 25
                        # print(f'Detecting equilibration: component {i+1}/{ncomp}')
                        wburnin = pmts.detectEquilibration(mcweights[start:, i])[0]+start
                        rburnin = pmts.detectEquilibration(mcrates[start:, i])[0]+start
                        weights.append(mcweights[wburnin:, i][pmts.subsampleCorrelatedData(mcweights[wburnin:, i])])
                        rates.append(mcrates[rburnin:, i][pmts.subsampleCorrelatedData(mcrates[rburnin:, i])])
                    plt.close('all')
                    attrs = ['weights', 'rates', 'mcweights', 'mcrates', 'ncomp', 'niter', 's', 't', 'name']
                    values = [weights, rates, mcweights, mcrates, ncomp, self.niter, s, t, residue]
                    save_results(attrs, values)
                else:
                    break
                    # print('Degenerate rates detected')
            plt.close('all')


def unique_rates(ncomp, mcrates, first_check=False):
    if first_check:
        means = mcrates[:5001].mean(axis=0)
        stds = mcrates[:5001].std(axis=0)
    else:
        means = mcrates.mean(axis=0)
        stds = mcrates.std(axis=0)
    lb, ub = means-stds, means+stds
    bools = np.empty([ncomp, ncomp])
    for j, mean in enumerate(means):
        for i in range(ncomp):
            bools[j, i] = ((mean < ub[i]) & (mean > lb[i]))
    sums = bools.sum(axis=0)
    deg_rts = sums[np.where(sums!=1)]
    return ncomp-len(deg_rts)


def get_s(x):
    Bins = get_bins(x, ts)
    Hist = plt.hist(x, bins=Bins[:-1], log=True)
    t, s = make_surv(Hist)
    plt.close('all')
    return t, s


def plot_results(results, cond='mean', save=False, show=False):
    outdir = results.name
    weight_posts = getattr(results, 'weights')
    rate_posts = getattr(results, 'rates')
    w_hists = [plt.hist(post, density=True, bins=50) for post in weight_posts]
    r_hists = [plt.hist(post, density=True, bins=50) for post in rate_posts]
    plt.close('all')
    if cond == 'mean':
        weights = np.array([w.mean() for w in results.weights])
        weights = weights/weights.sum()
        rates = np.array([r.mean() for r in results.rates])
    elif cond == 'ml':
        mlw, mlr = [], []
        for i in range(results.ncomp):
            # tmpw, tmpr = plt.hist(weights[i], density=True), plt.hist(rates[i], density=True)
            mlw.append(w_hists[i][1][w_hists[i][0].argmax()])
            mlr.append(r_hists[i][1][r_hists[i][0].argmax()])
        mlw = np.array(mlw)
        weights = mlw/mlw.sum()
        rates = np.array(mlr)
    else:
        raise ValueError('Only implemented for most likely (ml) and mean')

    plt.scatter(results.t, results.s, label='data')
    plt.plot(results.t, np.inner(weights, np.exp(np.outer(results.t, -rates))), label='fit', color='yellow')
    for i in range(results.ncomp):
        plt.plot(results.t, weights[i]*np.exp(results.t*-rates[i]), label=f'Comp.{i}', color=f'C{i}')
    plt.yscale('log')
    plt.ylim(1e-6, 2)
    plt.legend()
    if save:
        plt.savefig(f'{outdir}/figs/k{results.ncomp}-{cond}_results.png')
        plt.savefig(f'{outdir}/figs/k{results.ncomp}-{cond}_results.pdf')
    if show:
        plt.show()
    plt.close('all')


def plot_post(results, attr, comp=None, save=False, show=False):
    outdir = results.name
    Attr = getattr(results, attr)
    if comp:
        plt.hist(Attr[comp], density=True, bins=50, label=f'comp. {i}')
        plt.legend()
        if save:
            plt.savefig(f'{outdir}/figs/k{results.ncomp}-posterior_{attr}_comps-{"-".join([str(i) for i in comp])}.png')
            plt.savefig(f'{outdir}/figs/k{results.ncomp}-posterior_{attr}_comps-{"-".join([str(i) for i in comp])}.pdf')
        if show:
            plt.show()
        plt.close('all')
    else:
        for i in range(results.ncomp):
            plt.hist(Attr[i], density=True, bins=50, label=f'comp. {i}')
            plt.legend()
            if save:
                plt.savefig(f'{outdir}/figs/k{results.ncomp}-posterior_{attr}_comp-{i}.png')
                plt.savefig(f'{outdir}/figs/k{results.ncomp}-posterior_{attr}_comp-{i}.pdf')
            if show:
                plt.show()
            plt.close('all')


def plot_trace(results, attr, comp=None, xrange=None, yrange=None, save=False, show=False):
    outdir = results.name
    if attr=='weights':
        tmp = getattr(results, 'mcweights')
    elif attr=='rates':
        tmp = getattr(results, 'mcrates')
    if not comp:
        for j in range(results.ncomp):
            plt.plot(range(tmp.shape[0]), tmp[:, j], label=f'Comp. {j}')
        plt.xlabel('iteration')
        plt.ylabel(f'{attr}')
        # plt.title(f'Component {i+1}')
        plt.legend()
        if xrange!=None:
            plt.xlim(xrange[0], xrange[1])
        if yrange!=None:
            plt.ylim(yrange[0], yrange[1])
        if save:
            plt.savefig(f'{outdir}/figs/k{results.ncomp}-trace_{attr}.png')
            plt.savefig(f'{outdir}/figs/k{results.ncomp}-trace_{attr}.pdf')
    if comp:
        for i in comp:
            plt.plot(range(tmp.shape[0]), tmp[:, i], label=f'Comp. {i}')
            plt.xlabel('iteration')
            plt.ylabel(f'{attr}')
            # plt.title(f'Component {i+1}')
            plt.legend()
        if xrange!=None:
            plt.xlim(xrange[0], xrange[1])
        if yrange!=None:
            plt.ylim(yrange[0], yrange[1])
        if save:
            plt.savefig(f'{outdir}/figs/k{results.ncomp}-trace_{attr}_comps-{"-".join([str(i) for i in comp])}.png')
            plt.savefig(f'{outdir}/figs/k{results.ncomp}-trace_{attr}_comps-{"-".join([str(i) for i in comp])}.pdf')
    if show:
        plt.show()
    plt.close('all')


def collect_results():
    dirs = np.array(glob("*/"))
    sorted_inds = np.array([int(dir[1:-1]) for dir in dirs]).argsort()
    dirs = dirs[sorted_inds]
    t_slow = np.zeros(len(dirs))
    sd = np.zeros(len(dirs))
    for i, adir in enumerate(tqdm(dirs, desc='Collecting results')):
        try:
            max_comp_res = glob(f'{adir}/*results.pkl')[-1]
        except IndexError:
            t_slow[i]=0
            continue
        with open(max_comp_res, 'rb') as W:
            tmp_res = pickle.load(W)
        means = np.array([post.mean() for post in tmp_res.rates])
        if len(means) == 0:
            continue
        ind = np.where(means==means.min())[0][0]
        t_slow[i] = 1/means[ind]
        sd[i] = tmp_res.rates[ind].std()/means[ind]**2
    return t_slow, sd


def save_results(attr_names, values):
    r = Results()
    for i, attr in enumerate(attr_names):
        setattr(r, attr, values[i])

    if not os.path.exists(r.name):
        os.mkdir(r.name)
        os.mkdir(f'{r.name}/figs/')

    plot_results(r, cond='mean', save=True)
    plot_results(r, cond='ml', save=True)
    plot_post(r, 'weights', save=True)
    plot_post(r, 'rates', save=True)
    plot_trace(r, 'weights', save=True)
    plot_trace(r, 'rates', save=True)

    with open(f'{r.name}/K{r.ncomp}_results.pkl', 'wb') as W:
        pickle.dump(r, W)

    return r

def plot_protein(residues, t_slow, sd):
    resids = np.array([int(res[1:]) for res in residues])
    max_inds = np.where(t_slow>3*t_slow.mean())
    plt.scatter(resids, t_slow)
    plt.errorbar(resids, t_slow, yerr=sd, fmt='o')
    [plt.text(resids[ind], t_slow[ind], residues[ind]) for ind in max_inds[0]]
    # plt.text(resids[max_inds[0]], t_slow[max_inds[0]], residues[max_inds[0]])
    plt.ylabel(r'$\tau_{slow}$').set_rotation(0)
    plt.xlabel(r'residue')
    sns.despine(offset=10)
    plt.tight_layout()
    plt.show()

def run(gib):
    gib.run()

def run_residue(residue, time):
    x = np.array(time)
    try:
        proc = int(multiprocessing.current_process().name[-1])
    except ValueError:
        proc = 1
    gib = gibbs(x, residue, proc, niter=10000)
    run(gib)

def replot_results(residue):
    # for time, residue in zip(times, residues):
    if os.path.exists(residue):
        kmax = glob(f'{residue}/K*_results.pkl')[-1].split('/')[-1][1]
        with open(f'{residue}/K{kmax}_results.pkl', 'rb') as w:
            r = pickle.load(w)
        plot_results(r, cond='mean', save=True)


def check_results(residues, times):
    if not os.path.exists('result_check'):
        os.mkdir('result_check')
    for time, residue in zip(times, residues):
        if os.path.exists(residue):
            kmax = glob(f'{residue}/K*_results.pkl')[-1].split('/')[-1][1]
            os.popen(f'cp {residue}/figs/k{kmax}-mean_results.png result_check/{residue}-k{kmax}-results.png')
        else:
            t, s = get_s(np.array(time))
            plt.scatter(t, s, label='data')
            plt.yscale('log')
            plt.ylabel('s')
            plt.xlabel('t (ns)')
            plt.legend()
            plt.title('Results unavailable')
            plt.savefig(f'result_check/{residue}-s-vs-t.png')
            plt.close('all')


if __name__ == "__main__":
    # Parts of code taken from Shep (Centrifuge3.py, SuperMCMC.py)

    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--contacts')
    # args = parser.parse_args()
    # a = np.load(args.contacts)
    a = np.load('lipswap_contacts_7.0.npy')
    if os.path.exists('expanded_times.npy'):
        times = np.load('expanded_times.npy', allow_pickle=True)
    else:
        times = expand_times(a)
        np.save('expanded_times', times)

    rng = default_rng()
    nproc = 2
    N, ts, wtol, rtol = 1e5, 0.1, 1e-5, 1e-6
    u = mda.Universe('step7_fixed.pdb')
    ids = u.select_atoms('protein').residues.resids
    names = u.select_atoms('protein').residues.resnames
    names = np.array([mda.lib.util.convert_aa_code(name) for name in names])
    resids, resnames = ids[np.unique(a[:, 0]).astype(int)], names[np.unique(a[:, 0]).astype(int)]
    residues = np.array([f'{resnames[i]}{resids[i]}' for i in range(len(resids))])
    restart = False


    os.chdir('BaSiC-RTA')

    Pool(nproc, initializer=tqdm.set_lock, initargs=(Lock(),)).map(replot_results, residues)
    check_results(residues, times)

