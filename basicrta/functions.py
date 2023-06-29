from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import ast
import multiprocessing
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import pymbar.timeseries as pmts
from MDAnalysis.analysis.base import Results
import pickle
from glob import glob
import seaborn as sns
import math
from numpy.random import default_rng
from tqdm import tqdm
import MDAnalysis as mda
mpl.rcParams['pdf.fonttype'] = 42
rng = default_rng()

__all__ = ['gibbs', 'unique_rates', 'get_s', 'plot_results', 'plot_post',
           'plot_trace', 'collect_results', 'save_results', 
           'make_residue_plots', 'plot_protein', 'run', 'run_residue', 
           'check_results', 'get_dec', 'get_start_stop_frames',
           'write_trajs', 'plot_hists', 'get_remaining_residue_inds',
           'make_surv', 'norm_exp', 'get_dec'
           ]

def tm(Prot,i):
        dif = Prot['tm{0}'.format(i)][1]-Prot['tm{0}'.format(i)][0]
        return [Prot['tm{0}'.format(i)],dif]

class gibbs(object):
    def __init__(self, times, residue, loc, ts, ncomp, niter=10000):
        self.times, self.residue = times, residue
        self.niter, self.loc, self.ts, self.ncomp = niter, loc, ts, ncomp
    # def __repr__(self):
    #     return f'Gibbs sampler with N_comp={self.ncomp}'

    # def __str__(self):
    #     return f'Gibbs sampler with N_comp={self.ncomp}'

    def run(self):
        x, residue, niter_init = self.times, self.residue, 2500
        t, s = get_s(x, self.ts)
        if self.ncomp:
            ncomp = int(self.ncomp)
            inrates = 10 ** (np.linspace(-3, 1, ncomp))
            mcweights = np.zeros((self.niter + 1, ncomp))
            mcrates = np.zeros((self.niter + 1, ncomp))
            mcweights[0], mcrates[0] = inrates / sum(inrates), inrates
            whypers, rhypers = np.ones(ncomp) * [2], np.ones((ncomp, 2)) * [2, 1]  # guess hyperparameters
            weights, rates = [], []
            # indicator = np.memmap('indicator', dtype=float, mode='w+', shape=(ncomp, x.shape[0]))
            indicator = np.zeros((ncomp, x.shape[0]), dtype=float)
            # indicator = np.zeros((x.shape[0], ncomp), dtype=int)
            for i in tqdm(range(niter_init), desc=f'{residue}-K{ncomp}', position=self.loc, leave=False):
                tmp = mcweights[i] * norm_exp(x, mcrates[i]).T
                z = tmp.T / tmp.sum(axis=1)
                indicator += z
                Ns = z.sum(axis=1)
                mcweights[i + 1] = rng.dirichlet(whypers + Ns)
                mcrates[i + 1] = rng.gamma(rhypers[:, 0] + Ns, 1 / (rhypers[:, 1] + np.dot(z, x)))

            for i in range(ncomp):
                start = 25
                wburnin = pmts.detect_equilibration(mcweights[start:, i])[0] + start
                rburnin = pmts.detect_equilibration(mcrates[start:, i])[0] + start
                weights.append(mcweights[wburnin:, i][pmts.subsample_correlated_data(mcweights[wburnin:, i])])
                rates.append(mcrates[rburnin:, i][pmts.subsample_correlated_data(mcrates[rburnin:, i])])
            plt.close('all')
            attrs = ['weights', 'rates', 'mcweights', 'mcrates', 'ncomp', 'niter', 's', 't', 'name',
                     'indicator']
            values = [weights, rates, mcweights, mcrates, ncomp, self.niter, s, t, residue, indicator]
            r = save_results(attrs, values)
            make_residue_plots(r)
            plt.close('all')
            all_post_hist(r, save=True)
            plt.close('all')
            plot_r_vs_w(r)
        else:
            for ncomp in range(2, 8):
                inrates = 10**(np.linspace(-3, 1, ncomp))
                mcweights = np.zeros((self.niter + 1, ncomp))
                mcrates = np.zeros((self.niter + 1, ncomp))
                mcweights[0], mcrates[0] = inrates/sum(inrates), inrates
                whypers, rhypers = np.ones(ncomp) * [2], np.ones((ncomp, 2)) * [2, 1]  # guess hyperparameters
                weights, rates = [], []
                # indicator = np.memmap('indicator', dtype=float, mode='w+', shape=(ncomp, x.shape[0]))
                indicator = np.zeros((ncomp, x.shape[0]), dtype=float)
                # indicator = np.zeros((x.shape[0], ncomp), dtype=int)
                for i in tqdm(range(niter_init), desc=f'{residue}-K{ncomp}', position=self.loc, leave=False):
                    tmp = mcweights[i]*norm_exp(x, mcrates[i]).T
                    z = tmp.T / tmp.sum(axis=1)
                    indicator += z
                    Ns = z.sum(axis=1)
                    mcweights[i + 1] = rng.dirichlet(whypers + Ns)
                    mcrates[i + 1] = rng.gamma(rhypers[:, 0] + Ns, 1 / (rhypers[:, 1] + np.dot(z, x)))

                uniq_rts = unique_rates(ncomp, mcrates, niter_init, first_check=True)
                if uniq_rts != ncomp:
                    break
                else:
                    for i in tqdm(range(niter_init, self.niter), initial=niter_init, total=self.niter,
                                  desc=f'{residue}-K{ncomp}', position=self.loc, leave=False):
                        tmp = mcweights[i]*norm_exp(x, mcrates[i]).T
                        z = tmp.T / tmp.sum(axis=1)
                        indicator += z
                        Ns = z.sum(axis=1)
                        mcweights[i + 1] = rng.dirichlet(whypers + Ns)
                        mcrates[i + 1] = rng.gamma(rhypers[:, 0] + Ns, 1 / (rhypers[:, 1] + np.dot(z, x)))

                    uniq_rts = unique_rates(ncomp, mcrates, niter_init)
                    if uniq_rts == ncomp:
                        for i in range(ncomp):
                            start = 25
                            wburnin = pmts.detect_equilibration(mcweights[start:, i])[0]+start
                            rburnin = pmts.detect_equilibration(mcrates[start:, i])[0]+start
                            weights.append(mcweights[wburnin:, i][pmts.subsample_correlated_data(mcweights[wburnin:, i])])
                            rates.append(mcrates[rburnin:, i][pmts.subsample_correlated_data(mcrates[rburnin:, i])])
                        plt.close('all')
                        attrs = ['weights', 'rates', 'mcweights', 'mcrates', 'ncomp', 'niter', 's', 't', 'name', 'indicator']
                        values = [weights, rates, mcweights, mcrates, ncomp, self.niter, s, t, residue, indicator]
                        r = save_results(attrs, values)
                        make_residue_plots(r)
                        all_post_hist(r, save=True)
                    else:
                        break
                plt.close('all')


def unique_rates(ncomp, mcrates, niter_init, first_check=False):
    if first_check:
        means = mcrates[:niter_init].mean(axis=0)
        stds = mcrates[:niter_init].std(axis=0)
    else:
        means = mcrates.mean(axis=0)
        stds = mcrates.std(axis=0)
    lb, ub = means-stds, means+stds
    bools = np.empty([ncomp, ncomp])
    for j, mean in enumerate(means):
        for i in range(ncomp):
            bools[j, i] = ((mean < ub[i]) & (mean > lb[i]))
    sums = bools.sum(axis=0)
    deg_rts = sums[np.where(sums != 1)]
    return ncomp-len(deg_rts)


def get_s(x, ts):
    Bins = get_bins(x, ts)
    Hist = plt.hist(x, bins=Bins[:-1], log=True)
    t, s = make_surv(Hist)
    plt.close('all')
    return t, s

def plot_r_vs_w(r):
    plt.close()                                                               
    for k in range(r.ncomp):                                                 
        plt.plot(r.mcrates[5000:, k], r.mcweights[5000:, k], label=f'{k}')    
    plt.yscale('log')                                                         
    plt.xscale('log')                                                         
    plt.xlim(1e-4, 1)                                                         
    plt.ylim(1e-3, 1)                                                         
    plt.ylabel('weight')                                                        
    plt.xlabel('rate')                                                      
    plt.legend()                                                              
    plt.savefig(f'{r.name}/figs/k{r.ncomp}_r_vs_w.png')                                              
    plt.savefig(f'{r.name}/figs/k{r.ncomp}_r_vs_w.pdf')                                              


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
        plt.plot(results.t, weights[i] * np.exp(results.t * -rates[i]), label=f'Comp.{i}', color=f'C{i}')
    plt.yscale('log')
    plt.ylim(1e-6, 2)
    plt.legend()
    plt.ylabel('s')
    plt.xlabel('time (ns)')
    if save:
        plt.savefig(f'{outdir}/figs/k{results.ncomp}-{cond}_results.png')
        plt.savefig(f'{outdir}/figs/k{results.ncomp}-{cond}_results.pdf')
    if show:
        plt.show()
    plt.close('all')


def all_post_hist(results, save=False, show=False):
    outdir = results.name
    for attr in ['rates', 'weights']:
        Attr = getattr(results, attr)
        for i in range(results.ncomp):
            plt.hist(Attr[i], density=True, bins=25, label=f'comp. {i}', alpha=0.5)
        plt.legend()
        plt.xlabel(f'{attr}'), plt.ylabel('p')
        plt.yscale('log'), plt.xscale('log')
        if save:
            plt.savefig(f'{outdir}/figs/k{results.ncomp}-posterior_{attr}_comp-all.png')
            plt.savefig(f'{outdir}/figs/k{results.ncomp}-posterior_{attr}_comp-all.pdf')
        if show:
            plt.show()
        plt.close('all')

def plot_post(results, attr, comp=None, save=False, show=False):
    outdir = results.name
    Attr = getattr(results, attr)
    if comp:
        [plt.hist(Attr[i], density=True, bins=50, label=f'comp. {i}') for i in comp]
        plt.legend()
        if save:
            plt.savefig(f'{outdir}/figs/k{results.ncomp}-posterior_{attr}_comps-{"-".join([str(i) for i in comp])}.png')
            plt.savefig(f'{outdir}/figs/k{results.ncomp}-posterior_{attr}_comps-{"-".join([str(i) for i in comp])}.pdf')
        if show:
            plt.show()
        plt.close('all')
    else:
        for i in range(results.ncomp):
            plt.hist(Attr[i], density=True, label=f'comp. {i}')
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


def collect_results(ncomp=None):
    dirs = np.array(glob('?[0-9]*'))
    sorted_inds = np.array([int(adir[1:]) for adir in dirs]).argsort()
    dirs = dirs[sorted_inds]
    t_slow = np.zeros(len(dirs))
    sd = np.zeros(len(dirs))
    residues = np.empty((len(dirs)), dtype=object)
    indicators = []
    for i, adir in enumerate(tqdm(dirs, desc='Collecting results')):
        residues[i] = adir
        try:
            results = glob(f'{adir}/*results.pkl')
            results.sort()
            if ncomp and ncomp-1<=len(results):
                max_comp_res = results[ncomp-2]
            else:
                max_comp_res = results[-1]
        except IndexError:
            t_slow[i]=0
            continue
        with open(max_comp_res, 'rb') as W:
            tmp_res = pickle.load(W)

        means = np.array([post.mean() for post in tmp_res.rates])
        if len(means) == 0:
            continue
        ind = np.where(means == means.min())[0][0]
        t_slow[i] = 1/means[ind]
        sd[i] = tmp_res.rates[ind].std()/means[ind]**2
        indicators.append(tmp_res.indicator / tmp_res.indicator.sum(axis=0))
    return residues, t_slow, sd, indicators


def collect_n_plot(resids, comps):
    dirs = np.array(glob('?[0-9]*'))
    tmpresids = np.array([int(adir[1:]) for adir in dirs])
    sorted_inds = tmpresids.argsort()
    tmpresids.sort()
    dirs = dirs[sorted_inds]
    idinds = np.array([np.where(tmpresids == resid)[0][0] for resid in resids])
    dirs = dirs[idinds]

    for i, adir in enumerate(tqdm(dirs, desc='Collecting results')):
        results = glob(f'{adir}/*results.pkl')
        results.sort()
        max_comp_res = results[-1]
        with open(max_comp_res, 'rb') as W:
            tmp_res = pickle.load(W)

        make_residue_plots(tmp_res, comps)


def save_results(attr_names, values):
    r = Results()

    for attr, value in zip(attr_names, values):
        setattr(r, attr, value)

    if not os.path.exists(r.name):
        os.mkdir(r.name)

    with open(f'{r.name}/K{r.ncomp}_results.pkl', 'wb') as W:
        pickle.dump(r, W)

    return r


def make_residue_plots(results, comps=None, show=False):
    r = results

    if not os.path.exists(f'{r.name}/figs'):
        os.mkdir(f'{r.name}/figs/')

    plot_results(r, cond='mean', save=True, show=show)
    plot_results(r, cond='ml', save=True, show=show)
    plot_post(r, 'weights', comp=comps, save=True, show=show)
    plot_post(r, 'rates', comp=comps, save=True, show=show)
    plot_trace(r, 'weights', comp=comps, save=True, show=show)
    plot_trace(r, 'rates', comp=comps, save=True, show=show, yrange=[-0.1,1])


def plot_protein(residues, t_slow, sd, prot):
    with open('../../../../tm_dict.txt', 'r') as f:
        contents = f.read()
        prots = ast.literal_eval(contents)

    if not os.path.exists('figs'):
        os.mkdir('figs')

    height, width = 5, 7
    fig, axs = plt.subplots(2,1,figsize=(width, height),sharex=True)
    p =[Rectangle((tm(prots[prot]['helices'],i+1)[0][0],0),tm(prots[prot]['helices'],i+1)[1],1,fill=True) for i in range(7)]
    patches = PatchCollection(p)
    patches.set_color('C0')
    resids = np.array([int(res[1:]) for res in residues])
    max_inds = np.where(t_slow > 3 * t_slow.mean())
    axs[0].scatter(resids, t_slow)
    axs[0].errorbar(resids, t_slow, yerr=sd, fmt='o')
    [axs[0].text(resids[ind], t_slow[ind], residues[ind]) for ind in max_inds[0]]
    axs[1].add_collection(patches)
    if (prot=='cck1r') or (prot=='cck2r'):
        axs[0].set_ylim(0, 1300)
    else:
        axs[0].set_ylim(0, 500)
    axs[0].set_ylabel(r'$\tau_{slow}$ (ns)')
    axs[1].set_xlabel(r'residue')
    axs[0].get_xaxis().set_visible(False)
    axs[1].get_yaxis().set_visible(False)
    axs[1].set_aspect(7)
    axs[0].margins(x=0)
    sns.despine(offset=10)
    plt.subplots_adjust(hspace=-0.45,top=0.92)
    sns.despine(offset=10,ax=axs[0],bottom=True)
    sns.despine(ax=axs[1],top=True,bottom=False,left=True)
    plt.savefig('figs/t_slow.png', bbox_inches='tight')
    plt.savefig('figs/t_slow.pdf', bbox_inches='tight')


# def plot_frame_comp(indicators, trajtimes):
#     if not os.path.exists('figs'):
#         os.mkdir('figs')
#
#     plt.scatter(np.concatenate([*trajtimes]), indicators, s=2)
#     plt.ylabel('Component')
#     plt.xlabel('Frame')
#     sns.despine(offset=10)
#     plt.tight_layout()
#     plt.savefig('figs/frame_comp.png')
#     plt.savefig('figs/frame_comp.pdf')
#  ##  plt.show()


def run(gib):
    gib.run()


def run_residue(residue, time, ts, ncomp):
    x = np.array(time)
    if len(x)!=0:
        try:
            proc = int(multiprocessing.current_process().name[-1])
        except ValueError:
            proc = 1
        gib = gibbs(x, residue, proc, ts, ncomp, niter=10000)
        run(gib)


def check_results(residues, times, ts):
    if not os.path.exists('result_check'):
        os.mkdir('result_check')
    for time, residue in zip(times, residues):
        if os.path.exists(residue):
            kmax = glob(f'{residue}/K*_results.pkl')[-1].split('/')[-1].split('/')[-1].split('_')[0][1:]
            os.popen(f'cp {residue}/figs/k{kmax}-mean_results.png result_check/{residue}-k{kmax}-results.png')
        else:
            t, s = get_s(np.array(time), ts)
            plt.scatter(t, s, label='data')
            plt.ylabel('s')
            plt.xlabel('t (ns)')
            plt.legend()
            plt.title('Results unavailable')
            plt.savefig(f'result_check/{residue}-s-vs-t.png')
            plt.close('all')


# def get_frame_comps(time, indicator, trajtime):
#     inds = np.array([np.where(indicator.argmax(axis=0) == i)[0] for i in range(8)])
#     simtime = [time[inds[i]] for i in range(8)]
#     timelen = [trajtime[inds[i]] for i in range(8)]
#     return simtime, timelen


# def plot_fill(simtime, timelen, indicator):
#     bframe, eframe = get_start_stop_frames(simtime, timelen)
#     framec = (np.round(timelen, 1)*10).astype(int)
#     sortinds = bframe.argsort()
#
#     bframe.sort()
#     eframe = eframe[sortinds]
#     indicator = indicator[sortinds]
#
#     inds = np.array([np.where(indicator.argmax(axis=0) == i)[0] for i in range(8)])
#     lens = np.array([len(ind) for ind in inds])
#     ncomps = len(np.where(lens!=0)[0])
#     compframe, comptime = [bframe[ind] for ind in inds], [framec[ind] for ind in inds]
#     xvals = [np.insert(compframe[i], np.arange(len(compframe[i]))+1, compframe[i]+comptime[i]) for i in range(ncomps)]
#     cindicator = indicator.cumsum(axis=0)
#     compind = [cindicator[:ncomps][i][inds[i]] for i in range(ncomps)]
#     yvals = [np.insert(compind[i], np.arange(len(compind[i])) + 1, compind[i]) for i in range(ncomps)]


def get_dec(ts):
    if len(str(float(ts)).split('.')[1].rstrip('0')) == 0:
        dec = -len(str(ts)) + 1
    else:
        dec = len(str(float(ts)).split('.')[1].rstrip('0'))
    return dec


def get_start_stop_frames(simtime, timelen, ts):
    dec = get_dec(ts)
    framec = (np.round(timelen, dec)/ts).astype(int)
    frame = (np.round(simtime, dec)/ts).astype(int)
    return frame, frame+framec


def get_write_frames(u, time, trajtime, lipind, comp):
    dt, comp = u.trajectory.ts.dt/1000, comp-2 #nanoseconds
    bframes, eframes = get_start_stop_frames(trajtime, time, dt)
    sortinds = bframes.argsort()
    bframes.sort()
    eframes, lind = eframes[sortinds], lipind[sortinds]
    tmp = [np.arange(b, e) for b, e in zip(bframes, eframes)]
    tmpL = [np.ones_like(np.arange(b, e))*l for b, e, l in zip(bframes, eframes, lind)]
    write_frames, write_Linds = np.concatenate([*tmp]), np.concatenate([*tmpL]).astype(int)
    return write_frames, write_Linds

def write_trajs(u, time, trajtime, indicator, residue, lipind, step):
    try:
        proc = int(multiprocessing.current_process().name[-1])
    except ValueError:
        proc = 1

    prot, chol = u.select_atoms('protein'), u.select_atoms('resname CHOL')
    # dt = u.trajectory.ts.dt/1000 #nanoseconds
    inds = np.array([np.where(indicator.argmax(axis=0) == i)[0] for i in range(8)], dtype=object)
    lens = np.array([len(ind) for ind in inds])
    for comp in np.where(lens != 0)[0]:
        # bframes, eframes = get_start_stop_frames(trajtime[inds[comp]], time[inds[comp]], dt)
        # sortinds = bframes.argsort()
        # bframes.sort()
        # eframes, lind = eframes[sortinds], lipind[inds[comp]][sortinds]
        # tmp = [np.arange(b, e) for b, e in zip(bframes, eframes)]
        # tmpL = [np.ones_like(np.arange(b, e))*l for b, e, l in zip(bframes, eframes, lind)]
        # write_frames, write_Linds = np.concatenate([*tmp]), np.concatenate([*tmpL]).astype(int)
        write_frames, write_Linds = get_write_frames(u, time, trajtime, lipind, comp+2)
        if len(write_frames) > step:
            write_frames, write_Linds = write_frames[::step], write_Linds[::step]
        with mda.Writer(f"{residue}/comp{comp}_traj.xtc", len((prot+chol.residues[0].atoms).atoms)) as W:
            for i, ts in tqdm(enumerate(u.trajectory[write_frames]), desc=f"{residue}-comp{comp}", position=proc,
                              leave=False, total=len(write_frames)):
                ag = prot+chol.residues[write_Linds[i]].atoms
                W.write(ag)


def plot_hists(timelens, indicators, residues):
    for timelen, indicator, residue in tqdm(zip(timelens, indicators, residues), total=len(timelens),
                                            desc='ploting hists'):
        # framec = (np.round(timelen, 1) * 10).astype(int)
        #inds = np.array([np.where(indicator.argmax(axis=0) == i)[0] for i in range(8)], dtype=object)
        #lens = np.array([len(ind) for ind in inds])
        #ncomps = len(np.where(lens != 0)[0])
        ncomps = indicator[:,0].shape[0]

        plt.close()
        for i in range(ncomps):
            # h, edges = np.histogram(framec, density=True, bins=50, weights=indicator[i])
            h, edges = np.histogram(timelen, density=True, bins=50, weights=indicator[i])
            m = 0.5*(edges[1:]+edges[:-1])
            plt.plot(m, h, '.', label=i, alpha=0.5)
        plt.ylabel('p')
        plt.xlabel('time (ns)')
        plt.xscale('log')
        plt.yscale('log')
        plt.ylim(1e-6, 1)
        sns.despine(offset=5)
        plt.legend()
        plt.savefig(f'result_check/{residue}_hists_{ncomps}.png')
        plt.savefig(f'result_check/{residue}_hists_{ncomps}.pdf')


def get_remaining_residue_inds(residues, invert=True):
    dirs = np.array(glob('?[0-9]*'))
    resids = np.array([int(res[1:]) for res in residues])
    rem_residues = np.setdiff1d(residues, dirs)
    rem_resids = np.array([int(res[1:]) for res in rem_residues])
    rem_inds = np.in1d(resids, rem_resids, invert=invert)
    return rem_inds


def simulate_h2(n, params):
    n, [a, alp, bet] = int(n), params
    x = np.zeros(n)
    p = np.random.rand(n)

    x[p < a] = -np.log(np.random.rand(len(p[p < a])))/alp
    x[p > a] = -np.log(np.random.rand(len(p[p > a])))/bet
    x.sort()
    return x


def simulate_hn(n, weights, rates):
    n = int(n)
    x = np.zeros(n)
    p = np.random.rand(n)

    tmpw = np.concatenate(([0], np.cumsum(weights)))
    for i in range(len(weights)):
        x[(p > tmpw[i]) & (p <= tmpw[i+1])] = -np.log(np.random.rand(len(p[(p > tmpw[i]) & (p <= tmpw[i+1])])))/rates[i]
    x.sort()
    return x


def simulate_stretch_k2(n, weights, rates, exp):
    n = int(n)
    x = np.zeros(n)
    p = np.random.rand(n)

    tmpw = np.concatenate(([0], np.cumsum(weights)))

    x[(p > tmpw[0]) & (p <= tmpw[1])] = (-np.log(np.random.rand(len(p[(p > tmpw[0]) & (p <= tmpw[1])])))/rates[0])**(1/exp)
    x[(p > tmpw[1]) & (p <= tmpw[2])] = -np.log(np.random.rand(len(p[(p > tmpw[1]) & (p <= tmpw[2])])))/rates[1]

    x.sort()
    return x


def pdf_norm(x, mu, sigma):
    return np.exp(-0.5*((x-mu)/sigma)**2)/(np.sqrt(2*np.pi)*sigma)


def get_bins(x, ts):
    if isinstance(x, list):
        x = np.asarray(x)
    elif isinstance(x, np.ndarray):
        pass
    else:
        raise TypeError('Input should be a list or array')
    return np.arange(0, int(x.max()//ts)+2)*ts


def expand_times(contacts):
    a = contacts
    prots = np.unique(a[:, 0])
    lips = np.unique(a[:, 1])

    restimes = []
    Ns = []
    for i in tqdm(prots, desc='expanding times'):
        liptimes = []
        lipNs = []
        for j in lips:
            tmp = a[(a[:, 0] == i) & (a[:, 1] == j)]
            liptimes.append(np.round(tmp[:, 2], 1))
            lipNs.append(tmp[:, 3])
        restimes.append(liptimes)
        Ns.append(lipNs)
    times = np.asarray(restimes)
    Ns = np.asarray(Ns)

    alltimes = []
    for res in tqdm(range(times.shape[0])):
        restimes = []
        for lip in range(times.shape[1]):
            for i in range(times[res, lip].shape[0]):
                [restimes.append(j) for j in [times[res, lip][i]]*Ns[res, lip][i].astype(int)]
        alltimes.append(restimes)
    return np.asarray(alltimes)


def make_surv(ahist, max=1):
    y = ahist[0][ahist[0] != 0]
    tmpbin = ahist[1][1:]
    t = tmpbin[ahist[0] != 0]
    t = np.insert(t, 0, 0)
    y = np.cumsum(y)
    y = np.insert(y, 0, max)
    y = y/y[-1]
    s = 1-y
    return t, s


def stretch_exp_p(x, bet, lamda):
    return bet*lamda*x**(lamda-1)*np.exp(-bet*x**lamda)


def stretch_exp_s(x, bet, lamda):
    return np.exp(-bet*x**lamda)


def surv(x, weights, rates):
    return np.inner(weights, np.exp(np.outer(x, -rates)))


def surv2(x, w1, r1, r2):
    return w1*np.exp(-x*r1)+(1-w1)*np.exp(x*-r2)


def prob(x, weights, rates):
    return np.inner(weights*rates, np.exp(np.outer(x, -rates)))


def lnL(data, weights, rates):
    return np.sum(np.log(prob(data, weights, rates)))


def make_F(data):
    return lambda vars: -lnL(data, vars[0], vars[1])


def make_jac(x):
    return lambda params: jac(x, params)


def jac(x, params):
    a, alp, bet = params
    w1 = np.sum((alp*np.exp(-alp*x)-bet*np.exp(-bet*x))/prob(x, params))
    l1 = np.sum((a*np.exp(-alp*x)*(1-alp**2))/prob(x, params))
    l2 = np.sum((-(1-a)*np.exp(-bet*x)*(1-bet**2))/prob(x, params))
    tmparr = np.array((w1, l1, l2))
    return np.transpose(tmparr)


def norm_exp(x, rates):
    return np.array([rate*np.exp(-rate*x) for rate in rates])

# @njit
# def norm_exp_numba(x, rates):
#     tmparr = np.zeros((len(x), len(rates)))
#     for i, rate in enumerate(rates):
#         tmparr[:, i] = rate*np.exp(-rate*x)
#     return tmparr


def exp(x, rates):
    return np.asarray([np.exp(-rate*x) for rate in rates])


def w_prior(x, alp, bet):
    return lambda a: np.sqrt(sum((exp(x, alp)-exp(x, bet))**2/(a*exp(x, alp)+(1-a)*exp(x, bet))))


def approx_omega(x, lamda1, lamda2):
    lamdaA, lamda0 = lamda2/(lamda1-1), lamda1-1
    L = len(x)
    return np.log(np.sqrt(2*np.pi/(lamda0+L)))+(L+lamda0)*np.log((lamda0+L)/(lamdaA*lamda0+x.sum()))+\
           np.log((lamda0*lamdaA)**(lamda0+1)/(math.factorial(lamda0)*(lamdaA*lamda0+x.sum())))

