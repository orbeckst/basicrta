"""Functions used by other modules."""

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import ast, multiprocessing, os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle, bz2
from glob import glob
import seaborn as sns
from tqdm import tqdm
import MDAnalysis as mda
from scipy.optimize import linear_sum_assignment as lsa


mpl.rcParams['pdf.fonttype'] = 42


def siground(x, dec):
    return float(f'%.{dec - 1}e' % x)


def slice_trajectory(u, nslices):
    if type(u) == MDAnalysis.coordinates.base.FrameIteratorSliced:
        frames = np.arange(u.start, u.stop, u.step)
    elif type(u) == MDAnalysis.coordinates.base.FrameIteratorIndices:
        frames = u.frames
    else:
        frames = np.arange(len(u.trajectory))

    sliced_frames = np.array_split(frames, nslices)
    return sliced_frames


def KL_resort(r):
    mcweights, mcrates = r.mcweights.copy(), r.mcrates.copy()
    indicator[:] = indicator_bak
    Ls, niter = [L], 0
    for j in tqdm(range(r.niter)):
        sorts = mcweights[j].argsort()[::-1]
        mcweights[j] = mcweights[j][sorts]
        mcrates[j] = mcrates[j][sorts]

    while niter<10:
        Z = np.zeros_like(z)
        for j in tqdm(range(2000, 3000), desc='recomputing Q'):
            tmp = mcweights[j]*mcrates[j]*np.exp(np.outer(-mcrates[j],x)).T
            z = (tmp.T/tmp.sum(axis=1)).T
            Z += z
        Z = Z/1000

        for j in tqdm(range(2000, 3000), desc='resorting'):
            tmp = mcweights[j]*mcrates[j]*np.exp(np.outer(-mcrates[j],x)).T
            z = (tmp.T/tmp.sum(axis=1)).T

            tmpsum = np.ones((ncomp,ncomp), dtype=np.float64)
            for k in range(ncomp):
                tmpsum[k] = np.sum(z[:,k]*np.log(z[:,k]/Z.T), axis=1)

            tmpsum[tmpsum!=tmpsum] = 1e20
            sorts = lsa(tmpsum)[1]
            mcweights[j] = mcweights[j][sorts]
            mcrates[j] = mcrates[j][sorts]
        niter += 1


def tm(Prot,i):
        dif = Prot['tm{0}'.format(i)][1]-Prot['tm{0}'.format(i)][0]
        return [Prot['tm{0}'.format(i)],dif]


def confidence_interval(data, percentage=95):
    ds = np.sort(data)
    perc = np.arange(1, len(ds)+1)/(len(ds))
    lower, upper = (100-percentage)/200, (percentage+(100-percentage)/2)/100

    try:
        l = ds[np.where(perc <= lower)[0][-1]]
    except IndexError:
        l = ds[0]
    
    try:
        u = ds[np.where(perc >= upper)[0][0]]
    except IndexError:
        u = ds[-1]

    return [l, u]


def get_bars(tau):
    maxs = tau[:,1]
    lb, ub = tau[:,0], tau[:,2]
    return np.array([maxs-lb, ub-maxs])


def unique_rates(ncomp, mcrates):
    mclen = len(mcrates)*9//10
    means = mcrates[mclen:].mean(axis=0)
    stds = mcrates[mclen:].std(axis=0)
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
    Hist = np.histogram(x, bins=Bins)
    t, s = make_surv(Hist)
    return t, s


def plot_r_vs_w(r, rrange=None, wrange=None):
    plt.close()                                                               
    plt.figure(figsize=(4,3))
    for k in range(r.ncomp):                                                 
        plt.plot(r.mcrates[:, k], r.mcweights[:, k], label=f'{k}')
    plt.yscale('log')                                                         
    plt.xscale('log')                      
    if rrange:
        plt.xlim(*rrange)
    if wrange:
        plt.ylim(*wrange)                                                         
    plt.ylabel('weight')                                                        
    plt.xlabel('rate')                                                      
    plt.legend(loc='upper left')                                                              
    plt.savefig(f'{r.name}/figs/k{r.ncomp}_r_vs_w.png')                                              
    plt.savefig(f'{r.name}/figs/k{r.ncomp}_r_vs_w.pdf')


def plot_r_vs_w(weights, rates, labels, rrange=None, wrange=None):
    plt.close()
    plt.figure(figsize=(4, 3))
    ncomp = len(np.unique(labels))
    for k in range(ncomp):
        inds = np.where(labels == k)[0]
        plt.plot(rates[inds], weights[inds], '.', label=f'{k}')
    plt.yscale('log')
    plt.xscale('log')
    if rrange:
        plt.xlim(*rrange)
    if wrange:
        plt.ylim(*wrange)
    plt.ylabel('weight')
    plt.xlabel('rate')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()


def get_color(i):
    if i<0:
        color = i
    else:
        color = i % 20
    return color


def plot_results(results, cond='ml', save=False, show=False):
    outdir = results.name
    sortinds = np.argsort([line.mean() for line in results.rates])
    
    weight_posts = np.array(getattr(results, 'weights'), dtype=object)[sortinds]
    rate_posts = np.array(getattr(results, 'rates'), dtype=object)[sortinds]
    w_hists = [plt.hist(post, density=True) for post in weight_posts]
    r_hists = [plt.hist(post, density=True) for post in rate_posts]
    plt.close('all')
    if cond == 'mean':
        weights = np.array([w.mean() for w in results.weights])
        weights = weights/weights.sum()
        rates = np.array([r.mean() for r in results.rates])
    elif cond == 'ml':
        mlw, mlr = [], []
        for i in range(results.ncomp):
            mlw.append(w_hists[i][1][w_hists[i][0].argmax()])
            mlr.append(r_hists[i][1][r_hists[i][0].argmax()])
        mlw = np.array(mlw)
        weights = mlw/mlw.sum()
        rates = np.array(mlr)
    else:
        raise ValueError('Only implemented for most likely (ml) and mean')

    fig, axs = plt.subplots(figsize=(4,3))
    plt.scatter(results.t, results.s, s=15, label='data')
    plt.plot(results.t, np.inner(weights, np.exp(np.outer(results.t, -rates))),\
            label='fit', color='y', ls='dashed', lw=3)
    for i in range(results.ncomp):
        plt.plot(results.t, weights[i] * np.exp(results.t * -rates[i]), \
                 label=f'Comp.{i}', color=f'C{i}')
    plt.plot([], [], ' ', label=rf'$\tau$={np.round(1/rates.min(), 1)} ns')
    plt.yscale('log')
    plt.ylim(0.8*results.s[-2], 2)
    plt.xlim(-0.05*results.t[-2], 1.1*results.t[-2])
    plt.legend()
    plt.ylabel('s').set_rotation(0)
    plt.xlabel('time (ns)')
    plt.tight_layout()
    sns.despine(offset=3, ax=axs)
    if save:
        plt.savefig(f'{outdir}/figs/k{results.ncomp}-{cond}_results.png')
        plt.savefig(f'{outdir}/figs/k{results.ncomp}-{cond}_results.pdf')
    if show:
        plt.show()
    plt.close('all')


def all_post_hist(results, save=False, show=False, wlims=None, rlims=None):
    outdir = results.name
    for attr, unit in [['rates', ' (ns$^{-1}$)'], ['weights', '']]:
        Attr = getattr(results, attr)
        plt.figure(figsize=(4,3))
        for i in range(results.ncomp):
            plt.hist(Attr[i], density=True, bins=15, label=f'comp. {i}', \
                     alpha=0.5)
        plt.legend()
        plt.xlabel(f'{attr}{unit}'), plt.ylabel('p').set_rotation(0)
        plt.yscale('log'), plt.xscale('log')
        if attr=='rates' and rlims:
            plt.xlim(rlims[0])
            plt.ylim(rlims[1])
        if attr=='weights' and wlims:
            plt.xlim(wlims[0])
            plt.ylim(wlims[1])
        if save:
            name = f'{outdir}/figs/k{results.ncomp}-posterior_{attr}_comp-all'
            plt.savefig(f'{name}.png', bbox_inches='tight')
            plt.savefig(f'{name}.pdf', bbox_inches='tight')
        if show:
            plt.show()
        plt.close('all')


def plot_post(results, attr, comp=None, save=False, show=False):
    outdir = results.name
    Attr = getattr(results, attr)
    if attr == 'rates':
        unit=' (ns$^{-1}$)'
    else:
        unit=''

    if comp:
        [plt.hist(Attr[i], density=True, bins=50, label=f'comp. {i}') for i in comp]
        plt.legend()
        if save:
            plt.savefig(f'{outdir}/figs/k{results.ncomp}-posterior_{attr}_\
                           comps-{"-".join([str(i) for i in comp])}.png')
            plt.savefig(f'{outdir}/figs/k{results.ncomp}-posterior_{attr}_\
                           comps-{"-".join([str(i) for i in comp])}.pdf')
        if show:
            plt.show()
        plt.close('all')
    else:
        for i in range(results.ncomp):
            plt.close()
            fig, ax = plt.subplots(figsize=(4,3))
            plt.hist(Attr[i], density=True, bins=15, label=f'comp. {i}')
            #plt.legend()
            plt.ylabel('p').set_rotation(0)
            plt.xlabel(rf'{attr[:-1]} {unit}')
            ax.xaxis.major.formatter._useMathText = True
            if save:
                plt.savefig(f'{outdir}/figs/k{results.ncomp}-posterior_{attr}_'
                               'comp-{i}.png', bbox_inches='tight')
                plt.savefig(f'{outdir}/figs/k{results.ncomp}-posterior_{attr}_'
                               'comp-{i}.pdf', bbox_inches='tight')
            if show:
                plt.show()


def plot_trace(results, attr, comp=None, xrange=None, yrange=None, save=False,
               show=False):
    outdir = results.name
    if attr=='weights':
        tmp = getattr(results, 'mcweights')
    elif attr=='rates':
        tmp = getattr(results, 'mcrates')
    if not comp:
        plt.figure(figsize=(4,3))
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
        plt.figure(figsize=(4,3))
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
            plt.savefig(f'{outdir}/figs/k{results.ncomp}-trace_{attr}_comps-\
                          {"-".join([str(i) for i in comp])}.png')
            plt.savefig(f'{outdir}/figs/k{results.ncomp}-trace_{attr}_comps-\
                          {"-".join([str(i) for i in comp])}.pdf')
    if show:
        plt.show()
    plt.close('all')


def collect_results(ncomp=None):
    """returns (residues, tslow, stds)
    """
    dirs = np.array(glob('?[0-9]*'))
    sorted_inds = np.array([int(adir[1:]) for adir in dirs]).argsort()
    dirs = dirs[sorted_inds]
    t_slow = np.zeros(len(dirs))
    sd = np.zeros((len(dirs),2))
    residues = np.empty((len(dirs)), dtype=object)
    indicators = []
    for i, adir in enumerate(tqdm(dirs, desc='Collecting results')):
        residues[i] = adir
        try:
            tmp_res = pickle.load(bz2.BZ2File(f'{adir}/results_20000.pkl.bz2', 'rb'))
            tmp_res, rpinds = process_gibbs(tmp_res)
        #    with open(f'{adir}/processed_results_10000.pkl', 'rb') as f:
        #        tmp_res = pickle.load(f)
        #    results = glob(f'{adir}/*results.pkl')
        #    results.sort()
        #    if ncomp and ncomp-1<=len(results):
        #        max_comp_res = results[ncomp-2]
        #    else:
        #        max_comp_res = results[-1]
        except FileNotFoundError:
            t_slow[i]=0
            continue
        #with open(max_comp_res, 'rb') as W:
        #    tmp_res = pickle.load(W)
        

        means = np.array([(1/post).mean() for post in tmp_res.rates.T])
        if len(means) == 0:
            continue
        ind = np.where(means == means.max())[0][0]
        t_slow[i] = means[ind]
        sd[i] = get_bars(1/tmp_res.rates.T[ind])
        indicators.append((tmp_res.indicator.T/tmp_res.indicator.sum(axis=1)).T)
    return residues, t_slow, sd.T, indicators


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
        #max_comp_res = results[-1]
        for res in results:
            with open(res, 'rb') as W:
                tmp_res = pickle.load(W)

            make_residue_plots(tmp_res, comps)
            all_post_hist(tmp_res, save=True, rlims=[[1e-3,10],[1e-2, 1e3]],
                          wlims=[[1e-4, 1.1],[1e-1, 1e4]])
            plot_r_vs_w(tmp_res, rrange=[1e-3, 10], wrange=[1e-4, 5])


def make_residue_plots(results, comps=None, show=False):
    r = results

    if not os.path.exists(f'{r.name}/figs'):
        os.mkdir(f'{r.name}/figs/')

    plot_results(r, cond='mean', save=True, show=show)
    plot_results(r, cond='ml', save=True, show=show)
    plot_post(r, 'weights', comp=comps, save=True, show=show)
    plot_post(r, 'rates', comp=comps, save=True, show=show)
    plot_trace(r, 'weights', comp=comps, save=True, show=show, yrange=[-0.1,1.1])
    plot_trace(r, 'rates', comp=comps, save=True, show=show, yrange=[-0.1,6])


def plot_protein(residues, t_slow, bars, prot, label_cutoff=3, ylim=None):
    with open('../../../../tm_dict.txt', 'r') as f:
        contents = f.read()
        prots = ast.literal_eval(contents)

    if not os.path.exists('figs'):
        os.mkdir('figs')

    height, width = 3, 4
    fig, axs = plt.subplots(2,1,figsize=(width, height),sharex=True)
    axs[0].tick_params(axis='both', which='major', labelsize=10)
    axs[1].tick_params(axis='both', which='major', labelsize=10)
    p =[Rectangle((tm(prots[prot]['helices'], i+1)[0][0], 0),
        tm(prots[prot]['helices'], i+1)[1], 1, fill=True) for i in range(7)]
    patches = PatchCollection(p)
    patches.set_color('C0')
    resids = np.array([int(res[1:]) for res in residues])
    max_inds = np.where(t_slow > label_cutoff * t_slow.mean())
    axs[0].plot(resids, t_slow, '.', color='C0')
    axs[0].errorbar(resids, t_slow, yerr=bars, fmt='none', color='C0',
                    alpha=0.5)
    [axs[0].text(resids[ind], t_slow[ind], residues[ind]) for ind in
     max_inds[0]]
    axs[1].add_collection(patches)
    #if (prot=='cck1r') or (prot=='cck2r'):
    #    axs[0].set_ylim(0, 1300)
    #else:
    #    axs[0].set_ylim(0, 500)
    axs[0].set_ylabel(r'$\tau$ [ns]')
    axs[1].set_xlabel(r'residue')
    axs[0].get_xaxis().set_visible(False)
    axs[1].get_yaxis().set_visible(False)
    axs[1].xaxis.set_major_locator(MultipleLocator(100))
    axs[0].yaxis.set_major_locator(MultipleLocator(400))
    # axs[1].xaxis.set_minor_locator(MultipleLocator(10))
    axs[1].set_aspect(7)
    axs[0].margins(x=0)
    axs[1].set_ylim(ylim)
    plt.subplots_adjust(hspace=-0.45, top=0.92)
    sns.despine(offset=10, ax=axs[0], bottom=True)
    sns.despine(ax=axs[1], top=True, bottom=False, left=True)
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


# def run(gib):
#     gib.run()


def run_residue(residue, time, proc, ncomp, niter):
    from basicrta.gibbs import Gibbs
    x = np.array(time)
    if len(x) != 0:
        try:
            proc = int(multiprocessing.current_process().name.split('-')[-1])
        except ValueError:
            proc = 1

    gib = Gibbs(x, residue, proc, ncomp=ncomp, niter=niter)
    gib.run()


def check_results(residues, times, ts):
    if not os.path.exists('result_check'):
        os.mkdir('result_check')
    for time, residue in zip(times, residues):
        if os.path.exists(residue):
            kmax = glob(f'{residue}/K*_results.pkl')[-1].split('/')[-1].\
                        split('/')[-1].split('_')[0][1:]
            os.popen(f'cp {residue}/figs/k{kmax}-mean_results.png result_check/\
                       {residue}-k{kmax}-results.png')
        else:
            t, s = get_s(np.array(time), ts)
            plt.scatter(t, s, label='data')
            plt.ylabel('s')
            plt.xlabel('t (ns)')
            plt.legend()
            plt.title('Results unavailable')
            plt.savefig(f'result_check/{residue}-s-vs-t.png')
            plt.close('all')


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
        write_frames, write_Linds = get_write_frames(u, time, trajtime, lipind, comp+2)
        if len(write_frames) > step:
            write_frames = write_frames[::step]
            write_Linds = write_Linds[::step]
        with mda.Writer(f"{residue}/comp{comp}_traj.xtc", \
                len((prot+chol.residues[0].atoms).atoms)) as W:
            for i, ts in tqdm(enumerate(u.trajectory[write_frames]), \
                              desc=f"{residue}-comp{comp}", position=proc, \
                              leave=False, total=len(write_frames)):
                ag = prot+chol.residues[write_Linds[i]].atoms
                W.write(ag)


def plot_hists(timelens, indicators, residues):
    for timelen, indicator, residue in tqdm(zip(timelens, indicators, residues),
                                            total=len(timelens), 
                                            desc='ploting hists'):
        ncomps = indicator[:,0].shape[0]

        plt.close()
        for i in range(ncomps):
            h, edges = np.histogram(timelen, density=True, bins=50, \
                                    weights=indicator[i])
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


def simulate_hn(n, weights, rates):
    n = int(n)
    x = np.zeros(n)
    p = np.random.rand(n)

    tmpw = np.concatenate(([0], np.cumsum(weights)))
    for i in range(len(weights)):
        x[(p > tmpw[i]) & (p <= tmpw[i+1])] = \
        -np.log(np.random.rand(len(p[(p > tmpw[i]) & (p <= tmpw[i+1])])))/\
        rates[i]
    x.sort()
    return x


def make_surv(ahist):
    y = ahist[0][ahist[0] != 0]
    tmpbin = ahist[1][:-1]
    t = tmpbin[ahist[0] != 0]
    t = np.insert(t, 0, 0)
    y = np.cumsum(y)
    y = np.insert(y, 0, 0)
    y = y/y[-1]
    s = 1-y
    return t, s


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
                [restimes.append(j) for j in [times[res, lip][i]]*\
                                              Ns[res, lip][i].astype(int)]
        alltimes.append(restimes)
    return np.asarray(alltimes)


def get_bins(x, ts):
    if isinstance(x, list):
        x = np.asarray(x)
    elif isinstance(x, np.ndarray):
        pass
    else:
        raise TypeError('Input should be a list or array')
    return np.arange(1, int(x.max()//ts)+3)*ts


def mixture_and_plot(gibbs, method, **kwargs):
    from sklearn import mixture
    from scipy import stats

    clu = getattr(mixture, method)
    keyvalpairs = [f'{key}_{val}' for key,val in zip(kwargs.keys(),
                                                     kwargs.values())]
    kwarg_str = '_'.join(keyvalpairs)

    burnin_ind = gibbs.burnin // gibbs.g
    data_len = len(gibbs.times)
    wcutoff = 10 / data_len

    weights, rates = gibbs.mcweights[burnin_ind:], gibbs.mcrates[burnin_ind:]
    lens = np.array([len(row[row > wcutoff]) for row in weights])
    lmin, lmode, lmax = lens.min(), stats.mode(lens).mode, lens.max()
    train_param = lmode

    train_inds = np.where(lens == train_param)[0]
    train_weights = (weights[train_inds][weights[train_inds]>wcutoff].
                     reshape(-1, train_param))
    train_rates = (rates[train_inds][weights[train_inds]>wcutoff].
                   reshape(-1, train_param))

    inds = np.where(weights > wcutoff)
    aweights, arates = weights[inds], rates[inds]
    rcutoff = arates.min()
    data = np.stack((aweights, arates), axis=1)

    tweights, trates = train_weights.flatten(), train_rates.flatten()
    train_data = np.stack((tweights, trates), axis=1)

    tmpw, tmpr = np.delete(weights, train_inds), np.delete(rates, train_inds)
    pweights, prates = tmpw[tmpw > wcutoff], tmpr[tmpw > wcutoff]
    predict_data = np.stack((pweights, prates), axis=1)

    r = clu(**kwargs)
    labels = r.fit_predict(np.log(train_data))
    uniq_labels = np.unique(labels)
    leg_labels = np.array([f'{i}' for i in uniq_labels])
    predict_labels = r.predict(np.log(predict_data))

    #sorts = r.precisions_.argsort()[::-1]
    sorts = r.means_[:, 0].argsort()[::-1]
    sorts = np.array([np.where(sorts == i)[0][0] for i in uniq_labels])
    # tinds = np.array([np.where(labels == i)[0] for i in uniq_labels],
    #                  dtype=object)
    # pinds = np.array([np.where(predict_labels == i)[0] for i in uniq_labels],
    #                  dtype=object)
    #
    # tindsi = tinds[sorts]
    # pindsi = pinds[sorts]

    # for i in uniq_labels:
    #     labels[tinds[i]] =

    train_data_inds = np.array([np.where(data == col)[0][0] for col in
                                train_data])
    predict_data_inds = np.array([np.where(data == col)[0][0] for col in
                                  predict_data])
    all_labels = r.predict(np.log(data))

    labels = sorts[labels]
    predict_labels = sorts[predict_labels]
    all_labels = sorts[all_labels]

    tinds = [np.where(labels == i)[0] for i in uniq_labels]
    pinds = [np.where(predict_labels == i)[0] for i in uniq_labels]

    cmap = mpl.colormaps['tab10']
    cmap.set_under()
    scale, sparse = 3, 1

    fig, ax = plt.subplots(2, 2, figsize=(4*scale, 3*scale))
    for i in uniq_labels:
        bins = np.exp(np.linspace(np.log(trates[tinds[i]].min()),
                                  np.log(trates[tinds[i]].max()), 50))
        ax[0, 0].hist(prates[pinds[i]], bins=bins, label=leg_labels[i],
                      color=cmap(get_color(i)), zorder=1)
        ax[0, 0].hist(trates[tinds[i]], bins=bins, label=leg_labels[i],
                      color=cmap(get_color(i)), zorder=2, alpha=0.5,
                      edgecolor='k')

    ax[0, 0].set_xscale('log')
    ax[0, 0].set_xlabel(r'rate ($ns^{-1}$)')
    ax[0, 0].set_ylabel('count')
    ax[0, 0].set_xlim(rcutoff, 10)

    row, col = gibbs.mcweights[burnin_ind:].shape
    iter_arr = np.mgrid[:row, :col][0]
    iters = iter_arr[inds]
    titer, piter = iters[train_data_inds], iters[predict_data_inds]
    for i in uniq_labels:
        ax[0, 1].plot(piter[pinds[i]], pweights[pinds[i]][::sparse], '.',
                      label=leg_labels[i], color=cmap(get_color(i)), zorder=1)
        ax[1, 1].plot(piter[pinds[i]], prates[pinds[i]][::sparse], '.',
                      label=leg_labels[i], color=cmap(get_color(i)), zorder=1)
        ax[0, 1].plot(titer[tinds[i]], tweights[tinds[i]][::sparse], '.',
                      label=leg_labels[i], color=cmap(get_color(i)), zorder=2,
                      alpha=0.5, mec='k', mew=1)
        ax[1, 1].plot(titer[tinds[i]], trates[tinds[i]][::sparse], '.',
                      label=leg_labels[i], color=cmap(get_color(i)), zorder=2,
                      alpha=0.5, mec='k', mew=1)

    ax[0, 1].set_yscale('log')
    ax[0, 1].set_ylabel(r'weight')
    ax[1, 1].set_yscale('log')
    ax[1, 1].set_ylabel(r'rate ($ns^{-1}$)')
    ax[1, 1].set_xlabel('sample')
    ax[0, 1].set_xlabel('sample')
    ax[0, 1].set_ylim(wcutoff, 1)
    ax[1, 1].set_xlabel('sample')
    ax[1, 1].set_ylim(rcutoff, 10)

    for i in uniq_labels:
        ax[1, 0].plot(prates[pinds[i]], pweights[pinds[i]], '.',
                      label=leg_labels[i],
                      color=cmap(get_color(i)), zorder=1)
        ax[1, 0].plot(trates[tinds[i]], tweights[tinds[i]], '.',
                      label=leg_labels[i],
                      color=cmap(get_color(i)), zorder=2, alpha=0.5,
                      mec='k', mew=1)

    ax[1, 0].set_yscale('log')
    ax[1, 0].set_xscale('log')
    ax[1, 0].set_ylabel('weight')
    ax[1, 0].set_xlabel(r'rate ($ns^{-1}$)')
    ax[1, 0].set_xlim(rcutoff, 10)
    ax[1, 0].set_ylim(wcutoff, 1)

    handles, plot_labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, plot_labels, loc='lower center',
               ncols=len(plot_labels)/2, title='cluster')
    fig.suptitle(f'{method} '+' '.join(keyvalpairs), fontsize=16)
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    plt.savefig(f"{gibbs.residue}/results_{method}_{kwarg_str}.png",
                bbox_inches='tight')
    plt.show()
    return all_labels

