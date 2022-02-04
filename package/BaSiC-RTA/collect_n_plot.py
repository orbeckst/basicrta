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


def collect_results():
    dirs = np.array(glob('?[0-9]*'))
    sorted_inds = np.array([int(adir[1:]) for adir in dirs]).argsort()
    dirs = dirs[sorted_inds]
    t_slow = np.zeros(len(dirs))
    sd = np.zeros(len(dirs))
    residues = np.empty((len(dirs)), dtype=object)
    for i, adir in enumerate(tqdm(dirs, desc='Collecting results')):
        residues[i] = adir
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
    return residues, t_slow, sd


def plot_protein(residues, t_slow, sd):
    if not os.path.exists('figs'):
        os.mkdir('figs')

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
    plt.savefig('figs/t_slow.png')
    plt.savefig('figs/t_slow.pdf')
    #plt.show()

if __name__ == "__main__":
    os.chdir('BaSiC-RTA')
    residues, t_slow, sd = collect_results()
    plot_protein(residues, t_slow, sd)
