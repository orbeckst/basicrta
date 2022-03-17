#!/usr/bin/env python

import basicrta
import os
os.environ['MKL_NUM_THREADS'] = '1'
from tqdm import tqdm
import numpy as np
from glob import glob
import multiprocessing
from multiprocessing import Pool, Lock
from basicrta import get_dec

def lipswap(protlen, lip, memarr, ts):
    try:
        proc = int(multiprocessing.current_process().name[-1])
    except ValueError:
        proc = 1
    
    dset = []
    dec = get_dec(ts)
    lipmemarr = memarr[memarr[:, 2] == lip]
    for res in tqdm(range(protlen), desc=f'lipID {lip}', position=proc, leave=False):
        stimes = np.round(lipmemarr[:, -1][lipmemarr[:, 1] == res], dec)
        if len(stimes) == 0:
            continue
        stimes = np.concatenate([np.array([-1]), stimes, np.array([stimes[-1]+1])])
        diff = np.round(stimes[1:]-stimes[:-1], dec)
        #singles = stimes[np.where((diff[1:] > ts) & (diff[:-1] > ts))[0]+1]
        diff[diff > ts] = 0
        inds = np.where(diff == 0)[0]
        sums = [sum(diff[inds[i]:inds[i+1]]) for i in range(len(inds)-1)]
        clens = np.round(np.array(sums), dec)
        minds = np.where(clens != 0)[0]
        clens = clens[minds]+ts
        strt_times = stimes[inds[minds]+1]

        #[dset.append([res, lip, time, ts]) for time in singles]
        [dset.append([res, lip, time, clen]) for time, clen in zip(strt_times, clens)]
    dset = np.array(dset, dtype='float64')
    np.save('lip_{0:0>4}'.format(lip), dset)


def cat_lipids(cutoff, ctype):
    lip_files = glob('lip_0*.npy')
    lip_files.sort()

    tot = []
    for afile in lip_files:
        a = np.load(afile, allow_pickle=True)
        tot.append(a)
    tot = np.array(tot, dtype=object)
    contacts = np.concatenate([*tot])
    [os.system('rm {0}'.format(afile)) for afile in lip_files]
    np.save('{0}_contacts_{1}'.format(ctype, cutoff), contacts)


contact_types = {'lipswap':lipswap}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff")
    parser.add_argument("--ncores")
    args = parser.parse_args()
    cutoff = float(args.cutoff)
    nproc = int(args.ncores)

    with open('contacts.metadata', 'r') as data:
        line = data.readlines()[1].split(',')                              
        trajlen, protlen, liplen, sel, ts = int(line[0]), int(line[1]), int(line[2]), line[3], float(line[4])
    
    if os.path.exists('contacts.mmap'):
        with open('contacts.metadata', 'r') as meta:
            memlen = int(meta.readlines()[-1].split('=')[1])
        print('loading memmap')
        memmap = np.memmap('contacts.mmap', mode='r', shape=(memlen, 5), dtype=np.float64)

        print('applying cutoff')
        Time = np.unique(memmap[:, 4])
        memmap = memmap[memmap[:, -2] <= cutoff]

        params = [tuple([protlen, i, memmap[memmap[:, 2] == i], ts]) for i in range(liplen)]
        pool = Pool(nproc, initializer=tqdm.set_lock, initargs=(Lock(),))
        try:
            result = pool.starmap(lipswap, params)
        except KeyboardInterrupt:
            pool.terminate()
        pool.close()
        cat_lipids(cutoff, 'lipswap')
    else:
        print('contacts.mmap not found')
