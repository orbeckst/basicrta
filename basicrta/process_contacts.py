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


class ProcessContacts(object):
    def __init__(self, cutoff, nproc, map_name='contacts.npy'):
        self.cutoff, self.nproc = cutoff, nproc
        self.map_name = map_name


    def run(self):
        if os.path.exists(self.map_name):
            memmap = np.load(self.map_name, mmap_mode='r')
            memmap = memmap[memmap[:, -2] <= cutoff]
        else:
            raise FileNotFoundError(f'{self.map_name} not found. Specify the '
                                    'comtacts file using the "map_name" '
                                    'argument')
        Time = np.unique(memmap[:, 4])

    def lipswap(protlen, lip, memarr, ts):
        try:
            proc = int(multiprocessing.current_process().name[-1])
        except ValueError:
            proc = 1

        dset = []
        dec = get_dec(ts)
        for res in tqdm(range(protlen), desc=f'lipID {lip}', position=proc, leave=False):
            stimes = np.round(memarr[:, -1][memarr[:, 1] == res], dec)
            if len(stimes) == 0:
                continue
            stimes = np.concatenate([np.array([-1]), stimes, np.array([stimes[-1]+1])])
            diff = np.round(stimes[1:]-stimes[:-1], dec)
            singles = stimes[np.where((diff[1:] > ts) & (diff[:-1] > ts))[0]+1]
            diff[diff > ts] = 0
            inds = np.where(diff == 0)[0]
            sums = [sum(diff[inds[i]:inds[i+1]]) for i in range(len(inds)-1)]
            clens = np.round(np.array(sums), dec)
            minds = np.where(clens != 0)[0]
            clens = clens[minds]+ts
            strt_times = stimes[inds[minds]+1]

            [dset.append([res, lip, time, ts]) for time in singles]
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
