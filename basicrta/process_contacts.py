#!/usr/bin/env python

import os
os.environ['MKL_NUM_THREADS'] = '1'
from tqdm import tqdm
import numpy as np
import multiprocessing
from multiprocessing import Pool, Lock


class ProcessContacts(object):
    def __init__(self, cutoff, nproc, map_name='contacts.npy'):
        self.cutoff, self.nproc = cutoff, nproc
        self.map_name = map_name


    def run(self):
        if os.path.exists(self.map_name):
            memmap = np.load(self.map_name, mmap_mode='r')
            memmap = memmap[memmap[:, -2] <= self.cutoff]
        else:
            raise FileNotFoundError(f'{self.map_name} not found. Specify the '
                                    'contacts file using the "map_name" '
                                    'argument')

        self.ts = np.unique(memmap[1:, 4]-memmap[:-1, 4])[1]
        lresids = np.unique(memmap[:, 2])
        params = [[res, memmap[memmap[:, 2] == res]] for res in lresids]
        pool = Pool(self.nproc, initializer=tqdm.set_lock, initargs=(Lock(),))

        try:
            dsets = pool.starmap(self._lipswap, params)
        except KeyboardInterrupt:
            pool.terminate()
        pool.close()
        print(self.ts)
        np.save(f'contacts_{self.cutoff}', np.concatenate([*dsets]))
        print(f'\nSaved contacts to "contacts_{self.cutoff}.npy"')


    def _lipswap(self, lip, memarr):
        from basicrta.util import get_dec

        try:
            proc = int(multiprocessing.current_process().name[-1])
        except ValueError:
            proc = 1

        presids = np.unique(memarr[:, 1])
        dset = []
        dec, ts = get_dec(self.ts), self.ts
        for pres in tqdm(presids, desc=f'lipID {lip}', position=proc,
                        leave=False):
            stimes = np.round(memarr[:, -1][memarr[:, 1] == pres], dec)
            if len(stimes) == 0:
                continue
            stimes = np.concatenate([np.array([-1]), stimes,
                                     np.array([stimes[-1]+1])])
            diff = np.round(stimes[1:]-stimes[:-1], dec)
            singles = stimes[np.where((diff[1:] > ts) & (diff[:-1] > ts))[0]+1]
            diff[diff > ts] = 0
            inds = np.where(diff == 0)[0]
            sums = [sum(diff[inds[i]:inds[i+1]]) for i in range(len(inds)-1)]
            clens = np.round(np.array(sums), dec)
            minds = np.where(clens != 0)[0]
            clens = clens[minds]+ts
            strt_times = stimes[inds[minds]+1]

            [dset.append([pres, lip, time, ts]) for time in singles]
            [dset.append([pres, lip, time, clen]) for time, clen in
             zip(strt_times, clens)]
        return dset
