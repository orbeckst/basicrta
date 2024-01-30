#!/usr/bin/env python

import os
os.environ['MKL_NUM_THREADS'] = '1'
from MDAnalysis.lib import distances
import numpy as np
from tqdm import tqdm
import collections
from multiprocessing import Pool, Lock


class MapContacts(object):
    """
    This class is used to create the map of contacts between two groups of
    atoms. A single cutoff is used to define a contact between the two groups,
    where if any atomic distance between the two groups is less than the cutoff,
    a contact is considered formed.
    """

    def __init__(self, u, ag1, ag2, nproc=1, frames=None, cutoff=10.0):
        self.u, self.nproc = u, nproc
        self.ag1, self.ag2 = ag1, ag2
        self.cutoff, self.frames = cutoff, frames


    def run(self):
        if self.frames:
            sliced_frames = np.array_split(self.frames, self.nproc)
        else:
            sliced_frames = np.array_split(np.arange(len(self.u.trajectory)),
                                           self.nproc)

        input_list = [[i%self.nproc, self.u.trajectory[aslice]] for
                       i, aslice in enumerate(sliced_frames)]
        dsets = Pool(self.nproc, initializer=tqdm.set_lock, initargs=(Lock(),))\
                .starmap(self._run_contacts, input_list)

        np.save('contacts', np.concatenate([*dsets]))
        print('\nSaved contacts as "contacts.npy"')

    def _run_contacts(self, i, sliced_traj):
        from basicrta.util import get_dec

        dset, dec = [], get_dec(self.u.trajectory.ts.dt/1000) #convert to ns
        text = f'process {i+1} of {self.nproc}'
        for ts in tqdm(sliced_traj, desc=text, position=i,
                       total=len(sliced_traj), leave=False):
            b = distances.capped_distance(self.ag1.positions,
                                          self.ag2.positions,
                                          max_cutoff=self.cutoff)
            pairlist = [(self.ag1.resids[b[0][i, 0]],
                         self.ag2.resids[b[0][i, 1]]) for i in range(len(b[0]))]
            pairdir = collections.Counter(a for a in pairlist)
            lsum = 0
            for j in pairdir:
                temp = pairdir[j]
                dset.append([ts.frame, j[0], j[1], min(b[1][lsum:lsum+temp]),
                             np.round(ts.time, dec)/1000]) # convert to ns
                lsum += temp
        return dset
