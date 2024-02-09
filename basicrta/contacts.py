#!/usr/bin/env python
import os
os.environ['MKL_NUM_THREADS'] = '1'
from tqdm import tqdm
import numpy as np
import multiprocessing
from MDAnalysis.lib import distances
import collections
from multiprocessing import Pool, Lock
import MDAnalysis as mda
import pickle

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
        self.cutoff, self.frames, self.nslices = cutoff, frames, 500


    def run(self):
        if self.frames:
            sliced_frames = np.array_split(self.frames, self.nslices)
            # sliced_frames = np.array_split(self.frames, self.nproc)
        else:
            sliced_frames = np.array_split(np.arange(len(self.u.trajectory)),
                                           self.nslices)
            # sliced_frames = np.array_split(np.arange(len(self.u.trajectory)),
            #                                            self.nproc)

        input_list = [[i, self.u.trajectory[aslice]] for
                      i, aslice in enumerate(sliced_frames)]

        lens = (Pool(self.nproc, initializer=tqdm.set_lock, initargs=(Lock(),)).
                starmap_async(self._run_contacts, input_list))

        bounds = np.concatenate([[0], np.cumsum(lens)])
        mapsize = sum(lens)
        contact_map = np.memmap('contacts.pkl', mode='w+', shape=(mapsize, 5),
                                dtype=np.float64)
        for i in range(self.nproc):
            filename = f'.contacts_{i:04}'
            dset = []
            with open(filename, 'r') as f:
                for line in f:
                    dset.append([float(i) for i in line.strip('[]\n').
                                split(',')])
            contact_map[bounds[i]:bounds[i+1]] = dset

        # mmaps = []
        # for proc in range(self.nproc):
        #     filename = f'.contacts_{proc:03}.npy'
        #     mmaps.append(np.load(filename, mmap_mode='r'))

        dtype = np.dtype(np.float64,
                         metadata={'top': self.u.filename,
                                   'traj': self.u.trajectory.filename,
                                   'ag1': ag1, 'ag2': ag2})

        # outarr = np.concatenate([*mmaps], dtype=dtype)
        contact_map.dtype = dtype
        contact_map.dump('contacts.pkl')
        # with open('contacts.pkl', 'w+b') as f:
        #     pickle.dump(outarr, f)

        print('\nSaved contacts as "contacts.npy"')


    def _run_contacts(self, i, sliced_traj):
        from basicrta.util import get_dec

        try:
            proc = int(multiprocessing.current_process().name.split('-')[-1])
        except ValueError:
            proc = 1

        with open(f'.contacts_{i:04}', 'w+') as f:
            dec = get_dec(self.u.trajectory.ts.dt/1000)  # convert to ns
            text = f'slice {i+1} of {self.nslices}'
            data_len = 0
            for ts in tqdm(sliced_traj, desc=text, position=proc,
                           total=len(sliced_traj), leave=False):
                dset = []
                b = distances.capped_distance(self.ag1.positions,
                                              self.ag2.positions,
                                              max_cutoff=self.cutoff)
                pairlist = [(self.ag1.resids[b[0][i, 0]],
                             self.ag2.resids[b[0][i, 1]]) for i in
                            range(len(b[0]))]
                pairdir = collections.Counter(a for a in pairlist)
                lsum = 0
                for j in pairdir:
                    temp = pairdir[j]
                    dset.append([ts.frame, j[0], j[1],
                                 min(b[1][lsum:lsum+temp]),
                                 np.round(ts.time, dec)/1000])  # convert to ns
                    lsum += temp
                [f.write(f"{line}\n") for line in dset]
                data_len += len(dset)
            f.flush()
        return data_len

    # def _run_contacts(self, i, sliced_traj):
    #     from basicrta.util import get_dec
    #
    #     data_len = 0
    #     oldmap = np.memmap(f'.tmpmap', mode='w+', shape=(data_len + 1, 5),
    #                        dtype=np.float64)
    #     del oldmap
    #
    #     dec = get_dec(self.u.trajectory.ts.dt/1000)  # convert to ns
    #     text = f'process {i+1} of {self.nproc}'
    #     for ts in tqdm(sliced_traj, desc=text, position=i,
    #                    total=len(sliced_traj), leave=False):
    #         oldmap = np.memmap(f'.tmpmap', mode='r', shape=(data_len+1, 5),
    #                            dtype=np.float64)
    #         dset = []
    #         b = distances.capped_distance(self.ag1.positions,
    #                                       self.ag2.positions,
    #                                       max_cutoff=self.cutoff)
    #         pairlist = [(self.ag1.resids[b[0][i, 0]],
    #                      self.ag2.resids[b[0][i, 1]]) for i in
    #                     range(len(b[0]))]
    #         pairdir = collections.Counter(a for a in pairlist)
    #         lsum = 0
    #         for j in pairdir:
    #             temp = pairdir[j]
    #             dset.append([ts.frame, j[0], j[1],
    #                          min(b[1][lsum:lsum+temp]),
    #                          np.round(ts.time, dec)/1000])  # convert to ns
    #             lsum += temp
    #         new_len = data_len + len(dset) + 1
    #         newmap = np.memmap(f'.contacts_{i:03}', mode='w+',
    #                            shape=(new_len, 5), dtype=np.float64)
    #         newmap[:data_len] = oldmap[:data_len]
    #         newmap[data_len:new_len] = dset
    #         del oldmap
    #         oldmap = np.memmap(f'.tmpmap', mode='w+',
    #                            shape=(new_len, 5), dtype=np.float64)
    #         oldmap[:] = newmap[:]
    #         data_len += new_len
    #     # map.dump()
    #     return map

class ProcessContacts(object):
    def __init__(self, cutoff, nproc, map_name='contacts.pkl'):
        self.cutoff, self.nproc = cutoff, nproc
        self.map_name = map_name


    def run(self):
        from basicrta.util import siground

        if os.path.exists(self.map_name):
            with open(self.map_name, 'r+b') as f:
                memmap = pickle.load(f)
            # memmap = np.load(self.map_name, mmap_mode='r')
            memmap = memmap[memmap[:, -2] <= self.cutoff]
            dtype = memmap.dtype
        else:
            raise FileNotFoundError(f'{self.map_name} not found. Specify the '
                                    'contacts file using the "map_name" '
                                    'argument')

        self.ts = siground(np.unique(memmap[1:, 4]-memmap[:-1, 4])[1], 1)
        lresids = np.unique(memmap[:, 2])
        params = [[res, memmap[memmap[:, 2] == res]] for res in lresids]
        pool = Pool(self.nproc, initializer=tqdm.set_lock, initargs=(Lock(),))

        try:
            dsets = pool.starmap(self._lipswap, params)
        except KeyboardInterrupt:
            pool.terminate()
        pool.close()

        outarr = np.concatenate([*dsets], dtype=dtype)
        with open(f'contacts_{cutoff}.pkl', 'w+b') as f:
            pickle.dump(outarr, f)

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
        for pres in tqdm(presids, desc=f'lipID {int(lip)}', position=proc,
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


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--top', type=str)
    parser.add_argument('--traj', type=str, nargs='+')
    parser.add_argument('--sel1', type=str)
    parser.add_argument('--sel2', type=str)
    parser.add_argument('--cutoff', type=float)
    parser.add_argument('--nproc', type=int, default=1)
    args = parser.parse_args()

    u = mda.Universe(args.top)
    [u.load_new(traj) for traj in args.traj]
    cutoff, nproc = args.cutoff, args.nproc
    ag1 = u.select_atoms(args.sel1)
    ag2 = u.select_atoms(args.sel2)

    MapContacts(u, ag1, ag2, nproc=nproc).run()
    ProcessContacts(cutoff, nproc).run()