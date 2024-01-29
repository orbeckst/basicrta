#!/usr/bin/env python

import os
os.environ['MKL_NUM_THREADS'] = '1'
import MDAnalysis as mda
from MDAnalysis.lib import distances
import numpy as np
from tqdm import tqdm
import collections
from pmdautil import make_balanced_slices
from multiprocessing import Pool, Lock
from glob import glob


class MapContacts(object):
    """
    This class is used to create the map of contacts between two groups of
    atoms. A single cutoff is used to define a contact between the two groups,
    where if any atomic distance between the two groups is less than the cutoff,
    a contact is considered formed.
    """

    def __init__(self, cutoff, ag1, ag2):
        self.cutoff = cutoff


def run_contacts(top, traj, i, aslice, sel, bsn, nproc, Nbs):
    u = mda.Universe(top, traj)
    prot = u.select_atoms('protein')
    lip = u.select_atoms(sel)
    
    prot.residues.resids = np.arange(len(prot.residues))
    lip.residues.resids = np.arange(len(lip.residues))
    
    ulip = u.select_atoms(sel + ' and around {0} protein'.format(cutoff), updating=True)

    dset = []
    text = 'big-slice {0}/{1} proc {2} {3}'.format(bsn+1, Nbs, i, aslice)
    for ts in tqdm(u.trajectory[aslice], desc=text, position=i, total=len(u.trajectory[aslice]), leave=False):
        b = distances.capped_distance(prot.positions, ulip.positions, max_cutoff=cutoff)
        pairlist = [(prot.resids[b[0][i, 0]], ulip.resids[b[0][i, 1]]) for i in range(len(b[0]))]
        pairdir = collections.Counter(a for a in pairlist)
        lsum = 0
        for j in pairdir:
            temp = pairdir[j]
            dset.append([u.trajectory.frame, j[0], j[1], min(b[1][lsum:lsum+temp]), ts.time/1000])#convert to ns
            lsum += temp
    dset = np.asarray(dset)
    np.save('contacts_{0:0>4}.npy'.format(bsn*nproc+i), dset)

def make_memmap():
    files = glob('contacts_*.npy')
    files.sort()
    shapes = []
    for afile in tqdm(files, desc='Getting shape'):
        aa = np.load(afile)
        aashape = aa.shape
        shapes.append(aashape)
    shapes = np.asarray(shapes)
    memlen = sum(shapes[:,0])
    with open('contacts.metadata','a') as meta:
        meta.write('\n memlen={0}'.format(memlen))
    memmap = np.memmap('contacts.mmap', shape=(memlen,5), dtype=np.float64, mode='w+')

    it = 0
    for afile in tqdm(files, desc='Creating memmap'):
        aa = np.load(afile)
        alen = aa.shape[0]
        memmap[it:it+alen] = aa
        it+=alen
    [os.system('rm {0}'.format(afile)) for afile in files]
    return memmap

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--traj')
    parser.add_argument('--top')
    parser.add_argument('--sel',help='lipid selection')
    parser.add_argument('--ncores')
    parser.add_argument('-b',help='frame',nargs='?')
    parser.add_argument('-z',help='frame',nargs='?')
    parser.add_argument('-s',help='frame',nargs='?')
    args = parser.parse_args()

    top =args.top
    traj = args.traj
    nproc = int(args.ncores)
    cutoff=10.0
    
    if not args.b:
        b=0
    else:
        b=int(args.b)

    if not args.z:
        z=None
    else:
        z=int(args.z)

    if not args.s:
        s=1
    else:
        s=int(args.s)

    trajslice = slice(b,z,s)
    
    uu = mda.Universe(top,traj)
    ts = uu.trajectory.ts.dt/1000 #nanoseconds
    trajlen = len(uu.trajectory[trajslice])
    protlen = len(uu.select_atoms('protein').residues)
    liplen = len(uu.select_atoms(args.sel).residues)

    with open('contacts.metadata','w') as meta:              
        meta.write('trajlen,protlen,liplen,sel,ts \n')    
        meta.write('{0},{1},{2},{3},{4}'.format(trajlen,protlen,liplen,args.sel,ts))

    Nbs = len(uu.trajectory[trajslice])//100000
    if Nbs>0:
        big_slices = make_balanced_slices(len(uu.trajectory[trajslice]), Nbs, start=b, stop=z, step=s) 
    else:
        big_slices = [trajslice]
    
    for bsn,big_slice in enumerate(big_slices):
        slices = make_balanced_slices(len(uu.trajectory[big_slice]), nproc, start=big_slice.start, stop=big_slice.stop, step=big_slice.step)
        slices[-1] = slice(slices[-1].start,big_slice.stop,1)
        inputlist = np.array([[top, traj, i, slices[i], args.sel, bsn, nproc, Nbs] for i in range(nproc)])
        Pool(nproc, initializer=tqdm.set_lock, initargs=(Lock(),)).starmap(run_contacts, inputlist)

#    slices = make_balanced_slices(len(uu.trajectory), nproc)
#    inputlist = np.array([[top, traj, i, slices[i], args.sel, bsn, nproc, Nbs] for i in range(nproc)])
#    Pool(nproc, initializer=tqdm.set_lock, initargs=(Lock(),)).starmap(run_contacts, inputlist)

    mmap = make_memmap()
    
