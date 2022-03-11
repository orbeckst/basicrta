#!/usr/bin/env python

import os
os.environ['MKL_NUM_THREADS'] = '1'
import MDAnalysis as mda
from MDAnalysis.lib import distances
import numpy as np
from tqdm import tqdm
import collections
import time
from pmdautil import make_balanced_slices
from multiprocessing import Pool, Lock
from glob import glob
from pympler.tracker import SummaryTracker

def run_contacts(top,traj,i,aslice,sel,bsn,nproc,Nbs):
    u = mda.Universe(top,traj)
    uprot = u.select_atoms('protein')
    prot = u.select_atoms('protein').residues
    ulip = u.select_atoms(sel)
    
    xp = np.arange(1,len(uprot.residues)+1)
    uprot.residues.resnums = xp 
    xlip = np.arange(1,len(ulip.residues)+1)
    ulip.residues.resnums = xlip
    
    protresnums = uprot.resnums
    lipresnums = ulip.resnums
    prots = np.unique(uprot.resnums)
    ulip = u.select_atoms(sel + ' and around {0} protein'.format(cutoff),updating=True)

    dset = []
    text = 'big-slice {0}/{1} proc {2} {3}'.format(bsn+1,Nbs,i,aslice)
    entry = 0
    for k,ts in enumerate(tqdm(u.trajectory[aslice], desc=text, position=i, total=len(u.trajectory[aslice]),leave=False)):
        b = distances.capped_distance(uprot.positions,ulip.positions,max_cutoff=cutoff)
        lipress = ulip.resnums
        pairlist = [(protresnums[b[0][i,0]],lipress[b[0][i,1]]) for i in range(len(b[0]))]
        pairdir = collections.Counter(a for a in pairlist)
        lsum = 0
        for j in pairdir:
            temp = pairdir[j]
            dset.append([u.trajectory.frame,j[0]-1,j[1]-1,min(b[1][lsum:lsum+temp]),u.trajectory.time/1000])#convert to ns
            lsum += pairdir[j]
    dset = np.asarray(dset)
    np.save('contacts_{0:0>4}.npy'.format(bsn*nproc+i),dset)

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
        meta.write('trajlen,protlen,liplen,sel \n')    
        meta.write('{0},{1},{2},{3},{4}'.format(trajlen,protlen,liplen,args.sel,ts))

    tracker = SummaryTracker()

    Nbs = len(uu.trajectory[trajslice])//50000
    big_slices = make_balanced_slices(len(uu.trajectory[trajslice]), Nbs, start=b, stop=z, step=s) 
    
    start=time.time()
    for bsn,big_slice in enumerate(big_slices):
        slices = make_balanced_slices(len(uu.trajectory[big_slice]), nproc, start=big_slice.start, stop=big_slice.stop, step=big_slice.step)
        slices[-1] = slice(slices[-1].start,big_slice.stop,1)
        inputlist = np.array([[top, traj, i, slices[i], args.sel, bsn, nproc, Nbs] for i in range(nproc)])
        Pool(nproc, initializer=tqdm.set_lock, initargs=(Lock(),)).starmap(run_contacts, inputlist)
    stop = time.time()
    
    tracker.print_diff()
#    with open('memtracker.txt', 'w') as F:
#        F.write(tracker.print_diff())

    with open('analysis_time1.txt','w+') as w:
        w.write('prot_analysis time {0}'.format((stop-start)/3600))
    mmap = make_memmap()
    Times = np.unique(mmap[:,4])
    np.save('Time_arr',Times)
    print('elapsed time {0} hr'.format((stop-start)/3600))

    
