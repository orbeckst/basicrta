#!/usr/bin/env python

from __future__ import division
import os
import sys
os.environ['MKL_NUM_THREADS'] = '1' 
from pmdautil import make_balanced_slices
from tqdm import tqdm
import numpy as np
import pandas as pd
from glob import glob
from multiprocessing import Pool, Lock
import time
from collections import Counter
#from pympler.tracker import SummaryTracker

def lipswap(protlen,liplen,cutoff,i,aslice,memarr,ts,Time):
    dset = []
    for lip in range(aslice.start,aslice.stop,1):
        traj_times = []
        lipmemarr = memarr[memarr[:,2]==lip].astype(int) 
        for res in tqdm(range(protlen),desc='lipids {0}-{1}'.format(aslice.start,aslice.stop),position=i,leave=False):
            frame = lipmemarr[:,0][lipmemarr[:,1]==res].astype(int)
            try:
                tmparr = np.zeros(frame[-1]+3)
            except IndexError:
                continue
            tmparr[frame+1]=1
            maxs = np.where(tmparr==0)[0]-1
            mins = np.where(tmparr==0)[0]+1

            maxs = maxs[maxs>=0]
            mins = mins[mins<len(tmparr)]

            tmax = Time[maxs[np.where(tmparr[maxs]!=0)[0]]-1]
            tmin = Time[mins[np.where(tmparr[mins]!=0)[0]]-1]  

            times = (tmax-tmin)+ts
            TC = Counter(times)
            [dset.append([res,lip,time,TC[time]]) for time in TC.keys()]
            traj_times.append([times, tmin])
        dset = np.asarray(dset)
        traj_times = np.array(traj_times, dtype=object)
        np.save('lip_{0:0>4}'.format(lip),dset)
        np.save('traj_times_lip_{0:0>4}'.format(lip), traj_times)

def anylip(protlen,liplen,cutoff,i,aslice,ts):
    dset = []
    lip=-1
    for res in tqdm(range(aslice.start,aslice.stop),desc='residues {0}-{1}'.format(aslice.start,aslice.stop),position=i,leave=False):
        tmp = memarr[:,0][memarr[:,1]==res].astype(int) 
        try:
            tmparr = np.zeros(tmp[-1]+1)
        except IndexError:
            continue
        tmparr[tmp]=1
        s = pd.Series(tmparr)
        sums = np.asarray(s.groupby(s.eq(0).cumsum()).cumsum().tolist())
        times = sums[np.where(sums==0)[0]-1]*ts
        times = times[times!=0]
        TC = Counter(times)
        [dset.append([res,lip,time,TC[time]]) for time in TC.keys()]
    dset = np.asarray(dset)
    np.save('lip_{0:0>4}'.format(i),dset)

def cat_lipids(cutoff,ctype):
    lip_files = glob('lip_0*.npy')
    lip_files.sort()
    time_files = glob('traj_times_lip_0*')
    time_files.sort()

    shapes = []
    ress = []
    for afile in lip_files:
        a = np.load(afile,allow_pickle=True)
        shapes.append(a.shape[0])
        #ress.append(np.unique(a[:,0]).astype(int))
        ress.append(set(np.unique(a[:,0]).astype(int)))
    leng = sum(shapes)
    print(ress)
    ress = np.array(list(set.union(*ress)))
    #ress = np.unique(np.concatenate([*ress]))

    indmap = {}
    for i,res in enumerate(ress):
        indmap[res] = i

    res_lip = [[] for res in ress]
    for tfile,lfile in zip(time_files, lip_files):
        t = np.load(tfile, allow_pickle=True)
        l = np.load(lfile, allow_pickle=True)
        resinds = np.unique(l[:,0]).astype(int)

        tmpmap = {}
        for i,res in enumerate(resinds):
            tmpmap[res] = i

        for ind in resinds:
            res_lip[indmap[ind]].append(t[tmpmap[ind]])
    res_lip = np.array(res_lip, dtype=object)
    
    Ctimes = []                                  
    Ttimes = []                                  
    for i,res in tqdm(enumerate(res_lip)):
        ctimes, ttimes = [], []                  
        for lip in res:
            ctimes.append(lip[0])                
            ttimes.append(lip[1])                
        try:                                     
            Ctimes.append(np.concatenate(ctimes))
            Ttimes.append(np.concatenate(ttimes))
        except ValueError:                       
            continue

    np.save(f'{ctype}_trajtimes_{cutoff}', np.array([Ctimes,Ttimes], dtype=object))

    contacts = np.empty((leng,4))
    it = 0
    for i,afile in enumerate(lip_files):
        a = np.load(afile,allow_pickle=True)
        contacts[it:it+shapes[i],:] = a[:]
        it+=shapes[i]
    [os.system('rm {0}'.format(afile)) for afile in lip_files]
    [os.system('rm {0}'.format(afile)) for afile in time_files]
    np.save('{0}_contacts_{1}'.format(ctype,cutoff),contacts)



contact_types = {'anylip':anylip,
                 'lipswap':lipswap}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff")
    parser.add_argument("--ncores")
    parser.add_argument("--contact_type")
    args = parser.parse_args()
    cutoff = float(args.cutoff)
    nproc = int(args.ncores)

#    os.chdir('..')
    with open('contacts.metadata','r') as data:
        line = data.readlines()[1].split(',')                              
        trajlen,protlen,liplen,sel,ts = int(line[0]),int(line[1]),int(line[2]),line[3],float(line[4])

    #tracker = SummaryTracker()
    start = time.time()
    if os.path.exists('contacts.mmap'):
        with open('contacts.metadata','r') as meta:
            memlen = int(meta.readlines()[-1].split('=')[1])
        print('loading memmap')
        memmap = np.memmap('contacts.mmap', mode='r', shape=(memlen,5), dtype=np.float64)
    else:
        print('contacts.mmap not found')
    print('applying cutoff')
    Time = np.unique(memmap[:,4])
    memmap = memmap[memmap[:,-2]<=cutoff]
    #os.chdir('script_test')
   
    if args.contact_type=='anylip':
        resslices = make_balanced_slices(n_frames=protlen, n_blocks=nproc, start=0, 
                                      stop=protlen, step=1)
        memarr = memmap
        params = [tuple([protlen,liplen,cutoff,i,resslices[i],ts]) for i in range(nproc)]
        pool = Pool(nproc,initializer=tqdm.set_lock, initargs=(Lock(),))
        try:
            pool.starmap(contact_types[args.contact_type],params)
        except KeyboardInterrupt:
            pool.terminate()
    else:
        chunks = liplen//nproc
        if liplen%nproc!=0:
            chunks+=1
        for i in range(chunks):
            lipslices = make_balanced_slices(n_frames=nproc, n_blocks=nproc, start=i*nproc, 
                                          stop=(i+1)*nproc, step=1)
            lipslices = [aslice for aslice in lipslices if aslice.stop<=liplen] 
            nproc = len(lipslices)
            memarr = memmap[(memmap[:,2]>=lipslices[0].start) & (memmap[:,2]<lipslices[-1].stop)]
            params = [tuple([protlen,liplen,cutoff,i,lipslices[i],memarr,ts,Time]) for i in range(nproc)]
            pool = Pool(nproc,initializer=tqdm.set_lock, initargs=(Lock(),))
            try:
                pool.starmap(contact_types[args.contact_type],params)
            except KeyboardInterrupt:
                pool.terminate()
    stop = time.time()
    print('Runtime {0}'.format(stop-start))
    with open('time.txt','w') as t:
        t.write('runtime {0} hr'.format((stop-start)/3600))
        
    cat_lipids(cutoff,args.contact_type)
    #tracker.print_diff()
