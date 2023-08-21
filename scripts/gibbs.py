from basicrta import *
from multiprocessing import Pool, Lock
from basicrta import istarmap
import numpy as np
import MDAnalysis as mda
import os 
from tqdm import tqdm

if __name__ == "__main__":
    # Parts of code taken from Shep (Centrifuge3.py, SuperMCMC.py)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--contacts')
    parser.add_argument('--top')
    parser.add_argument('--protname')
    parser.add_argument('--ncore', nargs='?', default=1)
    parser.add_argument('--resids', nargs='?')
    parser.add_argument('--ncomp', nargs='?', default=None)
    parser.add_argument('--niter', nargs='?', default=None)
    args = parser.parse_args()
    a = np.load(args.contacts)

    with open('contacts.metadata', 'r') as data:
        line = data.readlines()[1].split(',')
        trajlen, protlen, liplen, sel, ts = int(line[0]), int(line[1]), int(line[2]), line[3], float(line[4])

    cutoff = float(args.contacts.split('.npy')[0].split('_')[-1])
    nproc, prot = int(args.ncore), args.protname
    if args.ncomp:
        ncomp = int(args.ncomp)
    else: 
        ncomp = None

    if args.niter:
        niter = int(args.niter)
    else: 
        niter = None

    u = mda.Universe(args.top)
    ids = u.select_atoms('protein').residues.resids
    names = u.select_atoms('protein').residues.resnames
    names = np.array([mda.lib.util.convert_aa_code(name) for name in names])
    uniqs = np.unique(a[:, 0]).astype(int)
    resids, resnames = ids[uniqs], names[uniqs]
    residues = np.array([f'{name}{resid}' for name, resid in zip(resnames, resids)])

    if args.resids:
        tmpresids = np.array([res for res in args.resids.strip('[]').split(',')]).astype(int)
        idinds = np.array([np.where(resids == resid)[0][0] for resid in tmpresids])
        residues = residues[idinds] 
        times = [a[a[:, 0] == i][:, 3] for i in uniqs[idinds]]
        trajtimes = [a[a[:, 0] == i][:, 2] for i in uniqs[idinds]]
    else:
        times = [a[a[:, 0] == i][:, 3] for i in uniqs]
        trajtimes = [a[a[:, 0] == i][:, 2] for i in uniqs]

    if not os.path.exists(f'BaSiC-RTA-{cutoff}'):
        os.mkdir(f'BaSiC-RTA-{cutoff}')
    os.chdir(f'BaSiC-RTA-{cutoff}')

    input_list = np.array([[residues[i], times[i], ts, ncomp, niter] for i in range(len(residues))], dtype=object)
    with Pool(nproc, initializer=tqdm.set_lock, initargs=(Lock(),)) as p:
        for _ in tqdm(p.istarmap(run_residue, input_list), total=len(residues), position=0, desc='overall progress'):
            pass

