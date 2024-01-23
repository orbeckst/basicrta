from multiprocessing import shared_memory
from basicrta import *
from basicrta.functions import simulate_hn
from multiprocessing import Pool, Lock
from basicrta import istarmap
import numpy as np
import MDAnalysis as mda
import os 
from tqdm import tqdm
import gc

if __name__ == "__main__":
    # Parts of code taken from Shep (Centrifuge3.py, SuperMCMC.py)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--niter', nargs='?', default=10000)
    parser.add_argument('-N', nargs='?', default=100000)
    parser.add_argument('--ncomp', nargs='?', default=10, type=int)
    args = parser.parse_args()

    ts, ncomp, N = 0.1, args.ncomp, args.N
    if args.niter:
        niter = int(args.niter)

    residue, ts, nproc = 'X1', 0.1, 1
    times = simulate_hn(N, [0.901, 0.09, 0.009], [5, 0.1, 0.001])

    if not os.path.exists(f'X1'):
        os.mkdir(f'X1')
    os.chdir(f'X1')

    input_list = np.array([residue, times, ts, ncomp, niter], dtype=object)
    with Pool(nproc, initializer=tqdm.set_lock, initargs=(Lock(),)) as p:
        for _ in tqdm(p.istarmap(run_residue, input_list), total=len(residue), position=0, desc='overall progress'):
            pass
