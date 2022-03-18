from basicrta import *
import multiprocessing
from multiprocessing import Pool, Lock
from basicrta import istarmap
import numpy as np
import MDAnalysis as mda
import os 
from tqdm import tqdm

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--contacts')
    parser.add_argument('--top')
    parser.add_argument('--traj')
    parser.add_argument('--ncore')
    args = parser.parse_args()
    a = np.load(args.contacts)

    with open('contacts.metadata', 'r') as data:
        line = data.readlines()[1].split(',')
        trajlen, protlen, liplen, sel, ts = int(line[0]), int(line[1]), int(line[2]), line[3], float(line[4])

    nproc = int(args.ncore)
    u = mda.Universe(os.path.abspath(args.top), os.path.abspath(args.traj))
    ids = u.select_atoms('protein').residues.resids
    names = u.select_atoms('protein').residues.resnames
    names = np.array([mda.lib.util.convert_aa_code(name) for name in names])
    uniqs = np.unique(a[:, 0]).astype(int)
    resids, resnames = ids[uniqs], names[uniqs]
    residues = np.array([f'{name}{resid}' for name, resid in zip(resnames, resids)])
    times = np.array([a[a[:, 0] == i][:, 3] for i in uniqs], dtype=object)
    trajtimes = np.array([a[a[:, 0] == i][:, 2] for i in uniqs], dtype=object)
    lipinds = np.array([a[a[:, 0] == i][:, 1] for i in uniqs], dtype=object)

    os.chdir('BaSiC-RTA')

    rem_inds = get_remaining_residue_inds(residues)
    times, trajtimes, lipinds = times[rem_inds], trajtimes[rem_inds], lipinds[rem_inds]
    residues, t_slow, sd, indicators = collect_results()

    input_list = np.array([[u, times[i], trajtimes[i], indicators[i], residues[i], lipinds[i]] for i in range(len(residues))],
                          dtype=object)
    try:
        with Pool(nproc, initializer=tqdm.set_lock, initargs=(Lock(),)) as p:
            for _ in tqdm(p.istarmap(write_trajs, input_list), total=len(input_list), position=0, desc='overall progress'):
                pass
    except:
        print('error')
    pool.close()
