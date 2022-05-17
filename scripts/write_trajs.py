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
    parser.add_argument('--ncore', nargs='?', default=1)
    parser.add_argument('--step', nargs='?')
    parser.add_argument('--resid', nargs='?')
    parser.add_argument('--ncomp', nargs='?', type=int)
    args = parser.parse_args()
    a = np.load(args.contacts)

    if not args.step:
        step = 1
    else: step = int(args.step)

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
    tmpresidues = np.array([f'{name}{resid}' for name, resid in zip(resnames, resids)])
    times = np.array([a[a[:, 0] == i][:, 3] for i in uniqs], dtype=object)
    trajtimes = np.array([a[a[:, 0] == i][:, 2] for i in uniqs], dtype=object)
    lipinds = np.array([a[a[:, 0] == i][:, 1] for i in uniqs], dtype=object)

    os.chdir('BaSiC-RTA')

    #rem_inds = get_remaining_residue_inds(residues)
    #times, trajtimes, lipinds = times[rem_inds], trajtimes[rem_inds], lipinds[rem_inds]
    residues, t_slow, sd, indicators = collect_results(args.ncomp)
    rem_inds = np.array([np.where(tmpresidues==residue)[0][0] for residue in residues])
    times, trajtimes, lipinds = times[rem_inds], trajtimes[rem_inds], lipinds[rem_inds]

    if args.resid:
        res_list = args.resid.strip('[]').split(',')
        res_list = np.array(res_list, dtype=int)
        resids = np.array([int(res[1:]) for res in residues])
        inds = np.array([np.where(resids==res)[0][0] for res in res_list])
        times, trajtimes, lipinds = times[inds], trajtimes[inds], lipinds[inds]
        residues, t_slow, sd = residues[inds], t_slow[inds], sd[inds]
        indicators = [indicators[ind] for ind in inds]

        input_list = np.array(
            [[u, times[i], trajtimes[i], indicators[i], residues[i], lipinds[i], step] for i in range(len(res_list))],
            dtype=object)
    # resids = np.array([int(res[1:]) for res in residues])
    # mat_inds = np.array([np.where(ids==resid)[0][0] for resid in resids])
    else:
        print(len(times), len(trajtimes), len(indicators), len(residues), len(lipinds))
        input_list = np.array([[u, times[i], trajtimes[i], indicators[i], residues[i], lipinds[i], step] for i in range(len(residues))],
                              dtype=object)
        with Pool(nproc, initializer=tqdm.set_lock, initargs=(Lock(),)) as p:
            for _ in tqdm(p.istarmap(write_trajs, input_list), total=len(input_list), position=0, desc='overall progress'):
                pass
