from basicrta.wdensity import WDensityAnalysis
from basicrta import *
import numpy as np
import MDAnalysis as mda
import os


if __name__ == "__main__":
    a = np.load('lipswap_contacts_7.0.npy')
    with open('contacts.metadata', 'r') as data:
        line = data.readlines()[1].split(',')
        trajlen, protlen, liplen, sel, ts = int(line[0]), int(line[1]), int(line[2]), line[3], float(line[4])

    u = mda.Universe(os.path.abspath('step7_fixed.pdb'), os.path.abspath('fixrot_dimer.xtc'))
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

    residues, t_slow, sd, indicators = collect_results(5)
    rem_inds = np.array([np.where(tmpresidues==residue)[0][0] for residue in residues])
    times, trajtimes, lipinds = times[rem_inds], trajtimes[rem_inds], lipinds[rem_inds]

    dt, comp = u.trajectory.ts.dt/1000, comp-2 #nanoseconds
    bframes, eframes = get_start_stop_frames(trajtimes[179], times[179], dt)
    sortinds = bframes.argsort()
    bframes.sort()
    eframes, lind = eframes[sortinds], lipinds[179][sortinds]
    tmp = [np.arange(b, e) for b, e in zip(bframes, eframes)]
    tmpL = [np.ones_like(np.arange(b, e))*l for b, e, l in zip(bframes, eframes, lipinds[179])]
    tmpI = [indic*np.ones((len(np.arange(b, e)), 5)) for b, e, indic in zip(bframes, eframes, indicators[179].T)]
    write_frames, write_Linds, write_Indics = np.concatenate([*tmp]), np.concatenate([*tmpL]).astype(int), np.concatenate([*tmpI])
    chol = u.select_atoms('resname CHOL')

    Ds = [WDensityAnalysis(chol, write_Indics, gridcenter=u.select_atoms(f'protein and resid {resid} and name BB').center_of_geometry(), xdim=30, ydim=30, zdim=30) for i in range(5)]
    [D.run(verbose=True) for D in Ds]
    [D.results.density.export(f"comp{i}.dx") for i,D in enumerate(Ds)]
