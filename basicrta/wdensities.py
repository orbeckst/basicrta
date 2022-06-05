   1:
    with open('contacts.metadata', 'r') as data:
        line = data.readlines()[1].split(',')
        trajlen, protlen, liplen, sel, ts = int(line[0]), int(line[1]), int(line[2]), line[3], float(line[4])

    nproc = 5
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

    #rem_inds = get_remaining_residue_inds(residues)
    #times, trajtimes, lipinds = times[rem_inds], trajtimes[rem_inds], lipinds[rem_inds]
    residues, t_slow, sd, indicators = collect_results(5)
    rem_inds = np.array([np.where(tmpresidues==residue)[0][0] for residue in residues])
    times, trajtimes, lipinds = times[rem_inds], trajtimes[rem_inds], lipinds[rem_inds]
   2: pwd
   3: cd ../../../
   4:
    with open('contacts.metadata', 'r') as data:
        line = data.readlines()[1].split(',')
        trajlen, protlen, liplen, sel, ts = int(line[0]), int(line[1]), int(line[2]), line[3], float(line[4])

    nproc = 5
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

    #rem_inds = get_remaining_residue_inds(residues)
    #times, trajtimes, lipinds = times[rem_inds], trajtimes[rem_inds], lipinds[rem_inds]
    residues, t_slow, sd, indicators = collect_results(5)
    rem_inds = np.array([np.where(tmpresidues==residue)[0][0] for residue in residues])
    times, trajtimes, lipinds = times[rem_inds], trajtimes[rem_inds], lipinds[rem_inds]
   5: import MDAnalysis as mda
   6:
from basicrta import *
import multiprocessing
from multiprocessing import Pool, Lock
from basicrta import istarmap
import numpy as np
import MDAnalysis as mda
import os 
from tqdm import tqdm
   7:
    with open('contacts.metadata', 'r') as data:
        line = data.readlines()[1].split(',')
        trajlen, protlen, liplen, sel, ts = int(line[0]), int(line[1]), int(line[2]), line[3], float(line[4])

    nproc = 5
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

    #rem_inds = get_remaining_residue_inds(residues)
    #times, trajtimes, lipinds = times[rem_inds], trajtimes[rem_inds], lipinds[rem_inds]
    residues, t_slow, sd, indicators = collect_results(5)
    rem_inds = np.array([np.where(tmpresidues==residue)[0][0] for residue in residues])
    times, trajtimes, lipinds = times[rem_inds], trajtimes[rem_inds], lipinds[rem_inds]
   8: a = np.load('lipswap_contacts_7.0.npy')
   9:
    with open('contacts.metadata', 'r') as data:
        line = data.readlines()[1].split(',')
        trajlen, protlen, liplen, sel, ts = int(line[0]), int(line[1]), int(line[2]), line[3], float(line[4])

    nproc = 5
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

    #rem_inds = get_remaining_residue_inds(residues)
    #times, trajtimes, lipinds = times[rem_inds], trajtimes[rem_inds], lipinds[rem_inds]
    residues, t_slow, sd, indicators = collect_results(5)
    rem_inds = np.array([np.where(tmpresidues==residue)[0][0] for residue in residues])
    times, trajtimes, lipinds = times[rem_inds], trajtimes[rem_inds], lipinds[rem_inds]
  10:
tmp = [np.arange(b, e) for b, e in zip(bframes, eframes)]
tmpL = [np.ones_like(np.arange(b, e))*l for b, e, l in zip(bframes, eframes, lipinds[179])]
tmpI = [indic*np.ones((len(np.arange(b, e)), 5)) for b, e, indic in zip(bframes, eframes, indicators[179].T)]
  11:
    dt, comp = u.trajectory.ts.dt/1000, comp-2 #nanoseconds
    bframes, eframes = get_start_stop_frames(trajtimes[179], times[179], dt)
    sortinds = bframes.argsort()
    bframes.sort()
    eframes, lind = eframes[sortinds], lipinds[179][sortinds]
    tmp = [np.arange(b, e) for b, e in zip(bframes, eframes)]
    tmpL = [np.ones_like(np.arange(b, e))*l for b, e, l in zip(bframes, eframes, lind)]
    write_frames, write_Linds = np.concatenate([*tmp]), np.concatenate([*tmpL]).astype(int)
  12: comp = 5
  13:
    dt, comp = u.trajectory.ts.dt/1000, comp-2 #nanoseconds
    bframes, eframes = get_start_stop_frames(trajtimes[179], times[179], dt)
    sortinds = bframes.argsort()
    bframes.sort()
    eframes, lind = eframes[sortinds], lipinds[179][sortinds]
    tmp = [np.arange(b, e) for b, e in zip(bframes, eframes)]
    tmpL = [np.ones_like(np.arange(b, e))*l for b, e, l in zip(bframes, eframes, lind)]
    write_frames, write_Linds = np.concatenate([*tmp]), np.concatenate([*tmpL]).astype(int)
  14:
tmp = [np.arange(b, e) for b, e in zip(bframes, eframes)]
tmpL = [np.ones_like(np.arange(b, e))*l for b, e, l in zip(bframes, eframes, lipinds[179])]
tmpI = [indic*np.ones((len(np.arange(b, e)), 5)) for b, e, indic in zip(bframes, eframes, indicators[179].T)]
  15: write_frames, write_Linds, write_Indics = np.concatenate([*tmp]), np.concatenate([*tmpL]).astype(int), np.concatenate([*tmpI])
  16: from basicrta.wdensity import WDensityAnalysis
  17:
Ds = [WDensityAnalysis(chol, write_Indics, gridcenter=u.select_atoms(f'protein and resid {resid} and name BB').center_of_geometry(), xdim=30, ydim=30, zdim=30) for i in range(5)]
[D.run(verbose=True) for D in Ds]
[D.results.density.export(f"comp{i}.dx") for i,D in enumerate(Ds)]
  18: chol = u.select_atoms('resname CHOL')
  19:
Ds = [WDensityAnalysis(chol, write_Indics, gridcenter=u.select_atoms(f'protein and resid {resid} and name BB').center_of_geometry(), xdim=30, ydim=30, zdim=30) for i in range(5)]
[D.run(verbose=True) for D in Ds]
[D.results.density.export(f"comp{i}.dx") for i,D in enumerate(Ds)]
  20: resid = 313
  21:
Ds = [WDensityAnalysis(chol, write_Indics, gridcenter=u.select_atoms(f'protein and resid {resid} and name BB').center_of_geometry(), xdim=30, ydim=30, zdim=30) for i in range(5)]
[D.run(verbose=True) for D in Ds]
[D.results.density.export(f"comp{i}.dx") for i,D in enumerate(Ds)]
  22: DensityAnalysis
  23: from MDAnalysis.analysis.density import DensityAnalysis
  24:
Ds = [WDensityAnalysis(chol, write_Indics, gridcenter=u.select_atoms(f'protein and resid {resid} and name BB').center_of_geometry(), xdim=30, ydim=30, zdim=30) for i in range(5)]
[D.run(verbose=True) for D in Ds]
[D.results.density.export(f"comp{i}.dx") for i,D in enumerate(Ds)]
  25: from basicrta.wdensity import WDensityAnalysis
  26:
Ds = [WDensityAnalysis(chol, write_Indics, gridcenter=u.select_atoms(f'protein and resid {resid} and name BB').center_of_geometry(), xdim=30, ydim=30, zdim=30) for i in range(5)]
[D.run(verbose=True) for D in Ds]
[D.results.density.export(f"comp{i}.dx") for i,D in enumerate(Ds)]
  27:
Ds = [WDensityAnalysis(chol, write_Indics, gridcenter=u.select_atoms(f'protein and resid {resid} and name BB').center_of_geometry(), xdim=30, ydim=30, zdim=30) for i in range(5)]
[D.run(verbose=True) for D in Ds]
[D.results.density.export(f"comp{i}.dx") for i,D in enumerate(Ds)]
  28:
Ds = [WDensityAnalysis(chol, write_Indics, gridcenter=u.select_atoms(f'protein and resid {resid} and name BB').center_of_geometry(), xdim=30, ydim=30, zdim=30) for i in range(5)]
[D.run(verbose=True) for D in Ds]
[D.results.density.export(f"comp{i}.dx") for i,D in enumerate(Ds)]
  29:
Ds = [WDensityAnalysis(chol, write_Indics, gridcenter=u.select_atoms(f'protein and resid {resid} and name BB').center_of_geometry(), xdim=30, ydim=30, zdim=30) for i in range(5)]
[D.run(verbose=True) for D in Ds]
[D.results.density.export(f"comp{i}.dx") for i,D in enumerate(Ds)]
  30:
Ds = [WDensityAnalysis(chol, write_Indics, gridcenter=u.select_atoms(f'protein and resid {resid} and name BB').center_of_geometry(), xdim=30, ydim=30, zdim=30) for i in range(5)]
[D.run(verbose=True) for D in Ds]
[D.results.density.export(f"comp{i}.dx") for i,D in enumerate(Ds)]
  31: from basicrta.wdensity import WDensityAnalysis
  32:
Ds = [WDensityAnalysis(chol, write_Indics, gridcenter=u.select_atoms(f'protein and resid {resid} and name BB').center_of_geometry(), xdim=30, ydim=30, zdim=30) for i in range(5)]
[D.run(verbose=True) for D in Ds]
[D.results.density.export(f"comp{i}.dx") for i,D in enumerate(Ds)]
  33: %history?
