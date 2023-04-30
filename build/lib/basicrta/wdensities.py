from basicrta.wdensity import WDensityAnalysis
from basicrta import *
import numpy as np
import MDAnalysis as mda
import os
from tqdm import tqdm

if __name__ == "__main__":
    ncomp, resid = 5, 216
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

    residues, t_slow, sd, indicators = collect_results(ncomp)
    rem_inds = np.array([np.where(tmpresidues==residue)[0][0] for residue in residues])
    times, trajtimes, lipinds = times[rem_inds], trajtimes[rem_inds], lipinds[rem_inds]

    resids = np.array([int(res[1:]) for res in residues])
    index = np.where(resids==resid)[0][0]
    print(residues[index], index)
    dt = u.trajectory.ts.dt/1000 #nanoseconds

    bframes, eframes = get_start_stop_frames(trajtimes[index], times[index], dt)
    sortinds = bframes.argsort()
    bframes.sort()
    eframes, lind = eframes[sortinds], lipinds[index][sortinds]
    single_inds, multi_inds = np.where(times[index]==dt), np.where(times[index]!=dt)
    single_frames = bframes[single_inds]
   
    bframes, eframes = bframes[multi_inds], eframes[multi_inds]
    tmp = [np.arange(b, e) for b, e in zip(bframes, eframes)]
    tmpL = [np.ones_like(np.arange(b, e))*l for b, e, l in zip(bframes, eframes, lind[multi_inds])]
    tmpI = [indic*np.ones((len(np.arange(b, e)), ncomp)) for b, e, indic in zip(bframes, eframes, indicators[index].T[sortinds][multi_inds])]
    
    write_frames, write_Linds, write_Indics = np.concatenate([single_frames, *tmp]), np.concatenate([lind[single_inds], *tmpL]).astype(int), np.concatenate([indicators[index].T[sortinds][single_inds], *tmpI])
    
    protein = u.select_atoms('protein')
    chol = u.select_atoms('resname CHOL')
    write_sel = protein+chol.residues[0].atoms
    with mda.Writer("chol_traj.xtc", len(write_sel.atoms)) as W:
        for i, ts in tqdm(enumerate(u.trajectory[write_frames]), total=len(write_frames)):
            W.write(protein+chol.residues[write_Linds[i]].atoms)
        
    u_red = mda.Universe('prot_chol.pdb', 'chol_traj.xtc')
    chol_red = u_red.select_atoms('resname CHOL')

    D = WDensityAnalysis(chol_red, write_Indics, gridcenter=u.select_atoms(f'protein and resid {resid}').center_of_geometry(), xdim=30, ydim=30, zdim=30)
    D.run(verbose=True)
    [d.export(f'wcomp{i}.dx') for i, d in enumerate(D.results.densities)]
