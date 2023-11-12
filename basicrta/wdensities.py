from basicrta.wdensity import WDensityAnalysis
from basicrta import *
import numpy as np
import MDAnalysis as mda
import os
from tqdm import tqdm
import pickle, bz2
from basicrta.functions import process_gibbs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--residue')
    parser.add_argument('--cutoff')
    parser.add_argument('--filterP', nargs='?', default=0.85)
    parser.add_argument('--step', nargs='?', default=1)
    args = parser.parse_args()

    residue, cutoff = args.residue, float(args.cutoff)
    resid, step, filterP = int(residue[1:]), int(args.step), float(args.filterP)
    
    r = pickle.load(bz2.BZ2File(f'BaSiC-RTA-{cutoff}/{residue}/results_20000.pkl.bz2', 'rb'))
    rp, rpinds = process_gibbs(r)
    ncomp = rp.ncomp
    
    files = ['1/fixrot_dimer.xtc', '2/fixrot_dimer.xtc', '3/fixrot_dimer.xtc']
    u = mda.Universe(os.path.abspath('step7_production.tpr'), files)
    uf = mda.Universe('step7_fixed.pdb')
    protein = u.select_atoms('protein')
    chol = u.select_atoms('resname CHOL')
    
    write_sel = protein+chol.residues[0].atoms
    resids = uf.select_atoms('protein').residues.resids
    index = np.where(resids==resid)[0][0]
    if not os.path.exists(f'BaSiC-RTA-{cutoff}/{residue}/den_write_data_step{step}.npy'):
        a = np.load(f'lipswap_contacts_combined_{cutoff}.npy')
        
        times = np.array(a[a[:, 0] == index][:, 3])
        trajtimes = np.array(a[a[:, 0] == index][:, 2])
        lipinds = np.array(a[a[:, 0] == index][:, 1])
        dt = u.trajectory.ts.dt/1000 #nanoseconds

        sortinds = np.argsort([line.mean() for line in rp.rates.T])
        indicators = (r.indicator.T/r.indicator.sum(axis=1)).T[inds][sortinds]

        bframes, eframes = get_start_stop_frames(trajtimes, times, dt)
        tmp = [np.arange(b, e) for b, e in zip(bframes, eframes)]
        tmpL = [np.ones_like(np.arange(b, e))*l for b, e, l in zip(bframes, eframes, lipinds)]
        tmpI = [indic*np.ones((len(np.arange(b, e)), ncomp)) for b, e, indic in zip(bframes, eframes, indicators.T)]
            
        write_frames = np.concatenate([*tmp]).astype(int) 
        write_Linds = np.concatenate([*tmpL]).astype(int)
        write_Indics = np.concatenate([*tmpI])
        
        wf, wl, wi = write_frames, write_Linds, write_Indics
        darray = np.zeros((len(wf),ncomp+2))
        darray[:, 0], darray[:,1], darray[:,2:] = wf, wl, wi
        np.save(f'BaSiC-RTA-{cutoff}/{residue}/den_write_data_step{step}', darray)
    else:
        tmp = np.load(f'BaSiC-RTA-{cutoff}/{residue}/den_write_data_step{step}.npy')
        wf, wl, wi = tmp[:,0], tmp[:,1], tmp[:,2:]

    if not os.path.exists(f"BaSiC-RTA-{cutoff}/{residue}/chol_traj_step{step}.xtc"):
        with mda.Writer(f"BaSiC-RTA-{cutoff}/{residue}/chol_traj_step{step}.xtc", len(write_sel.atoms)) as W:
            for i, ts in tqdm(enumerate(u.trajectory[wf[::step]]), total=len(wf)//step+1, desc='writing single lipid trajectory'):
                W.write(protein+chol.residues[wl[::step][i]].atoms)
    
    u_red = mda.Universe('prot_chol.gro',f'BaSiC-RTA-{cutoff}/{residue}/chol_traj_step{step}.xtc')
    chol_red = u_red.select_atoms('resname CHOL')

    filter_inds = np.where(wi[::step]>filterP)
    wf = wf[::step][filter_inds[0]].astype(int)
    wl = wl[::step][filter_inds[0]].astype(int)
    wi = wi[::step][filter_inds[0]]
    comp_inds = [np.where(filter_inds[1]==i)[0] for i in range(ncomp)]
    

    for i in range(ncomp):
        D = WDensityAnalysis(chol_red, wi[comp_inds[i], i], gridcenter=u_red.select_atoms(f'protein and resid {index}').center_of_geometry(), xdim=40, ydim=40, zdim=40)
        D.run(verbose=True, frames=filter_inds[0][comp_inds[i]])
        D.results.density.export(f'BaSiC-RTA-{cutoff}/{residue}/wcomp{i}_step{step}_p{filterP}.dx')
