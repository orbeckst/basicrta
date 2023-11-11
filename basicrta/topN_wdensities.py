from basicrta.wdensity import WDensityAnalysis
from basicrta import *
import numpy as np
import MDAnalysis as mda
import os
from tqdm import tqdm
import pickle

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--residue')
    parser.add_argument('--cutoff')
    parser.add_argument('--ncomp')
    parser.add_argument('--step', nargs='?', default=1)
    parser.add_argument('--nframes', nargs='?', default=1000)
    args = parser.parse_args()

    ncomp, residue, cutoff = int(args.ncomp), args.residue, float(args.cutoff)
    resid, step, nframes = int(residue[1:]), int(args.step), int(args.nframes)
    
    files = ['1/fixrot_dimer.xtc', '2/fixrot_dimer.xtc', '3/fixrot_dimer.xtc']
    u = mda.Universe(os.path.abspath('step7_production.tpr'), files)
    uf = mda.Universe('step7_fixed.pdb')
    protein = u.select_atoms('protein')
    chol = u.select_atoms('resname CHOL')
    
    write_sel = protein+chol.residues[0].atoms
    resids = uf.select_atoms('protein').residues.resids
    index = np.where(resids==resid)[0][0]
    if not os.path.exists(f'BaSiC-RTA-{cutoff}/{residue}/den_write_data_step{step}_ncomp{ncomp}.npy'):
        a = np.load(f'lipswap_contacts_combined_{cutoff}.npy')
        
        times = np.array(a[a[:, 0] == index][:, 3])
        trajtimes = np.array(a[a[:, 0] == index][:, 2])
        lipinds = np.array(a[a[:, 0] == index][:, 1])
        dt = u.trajectory.ts.dt/1000 #nanoseconds

        with open(f'BaSiC-RTA-{cutoff}/{residue}/K{ncomp}_results.pkl', 'rb') as f:
            pc = pickle.load(f)
        sortinds = np.argsort([line.mean() for line in pc.rates])
        indicators = pc.indicator[sortinds]/pc.indicator.sum(axis=0)

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
        np.save(f'BaSiC-RTA-{cutoff}/{residue}/den_write_data_step{step}_ncomp{ncomp}', darray)
    else:
        tmp = np.load(f'BaSiC-RTA-{cutoff}/{residue}/den_write_data_step{step}_ncomp{ncomp}.npy')
        wf, wl, wi = tmp[:,0].astype(int), tmp[:,1].astype(int), tmp[:,2:]

    if not os.path.exists(f"BaSiC-RTA-{cutoff}/{residue}/chol_traj_comp{ncomp}_step{step}.xtc"):
        with mda.Writer(f"BaSiC-RTA-{cutoff}/{residue}/chol_traj_comp{ncomp}_step{step}.xtc", len(write_sel.atoms)) as W:
            for i, ts in tqdm(enumerate(u.trajectory[wf[::step]]), total=len(wf)//step+1, desc='writing single lipid trajectory'):
                W.write(protein+chol.residues[wl[::step][i]].atoms)
   
    sortinds = [wi[:,i].argsort()[::-1] for i in range(ncomp)]
    u_red = mda.Universe('prot_chol.gro',f'BaSiC-RTA-{cutoff}/{residue}/chol_traj_comp{ncomp}_step{step}.xtc')
    chol_red = u_red.select_atoms('resname CHOL')

    for j in range(ncomp):
        with mda.Writer(f"BaSiC-RTA-{cutoff}/{residue}/chol_traj_comp{j+1}of{ncomp}_N{nframes}.xtc", len(write_sel.atoms)) as W:
            for i, ts in tqdm(enumerate(u.trajectory[wf[sortinds[j]][:nframes]]), total=nframes, desc='writing single lipid trajectory'):
                W.write(protein+chol.residues[wl[sortinds[j]][i]].atoms)

    for i in range(ncomp):
        D = WDensityAnalysis(chol_red, wi[sortinds[i], i][:nframes], gridcenter=u_red.select_atoms(f'protein and resid {index}').center_of_geometry(), xdim=40, ydim=40, zdim=40)
        D.run(verbose=True, frames=sortinds[i][:nframes])
        D.results.density.export(f'BaSiC-RTA-{cutoff}/{residue}/wcomp{i}_step{step}_ncomp{ncomp}_N{nframes}.dx')

