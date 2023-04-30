#!/usr/bin/env python

from basicrta.functions import collect_results, plot_protein, tm
import MDAnalysis as mda
import numpy as np
from tqdm import tqdm
from MDAnalysis.analysis import align

if __name__ =="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb',help='structure to b color')
    parser.add_argument('--outname',help='name of b_colored pdb')
    parser.add_argument('--fixed_pdb',help='pdb with correct resids/positions')
    parser.add_argument('--prot',help='pdb with correct resids/positions')
    args = parser.parse_args()
    pdb, fixed_pdb, outname = args.pdb, args.fixed_pdb, args.outname 
    prot = args.prot

    residues, t_slow, sd, indicators = collect_results()
    plot_protein(residues, t_slow, sd, prot)

    u = mda.Universe(pdb)
    ufixed = mda.Universe(fixed_pdb)
    prot = u.select_atoms('protein')
    protBB = u.select_atoms('protein and name BB or name CA')
    protF = ufixed.select_atoms('protein and name BB or name CA')

    protF.masses = protBB.masses
    resids = ufixed.select_atoms('protein').residues.resids
    prot.residues.resids = resids
    align.alignto(protBB, protF)

    data = np.zeros((len(prot.residues)))
    for res,t in zip(residues, t_slow):
        ind = np.where(resids==int(res[1:]))[0]
        data[ind] = t

    u.add_TopologyAttr('tempfactors')
    for k in tqdm(range(len(data))):
        res = u.select_atoms(f'protein and resid {prot.residues.resids[k]}')
        temp = data[k]
        res.tempfactors = np.round(temp,2)

    if not outname:
        prot.write('result_check/{0}_bcolored_{1}.pdb'.format(pdb,data))
    else:
        prot.write('result_check/{0}'.format(outname))

