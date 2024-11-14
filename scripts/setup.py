import os                                                                       
import numpy as np                                                              
import MDAnalysis as mda                                                        
import pickle 

with open('contacts_7.0.pkl', 'rb') as f:
    a = pickle.load(f)                                         

rg = a.dtype.metadata['ag1'].residues                                
ids = rg.resids
names = rg.resnames                             
names = np.array([mda.lib.util.convert_aa_code(name) for name in names])        
uniqs = np.unique(a[:, 0]).astype(int) 

inds = np.array([np.where(ids==val)[0][0] for val in uniqs])
resids, resnames = ids[inds], names[inds]                                     
residues = np.array([f'{name}{resid}' for name, resid in zip(resnames, resids)])

with open('residue_list.csv', 'w+') as r:
    for residue in residues:             
        r.write(f'{residue},')                                                                        
                                                                                
if not os.path.exists('basicrta-7.0'):                                         
    os.mkdir('basicrta-7.0')                                                   
                                                                                
os.chdir('basicrta-7.0')                                                       
                                                                                
for residue in residues:                                                        
    if not os.path.exists(residue):                                             
        os.mkdir(residue)                                                       


