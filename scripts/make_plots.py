from basicrta import *
from multiprocessing import Pool, Lock
from basicrta import istarmap
import numpy as np
import MDAnalysis as mda
import os 
from tqdm import tqdm

if __name__ == "__main__":
    # Parts of code taken from Shep (Centrifuge3.py, SuperMCMC.py)
    # Execute in BaSiC-RTA-{cutoff} directory
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--prot')
    args = parser.parserargs()
    prot, ts = args.prot, 0.1
    
    residues, t_slow, sd, indicators = collect_results()
    plot_protein(residues, t_slow, sd, prot)
    check_results(residues, times, ts)
    plot_hists(times, indicators, residues)

