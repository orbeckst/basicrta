#!/usr/env/bin python
from basicrta.functions import collect_n_plot

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--prot')
    args = parser.parser_args()

    prot = args.prot
    residues, t_slow, sd, indicators = collect_results()
    plot_protein(residues, t_slow, sd, prot)
