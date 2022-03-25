#!/usr/env/bin python
from basicrta.functions import collect_n_plot

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resids')
    parser.add_argument('--comps')
    args = parser.parse_args()
    resids, comps = args.resids, args.comps
    print(resids, comps)
    collect_n_plot(resids, comps)
