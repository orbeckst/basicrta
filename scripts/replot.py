if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resids')
    parser.add_argument('--comps')
    args = parser.parse_args()
    resids, comps = args.resids, args.comps
    collect_n_plot(resids)
