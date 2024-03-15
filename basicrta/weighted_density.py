from basicrta.wdensity import WDensityAnalysis
import numpy as np
import MDAnalysis as mda
import os
from tqdm import tqdm
from basicrta.util import get_start_stop_frames
import pickle
# from MDAnalysis.lib.util import realpath


class MapKinetics(object):
    def __init__(self, gibbs, contacts):
        self.gibbs = gibbs
        self.cutoff = float(contacts.split('/')[-1].strip('.pkl').
                            split('_')[-1])
        self.write_sel = None
        self.contacts = contacts

        with open(contacts, 'rb') as f:
            tmpcontacts = pickle.load(f)
        metadata = tmpcontacts.dtype.metadata
        self.ag1 = metadata['ag1']
        self.ag2 = metadata['ag2']
        self.ts = metadata['ts']
        self.utop = metadata['top']
        self.utraj = metadata['traj']
        del tmpcontacts

        self.dataname = (f'basicrta-{self.cutoff}/{self.gibbs.residue}/'
                         f'den_write_data.npy')
        self.topname = (f'basicrta-{self.cutoff}/{self.gibbs.residue}/'
                        f'reduced.gro')
        self.fulltraj = (f'basicrta-{self.cutoff}/{self.gibbs.residue}/'
                         f'chol_traj_all.xtc')

    def _create_data(self):
        from numpy.lib.format import open_memmap
        with open(self.contacts, 'rb') as f:
            contacts = pickle.load(f)

        resid = int(self.gibbs.residue[1:])
        ncomp = self.gibbs.processed_results.ncomp

        times = np.array(contacts[contacts[:, 0] == resid][:, 3])
        trajtimes = np.array(contacts[contacts[:, 0] == resid][:, 2])
        lipinds = np.array(contacts[contacts[:, 0] == resid][:, 1])
        del contacts

        indicators = self.gibbs.processed_results.indicator

        bframes, eframes = get_start_stop_frames(trajtimes, times, self.ts)
        tmplens = [len(np.arange(b, e)) for b, e in zip(bframes, eframes)]
        totlen = sum(tmplens)
        write_data = open_memmap(self.dataname, mode='w+', dtype=np.float64,
                                 shape=(totlen, ncomp+2))

        j = 0
        for b, e, l, i in tqdm(zip(bframes, eframes, lipinds, indicators),
                               total=len(bframes)):
            tmp = np.arange(b, e)
            tmpl = np.ones_like(np.arange(b, e)) * l
            tmpi = i * np.ones((len(np.arange(b, e)), ncomp))

            write_data[j:j+len(tmp), 0] = tmp
            write_data[j:j+len(tmp), 1] = tmpl
            write_data[j:j+len(tmp), 2:] = tmpi
            j += len(tmp)

    def create_traj(self, top_n=None):
        if os.path.exists(self.fulltraj) and top_n is None:
            raise FileExistsError(f'{self.fulltraj} exists, remove then rerun')

        write_ag = self.ag1.atoms + self.ag2.residues[0].atoms
        write_ag.atoms.write(self.topname)
        if not os.path.exists(self.dataname):
            self._create_data()

        tmp = np.load(self.dataname, mmap_mode='r')
        u = mda.Universe(f'{self.utop}', f'{self.utraj}')
        ag1 = u.atoms[self.ag1.indices]
        ag2 = u.atoms[self.ag2.indices]
        if top_n is not None:
            sortinds = [tmp[:, i+2].argsort()[::-1][:top_n] for i in
                        range(self.gibbs.processed_results.ncomp)]
            for k in range(self.gibbs.processed_results.ncomp):
                swf = tmp[sortinds[k], 0].astype(int)
                swl = tmp[sortinds[k], 1].astype(int)
                with mda.Writer(f'basicrta-{self.cutoff}/{self.gibbs.residue}/'
                                f'chol_traj_comp{k}_top{top_n}.xtc',
                                len(write_ag.atoms)) as W:
                    for i, ts in tqdm(enumerate(u.trajectory[swf]),
                                      total=len(swf),
                                      desc=f'writing component {k}'):
                        W.write(ag1 + ag2.select_atoms(f'resid {swl[i]}'))

        else:
            with mda.Writer(self.fulltraj, len(write_ag.atoms)) as W:
                for i, ts in tqdm(enumerate(u.trajectory[tmp[:, 0].
                                  astype(int)]), total=len(tmp),
                                  desc='writing trajectory'):
                    W.write(ag1 + ag2.select_atoms(f'resid {int(tmp[i, 1])}'))

    def weighted_densities(self, step=1, top_n=None, filterP=0):
        if not os.path.exists(self.fulltraj):
            self.create_traj()

        resid = int(self.gibbs.residue[1:])
        tmp = np.load(self.dataname, mmap_mode='r+')
        wi = tmp[:, 2:]

        # filter anything less than 50% certain
        if filterP > 0:
            wi[wi < filterP] = 0

        # filter_inds = np.where(wi > filterP)
        # wi = wi[filter_inds[0]][::self.step]
        # comp_inds = [np.where(filter_inds[1] == i)[0] for i in
        #              range(self.gibbs.processed_results.ncomp)]

        u_red = mda.Universe(self.topname, self.fulltraj)
        chol_red = u_red.select_atoms('not protein')

        if top_n is None:
            from basicrta.pwdensity import WDensityAnalysis
            u_red = mda.Universe(self.topname, self.fulltraj)
            chol_red = u_red.select_atoms('not protein')
            d = WDensityAnalysis(chol_red, wi,
                                 gridcenter=u_red.select_atoms(f'protein and '
                                                               f'resid {resid}')
                                 .center_of_geometry(), xdim=40, ydim=40,
                                 zdim=40)
            d.run(verbose=True, step=step)
            if step > 1:
                outnames = [(f'basicrta-{self.cutoff}/{self.gibbs.residue}/'
                             f'wcomp{k}_all_step{step}.dx')
                            for k in range(self.gibbs.processed_results.ncomp)]
            else:
                outnames = [(f'basicrta-{self.cutoff}/{self.gibbs.residue}/'
                             f'wcomp{k}_all.dx') for k in
                            range(self.gibbs.processed_results.ncomp)]

            [den.export(outnames[k]) for k, den in
             enumerate(d.results.densities)]

        else:
            from basicrta.wdensity import WDensityAnalysis
            sortinds = [wi[:, i].argsort()[::-1] for i in
                        range(self.gibbs.processed_results.ncomp)]

            for k in range(self.gibbs.processed_results.ncomp):
                frames = np.where(wi[sortinds[k], k] > 0)[0][:top_n:step]
                tmpwi = wi[frames, k]
                d = WDensityAnalysis(chol_red, tmpwi,
                                     gridcenter=u_red.select_atoms(f'protein '
                                                                   f'and resid '
                                                                   f'{resid}')
                                     .center_of_geometry(), xdim=40, ydim=40,
                                     zdim=40)
                d.run(verbose=True, frames=sortinds[k][frames])
                if step > 1:
                    outname = (f'basicrta-{self.cutoff}/{self.gibbs.residue}/'
                               f'wcomp{k}_top{top_n}_step{step}.dx')
                else:
                    outname = (f'basicrta-{self.cutoff}/{self.gibbs.residue}/'
                               f'wcomp{k}_top{top_n}.dx')

                d.results.density.export(outname)


if __name__ == "__main__":
    print('not implemented')

