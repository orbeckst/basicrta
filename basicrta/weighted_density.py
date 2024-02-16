from basicrta.wdensity import WDensityAnalysis
import numpy as np
import MDAnalysis as mda
import os
from tqdm import tqdm
from basicrta.util import get_start_stop_frames
import pickle
# from MDAnalysis.lib.util import realpath


class MapKinetics(object):
    def __init__(self, gibbs, contacts, n=None):
        self.gibbs, self.N = gibbs, n
        self.cutoff = float(contacts.split('/')[-1].strip('.pkl').
                            split('_')[-1])
        self.write_sel = None
        with open(contacts, 'rb') as f:
            self.contacts = pickle.load(f)

        metadata = self.contacts.dtype.metadata
        self.ag1, self.ag2 = metadata['ag1'], metadata['ag2']
        self.u = metadata['u']

        self.dataname = f'{self.gibbs.residue}/den_write_data_all.npy'
        self.topname = f'{self.gibbs.residue}/reduced.pdb'
        self.fulltraj = f'{self.gibbs.residue}/chol_traj_all.xtc'


    def _create_data(self):
            contacts = self.contacts
            resid = int(self.gibbs.residue[1:])
            ncomp = self.gibbs.processed_results.ncomp

            times = np.array(contacts[contacts[:, 0] == resid][:, 3])
            trajtimes = np.array(contacts[contacts[:, 0] == resid][:, 2])
            lipinds = np.array(contacts[contacts[:, 0] == resid][:, 1])
            dt = self.u.trajectory.ts.dt/1000 #nanoseconds

            indicators = self.gibbs.processed_results.indicator

            bframes, eframes = get_start_stop_frames(trajtimes, times, dt)
            tmp = [np.arange(b, e) for b, e in zip(bframes, eframes)]
            tmpL = [np.ones_like(np.arange(b, e)) * l for b, e, l in
                    zip(bframes, eframes, lipinds)]
            tmpI = [indic * np.ones((len(np.arange(b, e)), ncomp))
                    for b, e, indic in zip(bframes, eframes, indicators)]

            write_frames = np.concatenate([*tmp]).astype(int)
            write_linds = np.concatenate([*tmpL]).astype(int)
            write_indics = np.concatenate([*tmpI])

            darray = np.zeros((len(write_frames), ncomp + 2))
            darray[:, 0], darray[:, 1], darray[:, 2:] = (write_frames,
                                                         write_linds,
                                                         write_indics)
            np.save(self.dataname, darray)


    def create_traj(self):
        write_ag = self.ag1.atoms + self.ag2.residues[0].atoms
        write_ag.atoms.write(self.topname)

        if not os.path.exists(self.dataname):
            self._create_data()

        tmp = np.load(self.dataname)
        wf, wl, wi = tmp[:, 0].astype(int), tmp[:, 1].astype(int), tmp[:, 2:]

        if self.N is not None:
            sortinds = [wi[:, i].argsort()[::-1][:self.N] for i in
                        range(self.gibbs.processed_results.ncomp)]
            for k in range(self.gibbs.processed_results.ncomp):
                swf, swl = wf[sortinds[k]], wl[sortinds[k]]
                with mda.Writer(f'{self.gibbs.residue}/chol_traj_comp{k}_top'
                                f'{self.N}.xtc', len(write_ag.atoms)) as W:
                    for i, ts in tqdm(enumerate(self.u.trajectory[swf]),
                                      total=len(swf),
                                      desc=f'writing component {k}'):
                        W.write(self.ag1 +
                                self.ag2.select_atoms(f'resid {swl[i]}').atoms)

        else:
            with mda.Writer(self.fulltraj, len(write_ag.atoms)) as W:
                for i, ts in tqdm(enumerate(self.u.trajectory[wf]),
                                  total=len(wf), desc='writing trajectory'):
                    W.write(self.ag1 +
                            self.ag2.select_atoms(f'resid {wl[i]}').atoms)


    def weighted_densities(self, step=1):
        if not os.path.exists(self.fulltraj):
            self.create_traj()

        resid = int(self.gibbs.residue[1:])
        tmp = np.load(self.dataname)
        wi = tmp[:, 2:]

        # filter_inds = np.where(wi > filterP)
        # wi = wi[filter_inds[0]][::self.step]
        # comp_inds = [np.where(filter_inds[1] == i)[0] for i in
        #              range(self.gibbs.processed_results.ncomp)]

        u_red = mda.Universe(self.topname, self.fulltraj)
        chol_red = u_red.select_atoms('not protein')

        sortinds = [wi[:, i].argsort()[::-1] for i in
                    range(self.gibbs.processed_results.ncomp)]

        for k in range(self.gibbs.processed_results.ncomp):
            frames = np.where(wi[sortinds[k], k] > 0)[0][::step]
            tmpwi = wi[frames, k]
            d = WDensityAnalysis(chol_red, tmpwi,
                                 gridcenter=u_red.select_atoms(f'protein and '
                                                               f'resid {resid}')
                                 .center_of_geometry(), xdim=40, ydim=40,
                                 zdim=40)
            d.run(verbose=True, frames=sortinds[k][frames])
            if self.N is not None:
                outname = f'{self.gibbs.residue}/wcomp{k}_top{self.N}.dx'
            else:
                outname = f'{self.gibbs.residue}/wcomp{k}_all.dx'

            d.results.density.export(outname)



if __name__ == "__main__":
    print('not implemented')

