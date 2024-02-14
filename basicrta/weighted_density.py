from basicrta.wdensity import WDensityAnalysis
import numpy as np
import MDAnalysis as mda
import os
from tqdm import tqdm
from basicrta.util import get_start_stop_frames
import pickle
from MDAnalysis.lib.util import realpath


class WeightedDensity(object):
    def __init__(self, gibbs, contacts, step=1, N=1000):
        self.gibbs, self.N, self.step = gibbs, N, step
        self.cutoff = float(contacts.split('/')[-1].strip('.pkl').
                            split('_')[-1])
        self.write_sel = None
        with open(contacts, 'rb') as f:
            self.contacts = pickle.load(f)

        metadata = self.contacts.dtype.metadata
        top = realpath(metadata['top'])
        traj = realpath(metadata['traj'])
        self.ag1, self.ag2 = metadata['ag1'], metadata['ag2']
        self.u = mda.Universe(top, traj)
        self.dataname = (f'{self.gibbs.residue}/den_write_data_'
                         f'step{self.step}.npy')
        self.topname = (f'{self.gibbs.residue}/reduced.pdb')
        self.trajname = (f'{self.gibbs.residue}/chol_traj_step{self.step}.xtc')


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
        darray[:, 0], darray[:, 1], darray[:, 2:] = (write_frames, write_linds,
                                                     write_indics)
        np.save(self.dataname, darray)


    def _create_traj(self):
        write_ag = self.ag1.atoms + self.ag2.residues[0].atoms
        write_ag.atoms.write(self.topname)

        if not os.path.exists(self.dataname):
            self._create_data()

        tmp = np.load(self.dataname)
        wf, wl, wi = tmp[:, 0], tmp[:, 1], tmp[:, 2:]

        if not os.path.exists(self.trajname):
            with mda.Writer(self.trajname, len(write_ag.atoms)) as W:
                for i, ts in tqdm(enumerate(self.u.trajectory[wf[::self.step]]),
                                  total=len(wf)//(self.step+1),
                                  desc='writing trajectory'):
                    W.write(self.ag1.atoms+
                            self.ag2.residues[wl[::self.step][i]].atoms)


    def run(self):
        if not os.path.exists(self.trajname):
            self._create_traj()

        resid = int(self.gibbs.residue[1:])
        tmp = np.load(self.dataname)
        wi = tmp[:, 2:]

        # filter_inds = np.where(wi > filterP)
        # wi = wi[filter_inds[0]][::self.step]
        # comp_inds = [np.where(filter_inds[1] == i)[0] for i in
        #              range(self.gibbs.processed_results.ncomp)]

        u_red = mda.Universe(self.topname, self.trajname)
        chol_red = u_red.select_atoms('not protein')

        sortinds = [wi[:, i].argsort()[::-1] for i in
                    range(self.gibbs.processed_results.ncomp)]
        for i in range(self.gibbs.processed_results.ncomp):
            D = WDensityAnalysis(chol_red, wi[sortinds[i], i],
                                 gridcenter=u_red.select_atoms(f'protein and '
                                                               f'resid {resid}')
                                 .center_of_geometry(), xdim=40, ydim=40,
                                 zdim=40)
            D.run(verbose=True, frames=sortinds[i])
            D.results.density.export(f'basicrta-{self.cutoff}/'
                                     f'{self.gibbs.processed_results.residue}/'
                                     f'wcomp{i}_top{self.N}.dx')



if __name__ == "__main__":
    print('not implemented')

