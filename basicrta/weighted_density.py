from basicrta.wdensity import WDensityAnalysis
import numpy as np
import MDAnalysis as mda
import os
from tqdm import tqdm
from basicrta.util import get_start_stop_frames


class WeightedDensity(object):
    def __init__(self, gibbs, u, contacts, step=1, N=1000):
        self.gibbs, self.u, self.N = gibbs, u, N
        self.contacts, self.step = contacts, step
        self.cutoff = float(contacts.split('/')[-1].strip('.npy').
                            split('_')[-1])
        self.write_sel = None
        self.dataname = (f'basicrta-{self.cutoff}/{self.gibbs.residue}/'
                         f'den_write_data_step{self.step}.npy')
        self.trajname = (f'basicrta-{self.cutoff}/{self.gibbs.residue}/'
                         f'chol_traj_step{self.step}.xtc')


    def _prepare(self):
        print('no nothing')


    def _create_data(self):
        contacts = np.load(self.contacts, mmap_mode='r')
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
                for b, e, indic in zip(bframes, eframes, indicators.T)]

        write_frames = np.concatenate([*tmp]).astype(int)
        write_linds = np.concatenate([*tmpL]).astype(int)
        write_indics = np.concatenate([*tmpI])

        darray = np.zeros((len(write_frames), ncomp + 2))
        darray[:, 0], darray[:, 1], darray[:, 2:] = (write_frames, write_linds,
                                                     write_indics)
        np.save(self.dataname, darray)


    def _create_traj(self):
        protein = self.u.select_atoms('protein')
        chol = self.u.select_atoms('resname CHOL')
        write_sel = protein + chol.residues[0].atoms

        if not os.path.exists(self.dataname):
            self._create_data()

        tmp = np.load(self.dataname)
        wf, wl, wi = tmp[:, 0], tmp[:, 1], tmp[:, 2:]

        if not os.path.exists(self.trajname):
            with mda.Writer(self.trajname, len(write_sel.atoms)) as W:
                for i, ts in tqdm(enumerate(self.u.trajectory[wf[::self.step]]),
                                  total=len(wf)//(self.step+1),
                                  desc='writing trajectory'):
                    W.write(protein+chol.residues[wl[::self.step][i]].atoms)


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

        u_red = mda.Universe('prot_chol.gro', self.trajname)
        chol_red = u_red.select_atoms('resname CHOL')

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

