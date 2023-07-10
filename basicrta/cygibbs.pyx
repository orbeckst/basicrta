from basicrta.functions import norm_exp
from tqdm import tqdm
import numpy as np

def cygibbs(x, loc, residue):
    ncomp, niter = 100, 10000
    inrates = 10 ** (np.linspace(-3, 1, ncomp))
    mcweights = np.zeros((niter + 1, ncomp))
    mcrates = np.zeros((niter + 1, ncomp))
    mcweights[0], mcrates[0] = inrates / sum(inrates), inrates
    wh, rh = np.ones(ncomp) / [ncomp], np.ones((ncomp, 2)) * [2, 1]  # guess hyperparameters
    weights, rates = [], []
    Ns = np.zeros((niter, ncomp))
    # indicator = np.memmap('indicator', dtype=float, mode='w+', shape=(ncomp, x.shape[0]))
    #indicator = np.zeros((ncomp, x.shape[0]), dtype=float)
    # indicator = np.zeros((x.shape[0], ncomp), dtype=int)
    for j in tqdm(range(niter), desc=f'{residue}-K{ncomp}', position=loc, leave=False):
        tmp = mcweights[j]*norm_exp(x, mcrates[j]).T
        z = (tmp.T/tmp.sum(axis=1)).T
        c = z.cumsum(axis=1)
        uu = np.random.rand(len(c), 1)
        s = (uu < c).argmax(axis=1)
        Ns[j][:] = np.array([len(s[s==i]) for i in range(ncomp)])
        inds = [np.where(s==i)[0] for i in range(ncomp)]
        Ts = np.array([x[inds[i]].sum() for i in range(ncomp)])
        wtmp, rtmp = np.random.dirichlet(wh + Ns[j]), np.random.gamma(rh[:,0]+Ns[j], 1/(rh[:,1]+Ts))
        winds = wtmp.argsort()
        mcweights[j+1], mcrates[j+1] = wtmp[winds], rtmp[winds]


