#from basicrta.functions import norm_exp
from tqdm import tqdm
import numpy as np
cimport numpy as cnp
import gc
import cython

cnp.import_array()
DTI = np.int64
DTF = np.float64
ctypedef cnp.int64_t DTI_t
ctypedef cnp.float64_t DTF_t
@cython.wraparound(False)
@cython.boundscheck(False)

cdef norm_exp(cnp.ndarray x, cnp.ndarray rates):
    return np.array([rate*np.exp(-rate*x) for rate in rates])


def cygibbs_o(cnp.ndarray[DTF_t, ndim=1] times, int ncomp, int loc, str residue, niter=100000):
    #cdef cnp.ndarray[np.float64, ndim=1] times
    #DEF timelen = times.shape[0]
    cdef cnp.ndarray[DTF_t, ndim=1] x = np.zeros(times.shape[0], dtype=DTF)
    cdef cnp.ndarray[DTF_t, ndim=2] mcweights = np.zeros((niter + 1, ncomp),dtype=DTF)
    cdef cnp.ndarray[DTF_t, ndim=2] mcrates = np.zeros((niter + 1, ncomp), dtype=DTF)
    cdef cnp.ndarray[DTF_t, ndim=1] wh = np.ones(ncomp, dtype=DTI)/[ncomp] 
    cdef cnp.ndarray[DTI_t, ndim=2] rh = np.ones((ncomp, 2), dtype=DTI)*[2, 1] 
    cdef cnp.ndarray[DTI_t, ndim=2] Ns = np.zeros((niter, ncomp), dtype=DTI)

    cdef cnp.ndarray[DTF_t, ndim=2] tmp  
    cdef cnp.ndarray[DTF_t, ndim=2] z
    cdef list inds

    x[:] = times
    inrates = 10 ** (np.linspace(-3, 1, ncomp))
    mcweights[0], mcrates[0] = inrates / sum(inrates), inrates
    #weights, rates = [], []
    # indicator = np.memmap('indicator', dtype=float, mode='w+', shape=(ncomp, x.shape[0]))
    #indicator = np.zeros((ncomp, x.shape[0]), dtype=float)
    # indicator = np.zeros((x.shape[0], ncomp), dtype=int)

    for j in tqdm(range(niter), desc=f'{residue}-K{ncomp}', position=loc, leave=False):
        tmp = mcweights[j]*norm_exp(x, mcrates[j]).T
        z = tmp.T/tmp.sum(axis=1)
        Ns[j] = z.sum(axis=1)
        #c = z.cumsum(axis=1)
        #uu = np.random.rand(len(c), 1)
        #s = np.array((uu < c).argmax(axis=1), dtype=DTF)
        #Ns[j][:] = np.array([len(s[s==i]) for i in range(ncomp)])
        #inds = [np.where(s==i)[0] for i in range(ncomp)]
        #Ts = np.array([x[inds[i]].sum() for i in range(ncomp)])
        #wtmp, rtmp = np.random.dirichlet(wh + Ns[j]), np.random.gamma(rh[:,0]+Ns[j], 1/(rh[:,1]+Ts))
        #winds = wtmp.argsort()
        #mcweights[j+1], mcrates[j+1] = wtmp[winds], rtmp[winds]
        mcweights[j+1] = np.random.dirichlet(wh + Ns[j])
        mcrates[j+1] = np.random.gamma(rh[:,0]+Ns[j], 1/(rh[:,1]+np.dot(z, x)))
        gc.collect()
    return mcweights, mcrates, Ns

