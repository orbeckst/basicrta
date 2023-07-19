#from basicrta.functions import norm_exp
from tqdm import tqdm
import numpy as np
cimport numpy as cnp
import gc
import cython
gc.enable

cnp.import_array()
DTI = np.int64
DTF = np.float64
ctypedef cnp.int64_t DTI_t
ctypedef cnp.float64_t DTF_t
def cygibbs(cnp.ndarray[DTF_t, ndim=1] times, int loc, str residue, niter=100000):
    ncomp = 100 
    cdef cnp.ndarray[DTF_t, ndim=1] x = np.zeros(times.shape[0], dtype=DTF)
    cdef cnp.ndarray[DTF_t, ndim=2] mcweights = np.zeros((niter + 1, ncomp),dtype=DTF)
    cdef cnp.ndarray[DTF_t, ndim=2] mcrates = np.zeros((niter + 1, ncomp), dtype=DTF)
    cdef cnp.ndarray[DTF_t, ndim=1] wh = np.ones(ncomp, dtype=DTI)/[ncomp] 
    cdef cnp.ndarray[DTI_t, ndim=2] rh = np.ones((ncomp, 2), dtype=DTI)*[2, 1] 
    cdef cnp.ndarray[DTI_t, ndim=2] Ns = np.zeros((niter, ncomp), dtype=DTI)

    cdef cnp.ndarray[DTF_t, ndim=1] temp  
    cdef cnp.ndarray[DTF_t, ndim=2] tmp  
    cdef cnp.ndarray[DTF_t, ndim=2] z
    cdef cnp.ndarray[DTF_t, ndim=2] c
    cdef cnp.ndarray[DTF_t, ndim=2] uu
    cdef cnp.ndarray[DTF_t, ndim=1] s
    cdef cnp.ndarray[DTF_t, ndim=1] Ts
    cdef list inds

    x[:] = times
    inrates = 10 ** (np.linspace(-3, 1, ncomp))
    temp = np.exp(-50*np.linspace(0,10, ncomp))
    mcweights[0], mcrates[0] = temp/sum(temp), inrates
    for j in tqdm(range(niter), desc=f'{residue}-K{ncomp}', position=loc, leave=False):
        tmp = mcweights[j]*mcrates[j]*np.exp(np.outer(-mcrates[j], x)).T
        z = (tmp.T/tmp.sum(axis=1)).T
        c = z.cumsum(axis=1)
        uu = np.random.rand(len(c), 1)
        s = np.array((uu < c).argmax(axis=1), dtype=DTF)
        Ns[j][:] = np.array([len(s[s==i]) for i in range(ncomp)])
        inds = [np.where(s==i)[0] for i in range(ncomp)]
        Ts = np.array([x[inds[i]].sum() for i in range(ncomp)])
        #wtmp, rtmp = np.random.dirichlet(wh + Ns[j]), np.random.gamma(rh[:,0]+Ns[j], 1/(rh[:,1]+Ts))
        #winds = wtmp.argsort()
        #mcweights[j+1], mcrates[j+1] = wtmp[winds], rtmp[winds]
        mcweights[j+1], mcrates[j+1] = np.random.dirichlet(wh + Ns[j]), np.random.gamma(rh[:,0]+Ns[j], 1/(rh[:,1]+Ts))
        gc.collect()
    return mcweights, mcrates, Ns


cdef cygibbs_sorted(cnp.ndarray[DTF_t, ndim=1] times, int niter):
    ncomp = 100
    cdef cnp.ndarray[DTF_t, ndim=1] x = np.zeros(times.shape[0], dtype=DTF)
    cdef cnp.ndarray[DTF_t, ndim=2] mcweights = np.zeros((niter + 1, ncomp),dtype=DTF)
    cdef cnp.ndarray[DTF_t, ndim=2] mcrates = np.zeros((niter + 1, ncomp), dtype=DTF)
    cdef cnp.ndarray[DTF_t, ndim=1] wh = np.ones(ncomp, dtype=DTI)/[ncomp] 
    cdef cnp.ndarray[DTI_t, ndim=2] rh = np.ones((ncomp, 2), dtype=DTI)*[2, 1] 
    cdef cnp.ndarray[DTI_t, ndim=2] Ns = np.zeros((niter, ncomp), dtype=DTI)

    cdef cnp.ndarray[DTF_t, ndim=1] temp  
    cdef cnp.ndarray[DTF_t, ndim=2] tmp  
    cdef cnp.ndarray[DTF_t, ndim=2] z
    cdef cnp.ndarray[DTF_t, ndim=2] c
    cdef cnp.ndarray[DTF_t, ndim=2] uu
    cdef cnp.ndarray[DTF_t, ndim=1] s
    cdef cnp.ndarray[DTF_t, ndim=1] Ts
    cdef list inds

    x[:] = times
    inrates = 10 ** (np.linspace(-3, 1, ncomp))
    temp = np.exp(-50*np.linspace(0,10, ncomp))
    mcweights[0], mcrates[0] = temp/sum(temp), inrates
    for j in tqdm(range(niter), desc=f'K{ncomp}', position=loc, leave=False):
        tmp = mcweights[j]*mcrates[j]*np.exp(np.outer(-mcrates[j], x)).T
        z = (tmp.T/tmp.sum(axis=1)).T
        c = z.cumsum(axis=1)
        uu = np.random.rand(len(c), 1)
        s = np.array((uu < c).argmax(axis=1), dtype=DTF)
        Ns[j][:] = np.array([len(s[s==i]) for i in range(ncomp)])
        inds = [np.where(s==i)[0] for i in range(ncomp)]
        Ts = np.array([x[inds[i]].sum() for i in range(ncomp)])
        wtmp, rtmp = np.random.dirichlet(wh + Ns[j]), np.random.gamma(rh[:,0]+Ns[j], 1/(rh[:,1]+Ts))
        winds = wtmp.argsort()[::-1]
        mcweights[j+1], mcrates[j+1] = wtmp[winds], rtmp[winds]
        gc.collect()
    return mcweights, mcrates, Ns
