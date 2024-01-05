from basicrta.functions import simulate_hn
from basicrta.functions import newgibbs
import numpy as np
from scipy.optimize import linear_sum_assignment as lsa

def test_gibbs():
    wts = np.array([0.901, 0.09, 0.009])
    rts = [10, 0.1, 0.001]
    x = simulate_hn(1e5, wts, rts)
    G = newgibbs(x, 'X1', 0, 0.1, ncomp=3, niter=1000, sort=False)
    G.run()

    #for i in range(len(G.results.mcrates)):
    #    tmpsum = np.ones((3, 3), dtype=np.float64)
    #    for ii in range(3):
    #        for jj in range(3):
    #            tmpsum[ii,jj] = abs(G.results.mcrates[i][ii]-rts[jj])

    #    # Hungarian algorithm for minimum cost 
    #    sortinds = lsa(tmpsum)[1]

    #    # Relabel states
    #    G.results.mcweights[i] = G.results.mcweights[i][sortinds] 
    #    G.results.mcrates[i] = G.results.mcrates[i][sortinds]

    #tmp = np.array([np.sort(G.results.rates[:,i]) for i in range(G.results.ncomp)])
    #tmp2 = (tmp.cumsum(axis=1).T/tmp.cumsum(axis=1).T[-1])
    #tmp3 = tmp.T[[np.where((tmp2[:,i]>0.025)&(tmp2[:,i]<0.975))[0] for i in range(G.results.ncomp)][0]]
    #descsort = np.median(G.results.mcrates[1000:], axis=0).argsort()[::-1]
    #ci = np.array([[line[0],line[-1]] for line in tmp3.T])

    #Bools = np.array([(rts[i]>ci[descsort][i,0])&(rts[i]<ci[descsort][i,1]) for i in descsort])

    #assert Bools.all() == True
    assert G.results

def test_simdata():
    wts = np.array([0.901, 0.09, 0.009])
    rts = [10, 0.1, 0.001]
    x = simulate_hn(1e5, wts, rts)
    assert len(np.unique(x))==len(x)


if __name__=="__main__":
    test_parametric()

