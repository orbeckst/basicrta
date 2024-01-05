from basicrta.functions import simulate_hn
from basicrta.functions import newgibbs
import numpy as np

def test_parametric():
    wts = np.array([0.89, 0.098, 0.008, 0.002, 0.00056])
    wts = wts/wts.sum()
    x = simulate_hn(5e4, wts, [4.7, 0.8, 0.2, 0.02, 0.003])
    G = newgibbs(x, 'X1', 0, 0.1, ncomp=5, niter=20000, sort=False)
    G.run()
    tmp = np.array([np.sort(G.results.weights[:,i]) for i in range(G.results.ncomp)])
    tmp2 = (tmp.cumsum(axis=1).T/tmp.cumsum(axis=1).T[-1])
    tmp3 = tmp.T[[np.where((tmp2[:,i]>0.025)&(tmp2[:,i]<0.975))[0] for i in range(G.results.ncomp)][0]]
    descsort = G.results.mcweights.mean(axis=0).argsort()[::-1]
    ci = np.array([[line[0],line[-1]] for line in tmp3.T])


    Bools = np.array([(wts[i]>ci[descsort][i,0])&(wts[i]<ci[descsort][i,1]) for i in descsort])

    assert Bools.all() == True

if __name__=="__main__":
    test_parametric()
