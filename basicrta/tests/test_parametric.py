from basicrta.functions import simulate_hn
from basicrta.functions import newgibbs
import numpy as np
wts = np.array([0.89, 0.098, 0.008, 0.002, 0.00056])
x = simulate_hn(1e5, wts, [4.7, 0.8, 0.2, 0.02, 0.003])
G = newgibbs(x, 'X1', 0, 0.1, ncomp=5, niter=25000, sort=False)
G.run()
G.results
G.results.burnin
