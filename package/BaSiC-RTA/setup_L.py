import numpy as np
from tqdm import tqdm
from scipy.special import gamma
import math
from numba import njit


#class BALMP()


def simulate_h2(n, params):
    n, [a, alp, bet] = int(n), params
    x = np.zeros(n)
    p = np.random.rand(n)

    x[p < a] = -np.log(np.random.rand(len(p[p < a])))/alp
    x[p > a] = -np.log(np.random.rand(len(p[p > a])))/bet
    x.sort()
    return x


def simulate_hn(n, weights, rates):
    n = int(n)
    x = np.zeros(n)
    p = np.random.rand(n)

    tmpw = np.concatenate(([0], np.cumsum(weights)))
    for i in range(len(weights)):
        x[(p > tmpw[i]) & (p <= tmpw[i+1])] = -np.log(np.random.rand(len(p[(p > tmpw[i]) & (p <= tmpw[i+1])])))/rates[i]
    x.sort()
    return x


def simulate_stretch_k2(n, weights, rates, exp):
    n = int(n)
    x = np.zeros(n)
    p = np.random.rand(n)

    tmpw = np.concatenate(([0], np.cumsum(weights)))

    x[(p > tmpw[0]) & (p <= tmpw[1])] = (-np.log(np.random.rand(len(p[(p > tmpw[0]) & (p <= tmpw[1])])))/rates[0])**(1/exp)
    x[(p > tmpw[1]) & (p <= tmpw[2])] = -np.log(np.random.rand(len(p[(p > tmpw[1]) & (p <= tmpw[2])])))/rates[1]

    x.sort()
    return x


def pdf_norm(x, mu, sigma):
    return np.exp(-0.5*((x-mu)/sigma)**2)/(np.sqrt(2*np.pi)*sigma)


def get_bins(x, ts):
    if isinstance(x, list):
        x = np.asarray(x)
    elif isinstance(x, np.ndarray):
        None
    else:
        raise TypeError('Input should be a list or array')
    return np.arange(0, int(x.max()//ts)+2)*ts


def expand_times(contacts):
    a = contacts
    prots = np.unique(a[:, 0])
    lips = np.unique(a[:, 1])

    restimes = []                                          
    Ns = []                                        
    for i in tqdm(prots, desc='expanding times'):
        liptimes = []
        lipNs = []
        for j in lips:                                     
            tmp = a[(a[:, 0] == i) & (a[:, 1] == j)]
            liptimes.append(np.round(tmp[:, 2], 1))
            lipNs.append(tmp[:, 3])
        restimes.append(liptimes)
        Ns.append(lipNs)
    times = np.asarray(restimes)                           
    Ns = np.asarray(Ns)           

    alltimes = []
    for res in tqdm(range(times.shape[0])):
        restimes = []
        for lip in range(times.shape[1]):
            for i in range(times[res, lip].shape[0]):
                [restimes.append(j) for j in [times[res, lip][i]]*Ns[res, lip][i].astype(int)]
        alltimes.append(restimes)
    return np.asarray(alltimes)


def make_surv(ahist, max=1):
    y = ahist[0][ahist[0] != 0]
    tmpbin = ahist[1][1:]
    t = tmpbin[ahist[0] != 0]
    t = np.insert(t, 0, 0)
    y = np.cumsum(y)
    y = np.insert(y, 0, max)
    y = y/y[-1]
    s = 1-y
    return t, s


def stretch_exp_p(x, bet, lamda):
    return bet*lamda*x**(lamda-1)*np.exp(-bet*x**lamda)


def stretch_exp_s(x, bet, lamda):
    return np.exp(-bet*x**lamda)


def surv(x, weights, rates):
    return np.inner(weights, np.exp(np.outer(x, -rates)))


def surv2(x, w1, r1, r2):
    return w1*np.exp(-x*r1)+(1-w1)*np.exp(x*-r2)


def prob(x, weights, rates):
    return np.inner(weights*rates, np.exp(np.outer(x, -rates)))


def lnL(data, weights, rates):
    return np.sum(np.log(prob(data, weights, rates)))


def make_F(data):
    return lambda vars: -lnL(data, vars[0], vars[1])


def make_jac(x):
    return lambda params: jac(x, params)


def jac(x, params):
    a, alp, bet = params
    w1 = np.sum((alp*np.exp(-alp*x)-bet*np.exp(-bet*x))/prob(x, params))
    l1 = np.sum((a*np.exp(-alp*x)*(1-alp**2))/prob(x, params))
    l2 = np.sum((-(1-a)*np.exp(-bet*x)*(1-bet**2))/prob(x, params))
    tmparr = np.array((w1, l1, l2))
    return np.transpose(tmparr)


def norm_exp(x, rates):
    return np.array([rate*np.exp(-rate*x) for rate in rates])

@njit
def norm_exp_numba(x, rates):
    tmparr = np.zeros((len(x), len(rates)))
    for i, rate in enumerate(rates):
        tmparr[:, i] = rate*np.exp(-rate*x)
    return tmparr


def exp(x, rates):
    return np.asarray([np.exp(-rate*x) for rate in rates])


def w_prior(x, alp, bet):
    return lambda a: np.sqrt(sum((exp(x, alp)-exp(x, bet))**2/(a*exp(x, alp)+(1-a)*exp(x, bet))))


def approx_omega(x, lamda1, lamda2):
    lamdaA, lamda0 = lamda2/(lamda1-1), lamda1-1
    L = len(x)
    return np.log(np.sqrt(2*np.pi/(lamda0+L)))+(L+lamda0)*np.log((lamda0+L)/(lamdaA*lamda0+x.sum()))+\
           np.log((lamda0*lamdaA)**(lamda0+1)/(math.factorial(lamda0)*(lamdaA*lamda0+x.sum())))

