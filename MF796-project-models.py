""""
Group term-project for MF 796 Computational Methods
@Authers: Kibae Kim, Tiaga Schwarz, Karen Mu, Thomas Kuntz
"""

import pandas as pd
import numpy as np
import math
import QuantLib as ql
import scipy as sp
import scipy.stats as si
import statsmodels.api as sm
import seaborn as sns
import sympy as sy
from scipy.optimize import newton
from tabulate import tabulate
from pandas_datareader import data
import matplotlib.pyplot as plt
from datetime import datetime
import calendar


class Base:

    def __init__(self, one):
        self.one = one

    def __repr__(self):

        return f'nothing'


class Stochastic_Process(Base):

    def __init__(self, theta, kappa, sigma, r):
        self.theta = theta
        self.kappa = kappa
        self.sigma = sigma
        self.r = r

    def __repr__(self):

        return f'nothing'

    def characterisic_VG(self, S0, T, N):
        # w is the weight similar to the w in Heston
        w = - np.log(1 - self.theta * self.kappa - self.kappa/2 * self.sigma**2) / self.kappa
        rho = 1/ self.kappa
        gv = si.gamma(rho * T).rvs(N) / rho
        nv = si.norm.rvs(0,1,N)
        vGamma = self.theta * gv + self.sigma * np.sqrt(gv) * nv
        sT = S0 * np.exp((self.r - w) * T + vGamma)
        return sT.reshape((N,1))

    def monteCarloVG_tc(self, T, N, M):                                                                                  # time changed Brownian motion VG process
        dt = T / (N - 1)
        X0 = np.zeros((M,1))
        gv = si.gamma(dt / self.kappa, scale=self.kappa).rvs(size=(M, N-1))
        nv = si.norm.rvs(loc=0, scale=1, size=(M, N-1))
        steps = self.r * dt + self.theta * gv + self.sigma * np.sqrt(gv) * nv
        X = np.concatenate((X0, steps), axis=1).cumsum(1)
        return X

    def monteCarloVG_dg(self, T, N, M):                                                                                 # Monte Carlo simulation via difference of gammas
        dt = T / (N - 1)
        X0 = np.zeros((M, 1))
        mu_p = 0.5 * np.sqrt(self.theta ** 2 + (2 * self.sigma ** 2) / self.kappa) + self.theta / 2
        mu_q = 0.5 * np.sqrt(self.theta ** 2 + (2 * self.sigma ** 2) / self.kappa) - self.theta / 2
        gvPlus = si.gamma(dt / self.kappa, scale=self.kappa*mu_q).rvs(size=(M, N-1))
        gvMinus = si.gamma(dt / self.kappa, scale=self.kappa*mu_p).rvs(size=(M, N-1))
        steps = gvPlus - gvMinus
        X = np.concatenate((X0, steps), axis=1).cumsum(1)
        return X

class Asain_Options(Stochastic_Process):

    def __init__(self, one):
        self.one = one

    def __repr__(self):

        return f'nothing'

    def vanilla_Asain_Call(self, S0, K, T, r, sigma, N, M):
        S = sp.random.rand(N + 1)
        sumpayoff = 0.0
        premium = 0.0
        dt = T / N
        # STEP 2: MAIN SIMULATION LOOP
        for j in range(M):
            S[0] = S0
            # STEP 3: TIME INTEGRATION LOOP
            for i in range(N):
                epsilon = sp.random.randn(1)
                S[i + 1] = S[i] * (1 + r * dt + sigma * math.sqrt(dt) * epsilon)
            # STEP 4: COMPUTE PAYOFF
            S_avg = np.average(S)
            sumpayoff += max(0, S_avg - K) * np.exp(-r * T)
        # STEP 5: COMPUTE DISCOUNTED EXPECTED PAYOFF
        premium = math.exp(-r * T) * (sumpayoff / M)
        return premium

    def vanilla_Asain_Put(self, S0, K, T, r, sigma, N, M):
        S = sp.random.rand(N + 1)
        sumpayoff = 0.0
        premium = 0.0
        dt = T / N
        # STEP 2: MAIN SIMULATION LOOP
        for j in range(M):
            S[0] = S0
            # STEP 3: TIME INTEGRATION LOOP
            for i in range(N):
                epsilon = sp.random.randn(1)
                S[i + 1] = S[i] * (1 + r * dt + sigma * math.sqrt(dt) * epsilon)
            # STEP 4: COMPUTE PAYOFF
            S_avg = np.average(S)
            sumpayoff += max(0, K - S_avg) * np.exp(-r * T)
        # STEP 5: COMPUTE DISCOUNTED EXPECTED PAYOFF
        premium = math.exp(-r * T) * (sumpayoff / M)
        return premium




class Density_Comparison(Asain_Options):

    def __init__(self, one):
        self.one = one

    def __repr__(self):

        return f'nothing'


class Back_Test(Density_Comparison):

    def __init__(self, one):
        self.one = one

    def __repr__(self):

        return f'nothing'














if __name__ == '__main__':      # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    AO = Asain_Options(1)
    print('this is just a test')
    A_call = AO.vanilla_Asain_Call(100, 105, 1, 0, 0.25,253,10000)
    print(f'Asain call {A_call}')

    A_put = AO.vanilla_Asain_Put(100, 105, 1, 0, 0.25, 253, 10000)
    print(f'Asain put {A_put}')

    sp = Stochastic_Process(-0.1, 0.1, 0.2, 0.031)
    vg = sp.monteCarloVG_tc(2,504,10)
    plt.plot(vg.T)
    plt.show()

    vg2 = sp.monteCarloVG_dg(2, 504, 200)
    plt.plot(vg2.T)
    plt.show()










