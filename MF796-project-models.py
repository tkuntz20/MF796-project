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


class StochasticProcess(Base):

    def __init__(self, theta, kappa, sigma, r):
        self.theta = theta
        self.kappa = kappa
        self.sigma = sigma
        self.r = r

    def __repr__(self):

        return f'nothing'

    def VG_char_tcBM(self, S0, T, N):                                                       # time changed Varaince Gamma characteristic function
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
        steps = (gvPlus - gvMinus)
        X = np.concatenate((X0, steps), axis=1).cumsum(1)
        return X

class Options(StochasticProcess):

    def __init__(self, theta, kappa, sigma, r):
        self.theta = theta   # drift of the brownian motion
        self.kappa = kappa   # standard deviation of the BM
        self.sigma = sigma   # variance of the BM
        self.r = r

    def __repr__(self):

        return f'nothing'

    def vanilla_Asain_Call_fixed(self, S0, K, T, r, sigma, N, M):
        S = sp.random.rand(N + 1)
        sumpayoff = 0.0
        premium = 0.0
        dt = T / N
        for j in range(M):
            S[0] = S0
            for i in range(N):
                epsilon = sp.random.randn(1)
                S[i + 1] = S[i] * (1 + r * dt + sigma * math.sqrt(dt) * epsilon)
            S_avg = np.average(S)
            sumpayoff += max(0, S_avg - K) * np.exp(-r * T)
        premium = math.exp(-r * T) * (sumpayoff / M)
        return premium

    def vanilla_Asain_Put_fixed(self, S0, K, T, r, sigma, N, M):
        S = sp.random.rand(N + 1)
        sumpayoff = 0.0
        premium = 0.0
        dt = T / N
        for j in range(M):
            S[0] = S0
            for i in range(N):
                epsilon = sp.random.randn(1)
                S[i + 1] = S[i] * (1 + r * dt + sigma * math.sqrt(dt) * epsilon)
            S_avg = np.average(S)
            sumpayoff += max(0, K - S_avg) * np.exp(-r * T)
        premium = math.exp(-r * T) * (sumpayoff / M)
        return premium

    def vanilla_Asain_Call_float(self, S0, K, T, r, sigma, N, M, k):
        S = sp.random.rand(N + 1)
        sumpayoff = 0.0
        premium = 0.0
        dt = T / N
        for j in range(M):
            S[0] = S0
            for i in range(N):
                epsilon = sp.random.randn(1)
                S[i + 1] = S[i] * (1 + r * dt + sigma * math.sqrt(dt) * epsilon)
            S_avg = np.average(S)
            sumpayoff += max(0, (k * S_avg) - S0) * np.exp(-r * T)
        premium = math.exp(-r * T) * (sumpayoff / M)
        return premium

    def vanilla_Asain_Put_float(self, S0, K, T, r, sigma, N, M, k):
        S = sp.random.rand(N + 1)
        sumpayoff = 0.0
        premium = 0.0
        dt = T / N
        for j in range(M):
            S[0] = S0
            for i in range(N):
                epsilon = sp.random.randn(1)
                S[i + 1] = S[i] * (1 + r * dt + sigma * math.sqrt(dt) * epsilon)
            S_avg = np.average(S)
            sumpayoff += max(0, S0 - (k *S_avg)) * np.exp(-r * T)
        premium = math.exp(-r * T) * (sumpayoff / M)
        return premium

    def geometric_Asain_Call(self, S0, K, T, r, sigma, N, M):
        sigmaG = sigma / np.sqrt(3)
        b = 0.5 * (r - 0.5 * (sigmaG ** 2))
        d1 = (np.log(S0 / K) + (b + 0.5 * (sigmaG ** 2)) * T) / (sigmaG * np.sqrt(T))
        d2 = d1 - (sigmaG * np.sqrt(T))
        call = (S0 * np.exp((b - r) * T) * si.norm.cdf(d1)) - K * np.exp(-r * T) * si.norm.cdf(d2)
        return call

    def geometric_Asain_Put(self, S0, K, T, r, sigma, N, M):
        sigmaG = sigma / np.sqrt(3)
        b = 0.5 * (r - 0.5 * (sigmaG ** 2))
        d1 = (np.log(S0 / K) + (b + 0.5 * (sigmaG ** 2)) * T) / (sigmaG * np.sqrt(T))
        d2 = d1 - (sigmaG * np.sqrt(T))
        put = K * np.exp(-r * T) * si.norm.cdf(-d2) - (S0 * np.exp((b - r) * T) * si.norm.cdf(-d1))
        return put

    def bnp_paribas_Asain_call(self, S0, K, T, r, sigma, N, M, b):
        S = sp.random.rand(N + 1)
        sumpayoff = 0.0
        premium = 0.0
        indicator = 0.0
        dt = T / N
        for j in range(M):
            S[0] = S0
            for i in range(N):
                epsilon = sp.random.randn(1)
                S[i + 1] = S[i] * (1 + r * dt + sigma * math.sqrt(dt) * epsilon)
            running_avg = np.average(S)
            if running_avg < b:
                S_avg = 0
                sumpayoff += max(0, S_avg - K) * np.exp(-r * T)
            else:
                S_avg = np.average(S)
                sumpayoff += max(0, S_avg - K) * np.exp(-r * T)
                indicator += 1
        premium = math.exp(-r * T) * (sumpayoff / M)

        return premium * (indicator / M)

    def bnp_paribas_Asain_put(self, S0, K, T, r, sigma, N, M, b):
        S = sp.random.rand(N + 1)
        sumpayoff = 0.0
        premium = 0.0
        indicator = 0.0
        dt = T / N
        for j in range(M):
            S[0] = S0
            for i in range(N):
                epsilon = sp.random.randn(1)
                S[i + 1] = S[i] * (1 + r * dt + sigma * math.sqrt(dt) * epsilon)
            running_avg = np.average(S)
            if running_avg < b:
                S_avg = 0
                sumpayoff += max(0, K - S_avg) * np.exp(-r * T)
            else:
                S_avg = np.average(S)
                sumpayoff += max(0, K - S_avg) * np.exp(-r * T)
                indicator += 1
        premium = math.exp(-r * T) * (sumpayoff / M)

        return premium * (indicator / M)



class Density_Comparison(Options):

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

    AO = Options(-0.1, 0.1, 0.25, 0.031)
    print('this is just a test')
    A_call = AO.vanilla_Asain_Call_fixed(100, 105, 1, 0, 0.25,253,1000)
    print(f'Asain call {A_call}')

    A_put = AO.vanilla_Asain_Put_fixed(100, 105, 1, 0, 0.25, 253, 1000)
    print(f'Asain put {A_put}')

    s_p = StochasticProcess(-0.1, 0.1, 0.2, 0.031)
    vg = s_p.monteCarloVG_tc(2,504,100)
    plt.plot(vg.T)
    plt.show()

    vg2 = s_p.monteCarloVG_dg(1, 253, 10)
    plt.plot(vg2.T)
    plt.show()

    avg = np.sum(vg2)/252

    #print(vg2)

    print(f'the geometric call values is: {AO.geometric_Asain_Call(100, 105, 1, 0, 0.25 ,253 ,10)}')
    print(f'the geometric put values is: {AO.geometric_Asain_Put(100, 105, 1, 0, 0.25 ,253 ,10)}')
    print(f'the floating strike call values is: {AO.vanilla_Asain_Call_float(100, 105, 1, 0, 0.25 ,253 ,10 ,1)}')
    print(f'the conditional call values is: {AO.bnp_paribas_Asain_call(100, 105, 1, 0, 0.25 ,253 ,10, 50)}')
    print(f'the conditional put values is: {AO.bnp_paribas_Asain_put(100, 105, 1, 0, 0.25, 253, 10, 50)}')









