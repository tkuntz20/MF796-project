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
        premium = (sumpayoff / M)
        print(f'average is:  {S_avg},  sum of S_path/N+1:  {np.sum(S)/(N+1)}\n')
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
        premium = (sumpayoff / M)
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
            sumpayoff += max(0, (k * S_avg) - S[-1]) * np.exp(-r * T)
        premium = (sumpayoff / M)
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
            sumpayoff += max(0, S[-1] - (k * S_avg)) * np.exp(-r * T)
        premium = (sumpayoff / M)
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
        dt = T / N
        for j in range(M):
            S[0] = S0
            indicator = 0.0
            for i in range(N):
                # indicator should act like a digital option
                if S[i] < b:
                    epsilon = sp.random.randn(1)
                    S[i + 1] = S[i] * (1 + r * dt + sigma * math.sqrt(dt) * epsilon)
                    indicator += 1
                else:
                    S[i + 1] = S[i] * 0
                    indicator += 0
            # S_avg = np.average(S)
            S_avg = np.sum(S)
            # print(f'average:  {S_avg}')
            # print(f'indicator:  {indicator}')
            sumpayoff += max(0, (S_avg / indicator) - K) * np.exp(-r * T)
        premium = (sumpayoff / M)

        return premium

    def bnp_paribas_Asain_put(self, S0, K, T, r, sigma, N, M, b):
        S = sp.random.rand(N + 1)
        sumpayoff = 0.0
        premium = 0.0
        dt = T / N
        for j in range(M):
            S[0] = S0
            indicator = 0.0
            for i in range(N):
                # indicator should act like a digital option
                if S[i] > b:
                    epsilon = sp.random.randn(1)
                    S[i + 1] = S[i] * (1 + r * dt + sigma * math.sqrt(dt) * epsilon)
                    indicator += 1
                else:
                    S[i + 1] = S[i] * 0
                    indicator += 0
            #S_avg = np.average(S)
            S_avg = np.sum(S)
            #print(f'average:  {S_avg}')
            #print(f'indicator:  {indicator}')
            sumpayoff += max(0, K - (S_avg / indicator)) * np.exp(-r * T)
        premium = (sumpayoff / M)

        return premium

    def digital_call(self, S0, K, T, r, sigma, N, M):
        d2 = (np.log(S0/K) + (r - sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
        return np.exp(-r * T) * si.norm.cdf(d2)

    def digital_put(self, S0, K, T, r, sigma, N, M):
        d2 = (np.log(S0/K) + (r - sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
        return np.exp(-r * T) * si.norm.cdf(-d2)

    def d1(self, S0, K, T, r, sigma):
        return (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    def d2(self, S0, K, T, r, sigma):
        return (np.log(S0 / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    def euro_call(self, S0, K, T, r, sigma):
        call = (S0 * si.norm.cdf(self.d1(S0, K, T, r, sigma), 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(self.d2(S0, K, T, r, sigma), 0.0, 1.0))
        return float(call)

    def euro_put(self, S0, K, T, r, sigma):
        put = (K * np.exp(-r * T) * si.norm.cdf(-self.d2(S0, K, T, r, sigma), 0.0, 1.0) - S0 * si.norm.cdf(-self.d1(S0, K, T, r, sigma), 0.0, 1.0))
        return float(put)

class breedenLitzenberger(Options):

    def __init__(self,S, K, T, r, sigma):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma

    def digitalPayoff(self,S,K,type):
        if type == 'C':
            p = 1 if S>=K else 0
        else:
            p = 1 if S<=K else 0
        return p

    def digitalPrice(self,density,S,K,type):
        value = 0
        for i in range(0, len(S)-2):
            value += density[i] * self.digitalPayoff(S[i], K, type) * 0.1
            #print(value)
        return value

    def euroPayoff(self, density, S, K):
        payoff = 0
        for i in range(0, len(S)-2):
            payoff += density[i] * max(0, S[i] - K) * 0.1
        return payoff

    def strikeTransform(self,type, sigma, expiry, delta):
        transform = si.norm.ppf(delta)
        if type == 'P':
            K = 100 * np.exp(0.5 * sigma ** 2 * expiry + sigma * np.sqrt(expiry) * transform)
        else:
            K = 100 * np.exp(0.5 * sigma ** 2 * expiry - sigma * np.sqrt(expiry) * transform)
        return K

    def gammaTransform(self,S,K,T,r,sigma,h):
        value = Options.euro_call(self,S, K, T, r, sigma)
        valuePlus = Options.euro_call(self,S,K+h,T,r,sigma)
        valueDown = Options.euro_call(self,S, K-h, T, r, sigma)
        return (valueDown - 2 * value + valuePlus) / h**2

    def riskNeutral(self,S,K,T,r,sigma,h):
        pdf = []
        for jj, vol in enumerate(sigma):
            p = np.exp(r*T) * self.gammaTransform(S, K[jj], T, r, vol,h)
            pdf.append(p)
        return pdf

    def constantVolatiltiy(self,S,T,r,sigma,h):
        K = np.linspace(70, 130, 150)
        pdf = []
        for i, k in enumerate(K):
            p = np.exp(r*T) * self.gammaTransform(S, k, T, r, sigma,h)
            pdf.append(p)
        return pdf, K

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
    A_call = AO.vanilla_Asain_Call_fixed(100, 100, 1, 0.0, 0.25, 252, 10000)
    print(f'Asain call {A_call}')

    A_put = AO.vanilla_Asain_Put_fixed(100, 100, 1, 0.0, 0.25, 252, 10000)
    print(f'Asain put {A_put}\n')

    s_p = StochasticProcess(-0.1, 0.1, 0.2, 0.031)
    vg = s_p.monteCarloVG_tc(2,504,100)
    plt.plot(vg.T)
    plt.show()

    vg2 = s_p.monteCarloVG_dg(1, 253, 10)
    plt.plot(vg2.T)
    plt.show()

    avg = np.sum(vg2)/252

    #print(vg2)

    print(f'the geometric call values is:       {AO.geometric_Asain_Call(100, 100, 1, 0, 0.25 ,252 ,10000)}')
    print(f'the geometric put values is:        {AO.geometric_Asain_Put(100, 100, 1, 0, 0.25 ,252 ,10000)}')
    print(f'the floating strike call values is: {AO.vanilla_Asain_Call_float(100, 100, 1, 0, 0.25 ,253 ,10000 ,1)}')
    print(f'the floating strike put values is:  {AO.vanilla_Asain_Put_float(100, 100, 1, 0, 0.25, 253, 10000, 1)}')
    print(f'the conditional call values is:     {AO.bnp_paribas_Asain_call(100, 100, 1, 0.0, 0.25 ,252 ,10000, 150)}')
    print(f'the conditional put values is:      {AO.bnp_paribas_Asain_put(100, 100, 1, 0.0, 0.25, 252, 10000, 50)}\n')
    print(f'the digital call values is:         {AO.digital_call(100, 100, 1, 0.0, 0.25, 252, 10000)}')
    print(f'the digital put values is:          {AO.digital_put(100, 100, 1, 0.0, 0.25, 252, 10000)}')
    print(f'the euro call values is:            {AO.euro_call(100, 100, 1, 0.0, 0.25)}')
    print(f'the euro put values is:             {AO.euro_put(100, 100, 1, 0.0, 0.25)}')

    # Volatility table-------------------------------------------------
    expiryStrike = ['10DP', '25DP', '40DP', '50DP', '50DC', '40DC', '25DC', '10DC']
    vols = [[32.25, 28.36], [24.73, 21.78], [23.21, 20.18], [18.24, 16.45], [15.74, 14.62], [24.73, 21.78], [10.70, 11.56],
            [11.48, 10.94]]
    volDictionary = dict(zip(expiryStrike, vols))
    volDF = pd.DataFrame.from_dict(volDictionary, orient='index', columns=['1M', '3M'])
    # print(volDictionary)
    S = 100
    K = 0
    T = 0
    r = 0.0
    sigma = 0

    # part (a)
    BL = breedenLitzenberger(S, K, T, r, sigma)
    table = {}
    for row in volDictionary:
        delta = int(row[:2]) / 100
        type = row[-1]
        oneM = BL.strikeTransform(type, volDictionary[row][0] / 100, 1 / 12, delta)
        threeM = BL.strikeTransform(type, volDictionary[row][1] / 100, 6 / 12, delta)
        table[row] = [oneM, threeM]
    volTable = pd.DataFrame.from_dict(table, orient='index', columns=['1M', '3M'])
    print(f'This is the transformed strike tabel: \n {volTable}')

    # pull in USDTRY vol grid
    USDTRY_grid = pd.read_csv('USDTRY_04282022_grid.csv')
    print(USDTRY_grid)

    # part (b)
    strikeList = np.linspace(75, 115, 500)
    interp1M = np.polyfit(volTable['1M'], volDF['1M'] / 100, 2)
    interp3M = np.polyfit(volTable['3M'], volDF['3M'] / 100, 2)
    oneMvol = np.poly1d(interp1M)(strikeList)
    threeMvol = np.poly1d(interp3M)(strikeList)
    plt.plot(strikeList, oneMvol, color='r', label='1M vol')
    plt.plot(strikeList, threeMvol, color='b', label='3M vol')
    plt.xlabel('Strike Range')
    plt.ylabel('Volatilities')
    plt.title('Strike Against Volatility')
    plt.legend()
    plt.grid(linestyle='--', linewidth=0.75)
    plt.show()

    # part (c)
    pdf1 = BL.riskNeutral(S, strikeList, 1 / 12, r, oneMvol, 0.1)
    pdf2 = BL.riskNeutral(S, strikeList, 6 / 12, r, threeMvol, 0.1)
    plt.plot(strikeList, pdf1, label='1M volatility', linewidth=2, color='r')
    plt.plot(strikeList, pdf2, label='3M volatility', linewidth=2, color='b')
    plt.xlabel('Strike Range')
    plt.ylabel('Density')
    plt.title('Risk-Neutral Densities')
    plt.legend()
    plt.grid(linestyle='--', linewidth=0.75)
    plt.show()

    # part (d)
    cpdf1 = BL.constantVolatiltiy(S, 1 / 12, r, 0.1824, 0.1)
    cpdf2 = BL.constantVolatiltiy(S, 6 / 12, r, 0.1645, 0.1)
    plt.plot(cpdf1[1], cpdf1[0], label='1M volatility', linewidth=2, color='r')
    plt.plot(cpdf2[1], cpdf2[0], label='3M volatility', linewidth=2, color='b')
    plt.xlabel('Strike Range')
    plt.ylabel('Density')
    plt.title('Risk-Neutral Densities(const. vol)')
    plt.legend()
    plt.grid(linestyle='--', linewidth=0.75)
    plt.show()

    # part (e)
    S = np.linspace(75, 115, len(pdf1))
    p1 = BL.digitalPrice(pdf1, S, K=110, type='P')
    p2 = BL.digitalPrice(pdf2, S, K=105, type='C')
    v = (threeMvol + oneMvol) / 2
    eupdf = BL.riskNeutral(100, strikeList, 2 / 12, r, v, 0.1)
    p3 = BL.euroPayoff(eupdf, S, 100)
    print()
    print(f'1M European Digital Put Option with Strike 110:   {p1}')
    print(f'3M European Digital Call Option with Strike 105:  {p2}')
    print(f'2M European Call Option with Strike 100:          {p3}\n')









