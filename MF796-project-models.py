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

    def geometric_Asain_Call(self, S0, K, T, r, sigma):
        sigmaG = sigma / np.sqrt(3)
        b = 0.5 * (r - 0.5 * (sigmaG ** 2))
        d1 = (np.log(S0 / K) + (b + 0.5 * (sigmaG ** 2)) * T) / (sigmaG * np.sqrt(T))
        d2 = d1 - (sigmaG * np.sqrt(T))
        call = (S0 * np.exp((b - r) * T) * si.norm.cdf(d1)) - K * np.exp(-r * T) * si.norm.cdf(d2)
        return call

    def geometric_Asain_Put(self, S0, K, T, r, sigma):
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

    def digital_payoff(self,S,K,type):
        if type == 'C':
            p = 1 if S>=K else 0
        else:
            p = 1 if S<=K else 0
        return p

    def digital_price(self,density,S,K,type):
        value = 0
        for i in range(0, len(S)-2):
            value += density[i] * self.digital_payoff(S[i], K, type) * 0.1
            #print(value)
        return value

    def euroPayoff(self, density, S, K):
        payoff = 0
        for i in range(0, len(S)-2):
            payoff += density[i] * max(0, S[i] - K) * 0.1
        return payoff

    def strike_transform_euro(self,type, sigma, expiry, delta):
        transform = si.norm.ppf(delta)
        if type == 'P':
            K = 100 * np.exp(0.5 * sigma ** 2 * expiry + sigma * np.sqrt(expiry) * transform)
        else:
            K = 100 * np.exp(0.5 * sigma ** 2 * expiry - sigma * np.sqrt(expiry) * transform)
        return K

    def gamma_transform_euro(self,S,K,T,r,sigma,h):
        value = Options.euro_call(self,S, K, T, r, sigma)
        valuePlus = Options.euro_call(self,S,K+h,T,r,sigma)
        valueDown = Options.euro_call(self,S, K-h, T, r, sigma)
        return (valueDown - 2 * value + valuePlus) / h**2

    def risk_neutral_euro(self,S,K,T,r,sigma,h):
        pdf = []
        for jj, vol in enumerate(sigma):
            p = np.exp(r*T) * self.gamma_transform_euro(S, K[jj], T, r, vol,h)
            pdf.append(p)
        return pdf

    def constant_volatiltiy_euro(self,S,T,r,sigma,h):
        K = np.linspace(60, 150, 100)
        pdf = []
        for i, k in enumerate(K):
            p = np.exp(r*T) * self.gamma_transform_euro(S, k, T, r, sigma,h)
            pdf.append(p)
        return pdf, K

    # asian options equivilant
    def strike_transform_asian(self,type, sigma, expiry, delta):
        transform = si.norm.ppf(delta)
        if type == 'P':
            K = 100 * np.exp(0.5 * sigma ** 2 * expiry + sigma * np.sqrt(expiry) * transform)
        else:
            K = 100 * np.exp(0.5 * sigma ** 2 * expiry - sigma * np.sqrt(expiry) * transform)
        return K

    def gamma_transform_asian(self,S,K,T,r,sigma,h):
        value = Options.geometric_Asain_Call(self, S, K, T, r, sigma)
        valuePlus = Options.geometric_Asain_Call(self, S, K+h, T, r, sigma)
        valueDown = Options.geometric_Asain_Call(self, S, K-h, T, r, sigma)
        return (valueDown - 2 * value + valuePlus) / h**2

    def risk_neutral_asian(self,S,K,T,r,sigma,h):
        pdf = []
        for jj, vol in enumerate(sigma):
            p = np.exp(r*T) * self.gamma_transform_asian(S, K[jj], T, r, vol,h)
            pdf.append(p)
        return pdf

    def constant_volatiltiy_asian(self,S,T,r,sigma,h):
        K = np.linspace(60, 150, 100)
        pdf = []
        for i, k in enumerate(K):
            p = np.exp(r*T) * self.gamma_transform_euro(S, k, T, r, sigma,h)
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

    print(f'the geometric call values is:       {AO.geometric_Asain_Call(100, 100, 1, 0, 0.25)}')
    print(f'the geometric put values is:        {AO.geometric_Asain_Put(100, 100, 1, 0, 0.25)}')
    #print(f'the floating strike call values is: {AO.vanilla_Asain_Call_float(100, 100, 1, 0, 0.25 ,253 ,10000 ,1)}')
    #print(f'the floating strike put values is:  {AO.vanilla_Asain_Put_float(100, 100, 1, 0, 0.25, 253, 10000, 1)}')
    #print(f'the conditional call values is:     {AO.bnp_paribas_Asain_call(100, 100, 1, 0.0, 0.25 ,252 ,10000, 150)}')
    #print(f'the conditional put values is:      {AO.bnp_paribas_Asain_put(100, 100, 1, 0.0, 0.25, 252, 10000, 50)}\n')
    #print(f'the digital call values is:         {AO.digital_call(100, 100, 3/12, 0.0, 0.25, 252, 10000)}')
    #print(f'the digital put values is:          {AO.digital_put(100, 100, 1/12, 0.0, 0.25, 252, 10000)}')
    #print(f'the euro call values is:            {AO.euro_call(100, 100, 2/12, 0.0, 0.25)}')
    #print(f'the euro put values is:             {AO.euro_put(100, 100, 2/12, 0.0, 0.25)}')

    # Volatility table-------------------------------------------------
    # pull in USDTRY vol grid
    USDTRY_grid = pd.read_csv('USDTRY_04282022_grid.csv')
    USDTRY_grid = USDTRY_grid.set_index('ExpiryStrike')
    #print(USDTRY_grid)
    USDTRY_dict = USDTRY_grid.T.to_dict('list')
    USDTRYdf = pd.DataFrame.from_dict(USDTRY_dict, orient='index', columns=['1D', '1W', '2W', '3W', '1M', '2M', '3M', '4M', '5M', '6M', '9M', '1Y', '18M', '2Y', '3Y', '4Y', '5Y', '6Y', '7Y', '10Y', '15Y', '20Y', '25Y', '30Y'])
    print(f'the dictionary is: \n{USDTRY_dict}')
    print(f'the df from dict is: \n{USDTRYdf}')


    S = 100
    K = 100
    T = 0
    r = 0.0
    sigma = 0

    # part (a)
    BL = breedenLitzenberger(S, K, T, r, sigma)

    table1 = {}
    for row in USDTRY_dict:
        delta = int(row[:2]) / 100
        type = row[-1]
        oneD1 = BL.strike_transform_euro(type, USDTRY_dict[row][0] / 100, 1 / 365, delta)
        oneW1 = BL.strike_transform_euro(type, USDTRY_dict[row][1] / 100, 7 / 365, delta)
        twoW1 = BL.strike_transform_euro(type, USDTRY_dict[row][2] / 100, 14 / 365, delta)
        threeW1 = BL.strike_transform_euro(type, USDTRY_dict[row][3] / 100, 21 / 365, delta)
        oneM1 = BL.strike_transform_euro(type, USDTRY_dict[row][4] / 100, 1 / 12, delta)
        twoM1 = BL.strike_transform_euro(type, USDTRY_dict[row][5] / 100, 2 / 12, delta)
        threeM1 = BL.strike_transform_euro(type, USDTRY_dict[row][6] / 100, 3 / 12, delta)
        fourM1 = BL.strike_transform_euro(type, USDTRY_dict[row][7] / 100, 4 / 12, delta)
        fiveM1 = BL.strike_transform_euro(type, USDTRY_dict[row][8] / 100, 5 / 12, delta)
        sixM1 = BL.strike_transform_euro(type, USDTRY_dict[row][9] / 100, 6 / 12, delta)
        nineM1 = BL.strike_transform_euro(type, USDTRY_dict[row][10] / 100, 9 / 12, delta)
        oneY1 = BL.strike_transform_euro(type, USDTRY_dict[row][11] / 100, 1, delta)
        eiteenM1 = BL.strike_transform_euro(type, USDTRY_dict[row][12] / 100, 18 / 12, delta)
        twoY1 = BL.strike_transform_euro(type, USDTRY_dict[row][13] / 100, 2, delta)
        threeY1 = BL.strike_transform_euro(type, USDTRY_dict[row][14] / 100, 3, delta)
        fourY1 = BL.strike_transform_euro(type, USDTRY_dict[row][15] / 100, 4, delta)
        fiveY1 = BL.strike_transform_euro(type, USDTRY_dict[row][16] / 100, 5, delta)
        sixY1 = BL.strike_transform_euro(type, USDTRY_dict[row][17] / 100, 6, delta)
        sevenY1 = BL.strike_transform_euro(type, USDTRY_dict[row][18] / 100, 7, delta)
        tenY1 = BL.strike_transform_euro(type, USDTRY_dict[row][19] / 100, 10, delta)
        fifteenY1 = BL.strike_transform_euro(type, USDTRY_dict[row][20] / 100, 15, delta)
        twentyY1 = BL.strike_transform_euro(type, USDTRY_dict[row][21] / 100, 20, delta)
        twent5Y1 = BL.strike_transform_euro(type, USDTRY_dict[row][22] / 100, 25, delta)
        thirtyY1 = BL.strike_transform_euro(type, USDTRY_dict[row][23] / 100, 30, delta)
        table1[row] = [oneD1, oneW1, twoW1, threeW1, oneM1, twoM1, threeM1, fourM1, fiveM1, sixM1, nineM1, oneY1, eiteenM1, twoY1,
                      threeM1, fourY1, fiveY1, sixY1, sevenY1, tenY1, fifteenY1, twentyY1, twent5Y1, thirtyY1]
    strikeTable1 = pd.DataFrame.from_dict(table1, orient='index',
                                         columns=['1D', '1W', '2W', '3W', '1M', '2M', '3M', '4M', '5M', '6M', '9M',
                                                  '1Y', '18M', '2Y', '3Y', '4Y', '5Y', '6Y', '7Y', '10Y', '15Y', '20Y',
                                                  '25Y', '30Y'])
    print(f'This is the transformed strike tabel: \n {strikeTable1}')

    # part (b)
    strikeList1 = np.linspace(60, 150, 100)
    interp1M1 = np.polyfit(strikeTable1['1Y'], USDTRYdf['1Y'] / 100, 2)
    interp3M1 = np.polyfit(strikeTable1['6M'], USDTRYdf['6M'] / 100, 2)
    oneMvol1 = np.poly1d(interp1M1)(strikeList1)
    threeMvol1 = np.poly1d(interp3M1)(strikeList1)
    plt.plot(strikeList1, oneMvol1, color='r', label='1M vol')
    plt.plot(strikeList1, threeMvol1, color='b', label='3M vol')
    plt.xlabel('Strike Range')
    plt.ylabel('Volatilities')
    plt.title('Strike Against Volatility')
    plt.legend()
    plt.grid(linestyle='--', linewidth=0.75)
    plt.show()

    # part (c)
    pdf1 = BL.risk_neutral_euro(S, strikeList1, 5 / 12, r, oneMvol1, 0.1)
    pdf2 = BL.risk_neutral_euro(S, strikeList1, 0.5, r, threeMvol1, 0.1)
    plt.plot(strikeList1, pdf1, label='1M volatility', linewidth=2, color='r')
    plt.plot(strikeList1, pdf2, label='3M volatility', linewidth=2, color='b')
    plt.xlabel('Strike Range')
    plt.ylabel('Density')
    plt.title('Risk-Neutral Densities')
    plt.legend()
    plt.grid(linestyle='--', linewidth=0.75)
    plt.show()

    # part (d)
    cpdf11 = BL.constant_volatiltiy_euro(S, 5 / 12, r, 0.1, 0.1)
    cpdf21 = BL.constant_volatiltiy_euro(S, 0.5, r, 0.1, 0.1)
    plt.plot(cpdf11[1], cpdf11[0], label='1M volatility', linewidth=2, color='r')
    plt.plot(cpdf21[1], cpdf21[0], label='3M volatility', linewidth=2, color='b')
    plt.xlabel('Strike Range')
    plt.ylabel('Density')
    plt.title('Risk-Neutral Densities(const. vol)')
    plt.legend()
    plt.grid(linestyle='--', linewidth=0.75)
    plt.show()

    # part (e)
    S = np.linspace(60, 150, len(pdf1))
    p1 = BL.digital_price(pdf1, S, K=K, type='P')
    p2 = BL.digital_price(pdf2, S, K=K, type='C')
    v = (threeMvol1 + oneMvol1) / 2
    eupdf = BL.risk_neutral_euro(100, strikeList1, 2 / 12, r, v, 0.1)
    p3 = BL.euroPayoff(eupdf, S, K)
    print()
    print(f'1M European Digital Put Option with Strike {K}:   {p1}')
    print(f'3M European Digital Call Option with Strike {K}:  {p2}')
    print(f'2M European Call Option with Strike 100:          {p3}\n')


    """
    # asian option transform
    table = {}
    for row in USDTRY_dict:
        delta = int(row[:2]) / 100
        type = row[-1]
        oneD = BL.strike_transform_asian(type, USDTRY_dict[row][0] / 100, 1 / 365, delta)
        oneW = BL.strike_transform_asian(type, USDTRY_dict[row][1] / 100, 7 / 365, delta)
        twoW = BL.strike_transform_asian(type, USDTRY_dict[row][2] / 100, 14 / 365, delta)
        threeW = BL.strike_transform_asian(type, USDTRY_dict[row][3] / 100, 21 / 365, delta)
        oneM = BL.strike_transform_asian(type, USDTRY_dict[row][4] / 100, 1 / 12, delta)
        twoM = BL.strike_transform_asian(type, USDTRY_dict[row][5] / 100, 2 / 12, delta)
        threeM = BL.strike_transform_asian(type, USDTRY_dict[row][6] / 100, 3 / 12, delta)
        fourM = BL.strike_transform_asian(type, USDTRY_dict[row][7] / 100, 4 / 12, delta)
        fiveM = BL.strike_transform_asian(type, USDTRY_dict[row][8] / 100, 5 / 12, delta)
        sixM = BL.strike_transform_asian(type, USDTRY_dict[row][9] / 100, 6 / 12, delta)
        nineM = BL.strike_transform_asian(type, USDTRY_dict[row][10] / 100, 9 / 12, delta)
        oneY = BL.strike_transform_asian(type, USDTRY_dict[row][11] / 100, 1, delta)
        eiteenM = BL.strike_transform_asian(type, USDTRY_dict[row][12] / 100, 18 / 12, delta)
        twoY = BL.strike_transform_asian(type, USDTRY_dict[row][13] / 100, 2, delta)
        threeY = BL.strike_transform_asian(type, USDTRY_dict[row][14] / 100, 3, delta)
        fourY = BL.strike_transform_asian(type, USDTRY_dict[row][15] / 100, 4, delta)
        fiveY = BL.strike_transform_asian(type, USDTRY_dict[row][16] / 100, 5, delta)
        sixY = BL.strike_transform_asian(type, USDTRY_dict[row][17] / 100, 6, delta)
        sevenY = BL.strike_transform_asian(type, USDTRY_dict[row][18] / 100, 7, delta)
        tenY = BL.strike_transform_asian(type, USDTRY_dict[row][19] / 100, 10, delta)
        fifteenY = BL.strike_transform_asian(type, USDTRY_dict[row][20] / 100, 15, delta)
        twentyY = BL.strike_transform_asian(type, USDTRY_dict[row][21] / 100, 20, delta)
        twent5Y = BL.strike_transform_asian(type, USDTRY_dict[row][22] / 100, 25, delta)
        thirtyY = BL.strike_transform_asian(type, USDTRY_dict[row][23] / 100, 30, delta)
        table[row] = [oneD, oneW, twoW, threeW, oneM, twoM, threeM, fourM, fiveM, sixM, nineM, oneY, eiteenM, twoY,
                      threeM, fourY, fiveY, sixY, sevenY, tenY, fifteenY, twentyY, twent5Y, thirtyY]
    strikeTable = pd.DataFrame.from_dict(table, orient='index',
                                         columns=['1D', '1W', '2W', '3W', '1M', '2M', '3M', '4M', '5M', '6M', '9M',
                                                  '1Y', '18M', '2Y', '3Y', '4Y', '5Y', '6Y', '7Y', '10Y', '15Y', '20Y',
                                                  '25Y', '30Y'])
    print(f'This is the transformed strike tabel: \n {strikeTable}')

    # part (b)
    strikeList = np.linspace(60, 150, 100)
    interp1M = np.polyfit(strikeTable['1Y'], USDTRYdf['1Y'] / 100, 2)
    interp3M = np.polyfit(strikeTable['6M'], USDTRYdf['6M'] / 100, 2)
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
    pdf1 = BL.risk_neutral_asian(S, strikeList, 1, r, oneMvol, 0.1)
    pdf2 = BL.risk_neutral_asian(S, strikeList, 0.5, r, threeMvol, 0.1)
    plt.plot(strikeList, pdf1, label='1M volatility', linewidth=2, color='r')
    plt.plot(strikeList, pdf2, label='3M volatility', linewidth=2, color='b')
    plt.xlabel('Strike Range')
    plt.ylabel('Density')
    plt.title('Risk-Neutral Densities')
    plt.legend()
    plt.grid(linestyle='--', linewidth=0.75)
    plt.show()

    # part (d)
    cpdf1 = BL.constant_volatiltiy_asian(S, 1, r, 0.1, 0.1)
    cpdf2 = BL.constant_volatiltiy_asian(S, 0.5, r, 0.1, 0.1)
    plt.plot(cpdf1[1], cpdf1[0], label='1M volatility', linewidth=2, color='r')
    plt.plot(cpdf2[1], cpdf2[0], label='3M volatility', linewidth=2, color='b')
    plt.xlabel('Strike Range')
    plt.ylabel('Density')
    plt.title('Risk-Neutral Densities(const. vol)')
    plt.legend()
    plt.grid(linestyle='--', linewidth=0.75)
    plt.show()
    """






