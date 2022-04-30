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

    def delta_options_grid(self, vol_df, setIndex, columnNames):
        vol_df = vol_df.set_index(setIndex)
        # print()
        dict = vol_df.T.to_dict('list')
        USDTRY_df = pd.DataFrame.from_dict(dict, orient='index',columns=columnNames)
        return dict, USDTRY_df

    def print_risk_neutral_density(self, pdf1, type1, pdf2, type2, strike_lst, expiry1, expiry2):
        plt.plot(strike_lst, pdf1, label=f'{expiry1} volatility {type1}', linewidth=2, color='y')
        plt.plot(strike_lst, pdf2, label=f'{expiry2} volatility {type2}', linewidth=2, color='b')
        plt.xlabel('Strike Range')
        plt.ylabel('Density')
        plt.title(f'Risk-Neutral Densities: ({type1}{expiry1}) vs. ({type2}{expiry2})')
        plt.legend()
        plt.grid(linestyle='--', linewidth=0.75)
        plt.show()
        return

    def print_risk_neutral_const(self, cpdf1, type1, cpdf2, type2, expiry):
        plt.plot(cpdf1[1], cpdf1[0], label=f'{expiry} volatility {type1}', linewidth=2, color='y')
        plt.plot(cpdf2[1], cpdf2[0], label=f'{expiry} volatility {type2}', linewidth=2, color='b')
        plt.xlabel('Strike Range')
        plt.ylabel('Density')
        plt.title('Risk-Neutral Densities(const. vol)')
        plt.legend()
        plt.grid(linestyle='--', linewidth=0.75)
        plt.show()
        return

    def print_strike_check(self, vol1, type1, vol2, type2, strikelst, expiry):
        plt.plot(strikelst, vol1, color='y', label=f'{expiry} vol {type1}')
        plt.plot(strikelst, vol2, color='b', label=f'{expiry} vol {type2}')
        plt.xlabel('Strike Range')
        plt.ylabel('Volatilities')
        plt.title('Strike Against Volatility')
        plt.legend()
        plt.grid(linestyle='--', linewidth=0.75)
        plt.show()
        return

class StochasticProcess(Base):

    def __init__(self, theta, kappa, S0, K, T, r, sigma, N, M):
        self.theta = theta  # drift of the brownian motion
        self.kappa = kappa  # standard deviation of the BM
        self.S0 = S0
        self.K = K
        self.T = T
        self.sigma = sigma  # variance of the BM
        self.r = r
        self.N = N
        self.M = M

    def __repr__(self):
        return f'nothing'

    def brownian_motion(self, S, r, N, dt, sigma):
        for i in range(N):
            epsilon = sp.random.randn(1)
            S[i + 1] = S[i] * (1 + r * dt + sigma * math.sqrt(dt) * epsilon)
        return S

    def VG_char_tcBM(self, S0, T, N):                                                       # time changed Varaince Gamma characteristic function
        # w is the weight similar to the w in Heston
        w = - np.log(1 - self.theta * self.kappa - self.kappa/2 * self.sigma**2) / self.kappa
        rho = 1/ self.kappa
        gv = si.gamma(rho * T).rvs(N) / rho
        nv = si.norm.rvs(0,1,N)
        vGamma = self.theta * gv + self.sigma * np.sqrt(gv) * nv
        sT = S0 * np.exp((self.r - w) * T + vGamma)
        return sT.reshape((N,1))

    def monteCarloVG_tc(self, S0, T, N, M):                                                                                 # time changed Brownian motion VG process
        dt = T / (N - 1)
        X0 = np.zeros((M,1))
        gamma_v = si.gamma(dt / self.kappa, scale=self.kappa).rvs(size=(M, N-1))
        normal_v = si.norm.rvs(loc=0, scale=1, size=(M, N-1))
        steps = self.r * dt + self.theta * gamma_v + self.sigma * np.sqrt(gamma_v) * normal_v
        X = S0 * np.concatenate((X0, steps), axis=1).cumsum(1)
        return X.T + S0

    def monteCarloVG_dg(self, S0, T, N, M):                                                                                 # Monte Carlo simulation via difference of gammas
        dt = T / (N - 1)
        X0 = np.zeros((M, 1))
        mu_p = 0.5 * np.sqrt(self.theta ** 2 + (2 * self.sigma ** 2) / self.kappa) + self.theta / 2
        mu_q = 0.5 * np.sqrt(self.theta ** 2 + (2 * self.sigma ** 2) / self.kappa) - self.theta / 2
        gvPlus = si.gamma(dt / self.kappa, scale=self.kappa*mu_q).rvs(size=(M, N-1))
        gvMinus = si.gamma(dt / self.kappa, scale=self.kappa*mu_p).rvs(size=(M, N-1))
        steps = (gvPlus - gvMinus)
        X = S0 * np.concatenate((X0, steps), axis=1).cumsum(1)
        return X.T + S0

class VarianceGamma(StochasticProcess, Base):

    def __init__(self, theta, kappa, S0, K, T, r, sigma, N, M):
        self.theta = theta  # drift of the brownian motion
        self.kappa = kappa  # standard deviation of the BM
        self.S0 = S0
        self.K = K
        self.T = T
        self.sigma = sigma  # variance of the BM
        self.r = r
        self.N = N
        self.M = M

    def vanilla_Euro_Call(self):
        rho = 1 / self.kappa
        w = - np.log(1 - self.theta * self.kappa - self.kappa / 2 * self.sigma ** 2) / self.kappa
        gamma_v = si.gamma(rho * self.T).rvs(self.N) / rho
        norm_v = si.norm.rvs(0,1,self.N)
        vg_rand = self.theta * gamma_v + self.sigma * np.sqrt(gamma_v) * norm_v
        paths = self.S0 * np.exp((self.r - w) * self.T + vg_rand)
        avg = np.average(paths)
        return np.exp(-self.r * self.T) * sp.mean( np.maximum(paths - self.K, 0) )

    def vanilla_Euro_Put(self):
        rho = 1 / self.kappa
        w = - np.log(1 - self.theta * self.kappa - self.kappa / 2 * self.sigma ** 2) / self.kappa
        gamma_v = si.gamma(rho * self.T).rvs(self.N) / rho
        norm_v = si.norm.rvs(0,1,self.N)
        vg_rand = self.theta * gamma_v + self.sigma * np.sqrt(gamma_v) * norm_v
        paths = self.S0 * np.exp((self.r - w) * self.T + vg_rand)
        avg = np.average(paths)
        return np.exp(-self.r * self.T) * sp.mean( self.K - np.maximum(paths, 0) )

class Options(StochasticProcess, Base):

    def __init__(self, theta, kappa, S0, K, T, r, sigma, N, M):
        self.theta = theta  # drift of the brownian motion
        self.kappa = kappa  # standard deviation of the BM
        self.S0 = S0
        self.K = K
        self.T = T
        self.sigma = sigma  # variance of the BM
        self.r = r
        self.N = N
        self.M = M

    def __repr__(self):

        return f'nothing'

    def vanilla_Asain_Call_fixed(self, S0, K, T, r, sigma, N, M, walk):
        S = sp.random.rand(N + 1)
        sumpayoff = 0.0
        premium = 0.0
        dt = T / N
        if walk == 'tc':
            for j in range(M):
                S = self.monteCarloVG_tc(S0, T, N, 1)
                S_avg = np.average(S)
                sumpayoff += max(0, S_avg - K)
            premium = np.exp(-r * T) * (sumpayoff / M)
            print(f'tc paths')
            return premium
        elif walk == 'dg':
            for j in range(M):
                S = self.monteCarloVG_dg(S0, T, N, 1)
                S_avg = np.average(S)
                sumpayoff += max(0, S_avg - K)
            premium = np.exp(-r * T) * (sumpayoff / M)
            print(f'dg paths')
            return premium
        else:
            for j in range(M):
                S[0] = S0
                S = self.brownian_motion(S, r, N, dt, sigma)
                S_avg = np.average(S)
                sumpayoff += max(0, S_avg - K) * np.exp(-r * T)
            premium = np.exp(-r * T) * (sumpayoff / M)
            print(f'gbm paths')
            return premium

    def vanilla_Asain_Put_fixed(self, S0, K, T, r, sigma, N, M, walk):
        S = sp.random.rand(N + 1)
        sumpayoff = 0.0
        premium = 0.0
        dt = T / N
        if walk == 'tc':
            for j in range(M):
                S = self.monteCarloVG_tc(S0, T, N, 1)
                S_avg = np.average(S)
                sumpayoff += max(0, K - S_avg)
            premium = np.exp(-r * T) * (sumpayoff / M)
            print(f'tc paths')
            return premium
        elif walk == 'dg':
            for j in range(M):
                S = self.monteCarloVG_dg(S0, T, N, 1)
                S_avg = np.average(S)
                sumpayoff += max(0, K - S_avg)
            premium = np.exp(-r * T) * (sumpayoff / M)
            print(f'dg paths')
            return premium
        else:
            for j in range(M):
                S[0] = S0
                S = self.brownian_motion(S, r, N, dt, sigma)
                S_avg = np.average(S)
                sumpayoff += max(0, K - S_avg) * np.exp(-r * T)
            premium = np.exp(-r * T) * (sumpayoff / M)
            print(f'gbm paths')
            return premium

    def vanilla_Asain_Call_float(self, S0, K, T, r, sigma, N, M, k, walk):
        S = sp.random.rand(N + 1)
        sumpayoff = 0.0
        premium = 0.0
        dt = T / N
        if walk == 'tc':
            for j in range(M):
                S = self.monteCarloVG_tc(S0, T, N, 1)
                S_avg = np.average(S)
                sumpayoff += max(0, S[-1] - (k * S_avg))
            premium = np.exp(-r * T) * (sumpayoff / M)
            print(f'tc paths')
            return float(premium)
        elif walk == 'dg':
            for j in range(M):
                S = self.monteCarloVG_dg(S0, T, N, 1)
                S_avg = np.average(S)
                sumpayoff += max(0, S[-1] - (k * S_avg))
            premium = np.exp(-r * T) * (sumpayoff / M)
            print(f'dg paths')
            return float(premium)
        else:
            for j in range(M):
                S[0] = S0
                S = self.brownian_motion(S, r, N, dt, sigma)
                S_avg = np.average(S)
                sumpayoff += max(0, S[-1] - (k * S_avg)) * np.exp(-r * T)
            premium = np.exp(-r * T) * (sumpayoff / M)
            print(f'gbm paths')
            return float(premium)

    def vanilla_Asain_Put_float(self, S0, K, T, r, sigma, N, M, k, walk):
        S = sp.random.rand(N + 1)
        sumpayoff = 0.0
        premium = 0.0
        dt = T / N
        if walk == 'tc':
            for j in range(M):
                S = self.monteCarloVG_tc(S0, T, N, 1)
                S_avg = np.average(S)
                sumpayoff += max(0, (k * S_avg) - S[-1])
            premium = np.exp(-r * T) * (sumpayoff / M)
            print(f'tc paths')
            return float(premium)
        elif walk == 'dg':
            for j in range(M):
                S = self.monteCarloVG_dg(S0, T, N, 1)
                S_avg = np.average(S)
                sumpayoff += max(0, (k * S_avg) - S[-1])
            premium = np.exp(-r * T) * (sumpayoff / M)
            print(f'dg paths')
            return float(premium)
        else:
            for j in range(M):
                S[0] = S0
                S = self.brownian_motion(S, r, N, dt, sigma)
                S_avg = np.average(S)
                sumpayoff += max(0, (k * S_avg) - S[-1]) * np.exp(-r * T)
            premium = np.exp(-r * T) * (sumpayoff / M)
            print(f'gbm paths')
            return float(premium)

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

    def bnp_paribas_test(self, S0, K, T, r, sigma, N, M, b):
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

class Breeden_Litzenberger_Euro(Options):

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
            K = self.K * np.exp(0.5 * sigma ** 2 * expiry + sigma * np.sqrt(expiry) * transform)
        else:
            K = self.K * np.exp(0.5 * sigma ** 2 * expiry - sigma * np.sqrt(expiry) * transform)
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

    def constant_volatiltiy_euro(self,S,strikeList,T,r,sigma,h):
        K = strikeList
        pdf = []
        for i, k in enumerate(K):
            p = np.exp(r*T) * self.gamma_transform_euro(S, k, T, r, sigma,h)
            pdf.append(p)
        return pdf, K

    def build_strike_table_euro(self, dict, expirylst):
        table = {}
        for row in dict:
            delta = int(row[:2]) / 100
            type = row[-1]
            one = self.strike_transform_euro(type, dict[row][0] / 100, 1 / 365, delta)
            two = self.strike_transform_euro(type, dict[row][1] / 100, 7/365, delta)
            three = self.strike_transform_euro(type, dict[row][2] / 100, 14 / 365, delta)
            four = self.strike_transform_euro(type, dict[row][3] / 100, 21 / 365, delta)
            five = self.strike_transform_euro(type, dict[row][4] / 100, 1/12, delta)
            six = self.strike_transform_euro(type, dict[row][5] / 100, 2/12, delta)
            seven = self.strike_transform_euro(type, dict[row][6] / 100, 3/12, delta)
            eight = self.strike_transform_euro(type, dict[row][7] / 100, 4/12, delta)
            nine = self.strike_transform_euro(type, dict[row][8] / 100, 5/12, delta)
            ten = self.strike_transform_euro(type, dict[row][9] / 100, 6/12, delta)
            eleven = self.strike_transform_euro(type, dict[row][10] / 100, 9/12, delta)
            y1 = self.strike_transform_euro(type, dict[row][11] / 100, 1, delta)
            y1_5 = self.strike_transform_euro(type, dict[row][12] / 100, 1.5, delta)
            y2 = self.strike_transform_euro(type, dict[row][13] / 100, 2, delta)
            y3 = self.strike_transform_euro(type, dict[row][14] / 100, 3, delta)
            y4 = self.strike_transform_euro(type, dict[row][15] / 100, 4, delta)
            y5 = self.strike_transform_euro(type, dict[row][16] / 100, 5, delta)
            y6 = self.strike_transform_euro(type, dict[row][17] / 100, 6, delta)
            y7 = self.strike_transform_euro(type, dict[row][18] / 100, 7, delta)
            y10 = self.strike_transform_euro(type, dict[row][19] / 100, 10, delta)
            table[row] = [one, two, three, four, five, six, seven, eight, nine, ten, eleven, y1, y1_5, y2, y3, y4, y5, y6, y7, y10]
        strike_table = pd.DataFrame.from_dict(table, orient='index',columns=expirylst)
        return strike_table

class Breeden_Litzenberger_Asian(Options):

    def __init__(self, S, K, T, r, sigma):
            self.S = S
            self.K = K
            self.T = T
            self.r = r
            self.sigma = sigma

    def strike_transform_asian(self,type, sigma, expiry, delta):
        transform = si.norm.ppf(delta)
        if type == 'P':
            K = self.K * np.exp(0.5 * sigma ** 2 * expiry + sigma * np.sqrt(expiry) * transform)
        else:
            K = self.K * np.exp(0.5 * sigma ** 2 * expiry - sigma * np.sqrt(expiry) * transform)
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

    def constant_volatiltiy_asian(self,S,strikeList,T,r,sigma,h):
        K = strikeList
        pdf = []
        for i, k in enumerate(K):
            p = np.exp(r*T) * self.gamma_transform_asian(S, k, T, r, sigma,h)
            pdf.append(p)
        return pdf, K

    def build_strike_table_asian(self, dict, expirylst):
        table = {}
        for row in dict:
            delta = int(row[:2]) / 100
            type = row[-1]
            one = self.strike_transform_asian(type, dict[row][0] / 100, 1 / 365, delta)
            two = self.strike_transform_asian(type, dict[row][1] / 100, 7/365, delta)
            three = self.strike_transform_asian(type, dict[row][2] / 100, 14 / 365, delta)
            four = self.strike_transform_asian(type, dict[row][3] / 100, 21 / 365, delta)
            five = self.strike_transform_asian(type, dict[row][4] / 100, 1/12, delta)
            six = self.strike_transform_asian(type, dict[row][5] / 100, 2/12, delta)
            seven = self.strike_transform_asian(type, dict[row][6] / 100, 3/12, delta)
            eight = self.strike_transform_asian(type, dict[row][7] / 100, 4/12, delta)
            nine = self.strike_transform_asian(type, dict[row][8] / 100, 5/12, delta)
            ten = self.strike_transform_asian(type, dict[row][9] / 100, 6/12, delta)
            eleven = self.strike_transform_asian(type, dict[row][10] / 100, 9 / 12, delta)
            y1 = self.strike_transform_asian(type, dict[row][11] / 100, 1, delta)
            y1_5 = self.strike_transform_asian(type, dict[row][12] / 100, 1.5, delta)
            y2 = self.strike_transform_asian(type, dict[row][13] / 100, 2, delta)
            y3 = self.strike_transform_asian(type, dict[row][14] / 100, 3, delta)
            y4 = self.strike_transform_asian(type, dict[row][15] / 100, 4, delta)
            y5 = self.strike_transform_asian(type, dict[row][16] / 100, 5, delta)
            y6 = self.strike_transform_asian(type, dict[row][17] / 100, 6, delta)
            y7 = self.strike_transform_asian(type, dict[row][18] / 100, 7, delta)
            y10 = self.strike_transform_asian(type, dict[row][19] / 100, 10, delta)
            table[row] = [one, two, three, four, five, six, seven, eight, nine, ten, eleven, y1, y1_5, y2, y3, y4, y5,
                          y6, y7, y10]
        strike_table = pd.DataFrame.from_dict(table, orient='index',columns=expirylst)
        return strike_table

class Density_Comparison(Breeden_Litzenberger_Asian, Breeden_Litzenberger_Euro):

    def __init__(self, one):
        self.one = one

    def __repr__(self):

        return f'nothing'

    def asain_vs_euro_RN(self, S, strikeList, expiry, r, vol, maturity1, maturity2):
        pdf1 = self.risk_neutral_asian(S, strikeList, expiry, r, vol, 0.1)
        pdf2 = self.risk_neutral_euro(S, strikeList, expiry, r, vol, 0.1)
        self.print_risk_neutral_density(pdf1, 'asian', pdf2, 'euro', strikeList, maturity1,maturity2)
        return

    def maturity_vs_maturity_RN(self, S, strikeList, expiry1, expiry2, r, vol1, vol2, maturity1, maturity2, type):
        if type == 'asian':
            pdf1 = self.risk_neutral_asian(S, strikeList, expiry1, r, vol1, 0.1)
            pdf2 = self.risk_neutral_asian(S, strikeList, expiry2, r, vol2, 0.1)
            self.print_risk_neutral_density(pdf1, 'asian', pdf2, 'asian', strikeList, maturity1, maturity2)
        else:
            pdf1 = self.risk_neutral_euro(S, strikeList, expiry1, r, vol1, 0.1)
            pdf2 = self.risk_neutral_euro(S, strikeList, expiry2, r, vol2, 0.1)
            self.print_risk_neutral_density(pdf1, 'euro', pdf2, 'euro', strikeList, maturity1, maturity2)
        return

    def constant_vs_smile_vol(self, S, strikeList, expiry, r, vol,  type):
        if type == 'asian':
            pdf1 = self.constant_volatiltiy_asian(S, strikeList ,expiry, r, 0.18, 0.1)
            pdf2 = self.risk_neutral_asian(S, strikeList, expiry, r, vol, 0.1)
            plt.plot(pdf1[1], pdf1[0], label=f'{expiry1}y X {type} const. vol', linewidth=2, color='b')
            plt.plot(strikeList, pdf2, label=f'{expiry1}y X {type} vol smile', linewidth=2, color='y')
            plt.xlabel('Strike Range')
            plt.ylabel('Density')
            plt.title(f'Risk-Neutral vs Constant Vol ({expiry1}y X {type})')
            plt.legend()
            plt.grid(linestyle='--', linewidth=0.75)
            plt.show()
        else:
            pdf1 = self.constant_volatiltiy_euro(S, strikeList, expiry, r, 0.18, 0.1)
            pdf2 = self.risk_neutral_euro(S, strikeList, expiry, r, vol, 0.1)
            plt.plot(pdf1[1], pdf1[0], label=f'{expiry}y X {type} const. vol', linewidth=2, color='b')
            plt.plot(strikeList, pdf2, label=f'{expiry}y X {type} vol smile', linewidth=2, color='y')
            plt.xlabel('Strike Range')
            plt.ylabel('Density')
            plt.title(f'Risk-Neutral vs Constant Vol ({expiry1}X{type})')
            plt.legend()
            plt.grid(linestyle='--', linewidth=0.75)
            plt.show()
        return

    def strike_vs_vol(self, maturity1, maturity2, strikeList, df):
        interpolated1 = np.polyfit(euro_ST[maturity1], df[maturity1] / 100, 2)
        interpolated2 = np.polyfit(euro_ST[maturity2], df[maturity2] / 100, 2)
        vol1 = np.poly1d(interpolated1)(strikeList)
        vol2 = np.poly1d(interpolated2)(strikeList)
        return vol1, vol2

class Back_Test(Density_Comparison):

    def __init__(self, one):
        self.one = one

    def __repr__(self):

        return f'nothing'

    def euro_payoff(self, S_T, K, premium, type):
        if type == 'C':
            p = S_T-K if S_T>=K else 0
            p = p - premium
        else:
            p = K-S_T if S_T<=K else 0
            p = p - premium
        return p

    def asian_payoff(self, S_T, Avg, k, K, premium, type, flavor):
        if flavor == 'fixed':
            if type == 'C':
                p = Avg - K if S >= K else 0
                p = p - premium
            else:
                p = K - Avg if S <= K else 0
                p = p - premium
        else:
            if type == 'C':
                p = S_T - k*Avg if S_T >= k*Avg else 0
                p = p - premium
            else:
                p = k*Avg - S if S_T <= k*Avg else 0
                p = p - premium

        return p

    def digital_payoff(self, S, K, premium, type):
        if type == 'C':
            p = 1 if S>=K else 0
            p = p - premium
        else:
            p = 1 if S<=K else 0
            p = p - premium
        return p

    def conditional_asian_payoff(self):

        return

    def walk_forward(self, vanilla_op, exotic_op, start_date, expiry_date):
        vanilla_op   # vanilla option premium
        exotic_op    # some version of the asians premium
        start_date   # starting date
        expiry_date  # expiry of traded options


        # option price at start

        # starting date

        # pull the appropriate number of days into the future of data


        # test the actual payoff( need payoff functions to begin with)

        # roll forward?

        return




if __name__ == '__main__':      # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    Tenorlst = ['1D', '1W', '2W', '3W', '1M', '2M', '3M', '4M', '5M', '6M', '9M',\
                '1Y', '18M', '2Y', '3Y', '4Y', '5Y', '6Y', '7Y', '10Y', '15Y', '20Y','25Y', '30Y']
    expirylst = ['1D', '1W', '2W', '3W', '1M', '2M', '3M', '4M', '5M', '6M', '9M',\
                '1Y', '18M', '2Y', '3Y', '4Y', '5Y', '6Y', '7Y', '10Y']

    theta = 0.0
    sigma = 0.3106
    kappa = sigma**2
    S0 = 3.888
    K = 3.888
    T = 1
    r = 0.0
    k = 1
    N = 252
    M = 100
    walk = 'dg'
    b = K/2
    start_date = '2019-01-02'
    end_date = '2019-09-02'

    # pull in the historical data
    historical_data = pd.read_csv('HIstorical Data.csv')
    #historical_data = historical_data.set_index('date')
    historical_data.index = pd.to_datetime(historical_data['date'])
    dfSlice = historical_data[historical_data.index == '2019-01-02'].values.tolist()[0][0:]
    print(dfSlice)


    AO = Options(theta, kappa, S0, K, T, r, sigma, N, M)
    A_call = AO.vanilla_Asain_Call_fixed(S0, K, T, r, sigma, N, M, walk)
    print(f'Asain call {A_call}')

    A_put = AO.vanilla_Asain_Put_fixed(S0, K, T, r, sigma, N, M, walk)
    print(f'Asain put {A_put}\n')

    """
    s_p = StochasticProcess(0.0, 0.25**2, 100, 100, 1, 0.0, 0.25, 252, 10000)
    vg = s_p.monteCarloVG_tc(100, 1,252,100)
    plt.plot(vg)
    plt.show()
    vg2 = s_p.monteCarloVG_dg(100, 1, 253, 100)
    plt.plot(vg2)
    plt.show()
    """

    print(f'the geometric call values is:       {AO.geometric_Asain_Call(S0, K, T, r, sigma)}')
    print(f'the geometric put values is:        {AO.geometric_Asain_Put(S0, K, T, r, sigma)}')
    print(f'the floating strike call values is: {AO.vanilla_Asain_Call_float(S0, K, T, r, sigma, N, M, k, walk)}')
    print(f'the floating strike put values is:  {AO.vanilla_Asain_Put_float(S0, K, T, r, sigma, N, M, k, walk)}')
    print(f'the conditional call values is:     {AO.bnp_paribas_Asain_call(S0, K, T, r, sigma, N, M, b)}')
    print(f'the conditional put values is:      {AO.bnp_paribas_Asain_put(S0, K, T, r, sigma, N, M, b)}\n')
    print(f'the digital call values is:         {AO.digital_call(S0, K, T, r, sigma, N, M)}')
    print(f'the digital put values is:          {AO.digital_put(S0, K, T, r, sigma, N, M)}')
    print(f'the euro call values is:            {AO.euro_call(S0, K, T, r, sigma)}')
    print(f'the euro put values is:             {AO.euro_put(S0, K, T, r, sigma)}')

    # Volatility table-------------------------------------------------
    # pull in USDTRY vol grid
    USDTRY_grid = pd.read_csv('USDTRY_04282022_grid.csv')
    USDBRL_grid = pd.read_csv('USDBRL_01022019_grid.csv')
    USDARS_grid = pd.read_csv('USDARS_01022019_grid.csv')
    USDAUD_grid = pd.read_csv('USDAUD_01022019_grid.csv')
    USDCAD_grid = pd.read_csv('USDCAD_01022019_grid.csv')
    USDCHF_grid = pd.read_csv('USDCHF_01022019_grid.csv')
    USDDKK_grid = pd.read_csv('USDDKK_01022019_grid.csv')
    USDEUR_grid = pd.read_csv('USDEUR_01022019_grid.csv')
    USDGBP_grid = pd.read_csv('USDGBP_01022019_grid.csv')
    USDHUF_grid = pd.read_csv('USDHUF_01022019_grid.csv')
    USDILS_grid = pd.read_csv('USDILS_01022019_grid.csv')
    USDJPY_grid = pd.read_csv('USDJPY_01022019_grid.csv')
    USDNOK_grid = pd.read_csv('USDNOK_01022019_grid.csv')

    base = Base(1)
    dict, df = base.delta_options_grid(USDBRL_grid,'ExpiryStrike', Tenorlst)

    S = S0
    K = S0
    T = 0.0
    r = 0.01
    sigma = 0.165
    expiry1 = 9/12
    expiry2 = 9/12
    strikeList = np.linspace(K*0.1, K*1.9, 100)
    maturity1 = '9M'
    maturity2 = '9M'

    # part (a)
    BL = Breeden_Litzenberger_Euro(S, K, T, r, sigma)
    euro_ST = BL.build_strike_table_euro(dict, expirylst)
    #print(f'The Asian transformed strike tabel: \n {euro_ST}')

    # part (b)
    DC = Density_Comparison(1)
    vol1, vol2 = DC.strike_vs_vol(maturity1, maturity2,strikeList,df)
    base.print_strike_check(vol1, 'market', vol2, 'market implied', strikeList, maturity1)

    """
    # part (c)
    pdf1_euro = BL.risk_neutral_euro(S, strikeList, expiry1, r, vol1, 0.1)
    pdf2_euro = BL.risk_neutral_euro(S, strikeList, expiry2, r, vol2, 0.1)

    # part (d)
    cpdf1_euro = BL.constant_volatiltiy_euro(S, strikeList, expiry1, r, sigma, 0.1)
    cpdf2_euro = BL.constant_volatiltiy_euro(S, strikeList, expiry2, r, sigma, 0.1)

    # asian option transform
    BL_asain = Breeden_Litzenberger_Asian(S, K, T, r, sigma)
    asain_ST = BL_asain.build_strike_table_asian(dict, expirylst)

    # part (b)
    base.print_strike_check(vol1, 'market', vol2, 'market implied', strikeList, maturity1)

    # part (c)
    pdf1_asian = BL_asain.risk_neutral_asian(S, strikeList, expiry1, r, vol1, 0.1)
    pdf2_asian = BL_asain.risk_neutral_asian(S, strikeList, expiry2, r, vol2, 0.1)
    base.print_risk_neutral_density(pdf2_asian, 'asian', pdf2_euro, 'euro', strikeList, maturity1, maturity2)

    # part (d)
    cpdf1_asian = BL_asain.constant_volatiltiy_asian(S, strikeList, expiry1, r, sigma, 0.1)
    cpdf2_asian = BL_asain.constant_volatiltiy_asian(S, strikeList, expiry2, r, sigma, 0.1)
    base.print_risk_neutral_const(cpdf2_asian, 'asian', cpdf2_euro, 'euro', maturity1)
    """

    # density comps
    asianveuro = DC.asain_vs_euro_RN(S, strikeList, expiry1, r, vol1, maturity1, maturity1)
    mVm = DC.maturity_vs_maturity_RN(S, strikeList, expiry1, expiry2, r, vol1, vol2, maturity1, maturity2, 'euro')
    cVm = DC.constant_vs_smile_vol(S, strikeList, expiry1, r, vol1, 'asian')

    # -------------------start of back testing------------------
    bt = historical_data[historical_data.index >= start_date]
    bt = bt[bt.index <= end_date]
    print(f'         the backtest is:\n {bt}')


    back_test = bt['USDBRL CURNCY']
    S_0 = back_test[back_test.index == start_date].values
    K = S_0
    S_T = back_test[back_test.index == end_date].values
    plt.plot(back_test)
    plt.show()
    avg = np.average(back_test)
    vanill_op = AO.euro_call(S_0, K, 9/12, r, sigma)
    exotic_op = AO.geometric_Asain_Call(S_0, K, 9/12, r, sigma)
    print(f' the net premium from the exotic is: {exotic_op}\n the premium from the vanilla is {vanill_op}')
    BT = Back_Test(1)
    result = BT.asian_payoff(S_0,avg,k,S_T,exotic_op,'call', 'fixed')
    result1 = BT.euro_payoff(S_0, S_0, vanill_op, 'call')
    print(f' the net payoff from the exotic is: {result}\n the payoff from the vanilla is {result1}')








    # --------------------end of back testing-------------------