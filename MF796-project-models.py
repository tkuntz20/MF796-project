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

    def print_risk_neutral_density(self, pdf1, type1, pdf2, type2, strike_lst, expiry):
        plt.plot(strike_lst, pdf1, label=f'{expiry} volatility {type1}', linewidth=2, color='y')
        plt.plot(strike_lst, pdf2, label=f'{expiry} volatility {type2}', linewidth=2, color='b')
        plt.xlabel('Strike Range')
        plt.ylabel('Density')
        plt.title('Risk-Neutral Market Implied Densities')
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
            K = 5.48 * np.exp(0.5 * sigma ** 2 * expiry + sigma * np.sqrt(expiry) * transform)
        else:
            K = 5.48 * np.exp(0.5 * sigma ** 2 * expiry - sigma * np.sqrt(expiry) * transform)
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
        K = np.linspace(4, 8, 100)
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
            table[row] = [one, two, three, four, five, six, seven, eight, nine, ten]
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
            K = 5.48 * np.exp(0.5 * sigma ** 2 * expiry + sigma * np.sqrt(expiry) * transform)
        else:
            K = 5.48 * np.exp(0.5 * sigma ** 2 * expiry - sigma * np.sqrt(expiry) * transform)
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
        K = np.linspace(4, 8, 100)
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
            table[row] = [one, two, three, four, five, six, seven, eight, nine, ten]
        strike_table = pd.DataFrame.from_dict(table, orient='index',columns=expirylst)
        return strike_table

class Density_Comparison(Options):

    def __init__(self, one):
        self.one = one

    def __repr__(self):

        return f'nothing'

    def convert_strike_grid(self):


        return

class Back_Test(Density_Comparison):

    def __init__(self, one):
        self.one = one

    def __repr__(self):

        return f'nothing'



if __name__ == '__main__':      # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    Tenorlst = ['1D', '1W', '2W', '3W', '1M', '2M', '3M', '4M', '5M', '6M', '9M','1Y', '18M', '2Y', '3Y', '4Y', '5Y', '6Y', '7Y', '10Y', '15Y', '20Y','25Y', '30Y']
    expirylst = ['1D', '1W', '2W', '3W', '1M', '2M', '3M', '4M', '5M', '6M']

    theta = 0.0
    sigma = 0.3106
    kappa = sigma**2
    S0 = 5.48
    K = 5.58
    T = 1
    r = 0.0
    k = 1
    N = 252
    M = 10000
    walk = 'dg'
    b = K/2



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
    base = Base(1)
    dict, df = base.delta_options_grid(USDTRY_grid,'ExpiryStrike', Tenorlst)
    #print(f'\n the dictionary from base: \n{dict}')
    #print(f'\n the df from dict from base is: \n{df}')

    S = 5.48
    K = 5.58
    T = 0
    r = 0.0
    sigma = 0

    # part (a)
    BL = Breeden_Litzenberger_Euro(S, K, T, r, sigma)
    euro_ST = BL.build_strike_table_euro(dict, expirylst)
    #print(f'The Asian transformed strike tabel: \n {euro_ST}')

    # part (b)
    strikeList = np.linspace(4, 8, 100)
    interp1M_euro = np.polyfit(euro_ST['3M'], df['3M'] / 100, 2)
    interp3M_euro = np.polyfit(euro_ST['1W'], df['1W'] / 100, 2)
    oneMvol1 = np.poly1d(interp1M_euro)(strikeList)
    threeMvol1 = np.poly1d(interp3M_euro)(strikeList)

    # part (c)
    pdf1_euro = BL.risk_neutral_euro(S, strikeList, 3/12, r, oneMvol1, 0.1)
    pdf2_euro = BL.risk_neutral_euro(S, strikeList, 1/12, r, threeMvol1, 0.1)

    # part (d)
    cpdf1_euro = BL.constant_volatiltiy_euro(S, 3/12, r, 0.1, 0.1)
    cpdf2_euro = BL.constant_volatiltiy_euro(S, 1/12, r, 0.1, 0.1)

    # asian option transform
    BL_asain = Breeden_Litzenberger_Asian(S, K, T, r, sigma)
    asain_ST = BL_asain.build_strike_table_asian(dict, expirylst)
    #print(f'The Asian transformed strike tabel: \n {asain_ST}')

    # part (b)
    st_asian = np.linspace(4, 8, 100)
    asian_3m = np.polyfit(asain_ST['3M'], df['3M'] / 100, 2)
    asian_6m = np.polyfit(asain_ST['1W'], df['1W'] / 100, 2)
    a3M_vol = np.poly1d(asian_3m)(st_asian)
    a6M_vol = np.poly1d(asian_6m)(st_asian)
    base.print_strike_check(a3M_vol, 'asian', a6M_vol, 'asian', st_asian, '1W')

    # part (c)
    pdf1_asian = BL_asain.risk_neutral_asian(S, st_asian, 3/12, r, a3M_vol, 0.1)
    pdf2_asian = BL_asain.risk_neutral_asian(S, st_asian, 1/12, r, a6M_vol, 0.1)
    base.print_risk_neutral_density(pdf1_asian, 'asian', pdf1_euro, 'euro', st_asian, '1W')

    # part (d)
    cpdf1_asian = BL_asain.constant_volatiltiy_asian(S, 3/12, r, 0.1, 0.1)
    cpdf2_asian = BL_asain.constant_volatiltiy_asian(S, 1/12, r, 0.1, 0.1)
    base.print_risk_neutral_const(cpdf1_asian, 'asian', cpdf1_euro, 'euro', '1W')







