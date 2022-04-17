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

    def __init__(self, one):
        self.one = one

    def __repr__(self):

        return f'nothing'


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
    A_call = AO.vanilla_Asain_Call(100, 100, 1, 0, 0.25,253,10000)
    print(f'Asain call {A_call}')

    A_put = AO.vanilla_Asain_Put(100, 100, 1, 0, 0.25, 253, 10000)
    print(f'Asain put {A_put}')












