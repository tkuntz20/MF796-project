import math
import numpy as np
import pandas as pd
import cmath
import scipy.stats as si
import matplotlib.pyplot as plt
import time
from scipy.optimize import root, minimize
from scipy import interpolate
import warnings
warnings.filterwarnings("ignore")


class base:

    def __init__(self, S, K, T, r, sigma):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma

    def __repr__(self):
        return f'Base Class initial price level is: {self.S}, strike is: {self.K}, expiry is: {self.T}, interest rate is: {self.r}, volatility is: {self.sigma}.'

    def d1(self,S, K, T, r, sigma):
        return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    def d2(self,S, K, T, r, sigma):
        return (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    def euroCall(self, S, K, T, r, sigma):
        call = (S * si.norm.cdf(self.d1(S, K, T, r, sigma), 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(self.d2(S, K, T, r, sigma), 0.0, 1.0))
        return float(call)

    def euroPut(self, S, K, T, r, sigma):
        put = (K * np.exp(-r * T) * si.norm.cdf(-self.d2(S, K, T, r, sigma), 0.0, 1.0) - S * si.norm.cdf(-self.d1(S, K, T, r, sigma), 0.0, 1.0))
        return float(put)

    def discountFactor(self,f,t):
        return 1/(1 + f)**t

class euroGreeks(base):

    def __init__(self, S, K, T, r, sigma):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma

    def delta(self):
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        deltaCall = si.norm.cdf(self.d1(self.S, self.K, self.T, self.r, self.sigma), 0.0, 1.0)
        deltaPut = si.norm.cdf(-self.d1(self.S, self.K, self.T, self.r, self.sigma), 0.0, 1.0)
        return deltaCall, -deltaPut

    def gamma(self):
        return (1 / np.sqrt(2 * np.pi) * np.exp(-self.d1(self.S, self.K, self.T, self.r, self.sigma) ** 2 * 0.5)) / (self.S * self.sigma * np.sqrt(self.T))

    def vega(self):
        return self.S * (1 / np.sqrt(2 * np.pi) * np.exp(-self.d1(self.S, self.K, self.T, self.r, self.sigma) ** 2 * 0.5)) * np.sqrt(self.T)

    def theta(self):
        density = 1 / np.sqrt(2 * np.pi) * np.exp(-self.d1(self.S, self.K, self.T, self.r, self.sigma) ** 2 * 0.5)
        cTheta = (-self.sigma * self.S * density) / (2 * np.sqrt(self.T)) - self.r * self.K * np.exp(-self.r * self.T) * si.norm.cdf(self.d2(self.S, self.K, self.T, self.r, self.sigma), 0.0, 1.0)
        pTheta = (-self.sigma * self.S * density) / (2 * np.sqrt(self.T)) + self.r * self.K * np.exp(-self.r * self.T) * si.norm.cdf(-self.d2(self.S, self.K, self.T, self.r, self.sigma), 0.0, 1.0)
        return cTheta, pTheta

class FastFourierTransforms(euroGreeks):

    def __init__(self, T, lst, S=267.15, r=0.0015, q=0.0177):
        self.S = S
        self.T = T
        self.r = r
        self.q = q  # dividend (in this case q=0)
        self.kappa = lst[0]
        self.theta = lst[1]
        self.sigma = lst[2]
        self.rho = lst[3]
        self.nu = lst[4]

    def __repr__(self):
        return f'FFT Class initial price level is: {self.S}, expiry is: {self.T}, interest rate is: ' \
               f'{self.r},\n dividend is: {self.q}, volatility is: {self.sigma}, nu is: {self.nu}, kappa is: {self.kappa}, rho is: {self.rho}, theta is: {self.theta}'

    def helper(self, n):

        delta = np.zeros(len(n), dtype=complex)
        delta[n == 0] = 1
        return delta

    def CharaceristicHeston(self, u):

        sigma = self.sigma
        nu = self.nu
        kappa = self.kappa
        rho = self.rho
        theta = self.theta
        S = self.S
        r = self.r
        T = self.T
        q = self.q

        i = complex(0, 1)
        Lambda = cmath.sqrt(sigma ** 2 * (u ** 2 + i * u) + (kappa - i * rho * sigma * u) ** 2)
        omega = np.exp(i * u * np.log(S) + i * u * (r - q) * T + kappa * theta * T * (kappa - i * rho * sigma * u) / sigma ** 2) / ((cmath.cosh(Lambda * T / 2) + (kappa - i * rho * sigma * u) / Lambda * cmath.sinh(Lambda * T / 2)) ** (2 * kappa * theta / sigma ** 2))
        phi = omega * np.exp(-(u ** 2 + i * u) * nu / (Lambda / cmath.tanh(Lambda * T / 2) + kappa - i * rho * sigma * u))
        return phi

    def heston(self, alpha, K, Klst, N, B):

        t = time.time()
        tau = B / (2 ** N)
        Lambda = (2 * math.pi / (2 ** N)) / tau
        dx = (np.arange(1, (2 ** N) + 1, dtype=complex) - 1) * tau
        chi = np.log(self.S) - Lambda * (2 ** N) / 2
        dy = chi + (np.arange(1, (2 ** N) + 1, dtype=complex) - 1) * Lambda
        i = complex(0, 1)
        chiDx = np.zeros(len(np.arange(1, (2 ** N) + 1, dtype=complex)), dtype=complex)
        for ff in range(0, (2 ** N)):
            u = dx[ff] - (alpha + 1) * i
            chiDx[ff] = self.CharaceristicHeston(u) / ((alpha + dx[ff] * i) * (alpha + 1 + dx[ff] * i))
        FFT = (tau / 2) * chiDx * np.exp(-i * chi * dx) * (
                    2 - self.helper(np.arange(1, (2 ** N) + 1, dtype=complex) - 1))
        ff = np.fft.fft(FFT)
        mu = np.exp(-alpha * np.array(dy)) / np.pi
        ffTwo = mu * np.array(ff).real
        List = list(chi + (np.cumsum(np.ones(((2 ** N), 1))) - 1) * Lambda)
        Kt = np.exp(np.array(List))
        Kfft = []
        ffT = []
        for gg in range(len(Kt)):
            if (Kt[gg] > 1e-16) & (Kt[gg] < 1e16) & (Kt[gg] != float("inf")) & (Kt[gg] != float("-inf")) & (
                    ffTwo[gg] != float("inf")) & (ffTwo[gg] != float("-inf")) & (ffTwo[gg] is not float("nan")):
                Kfft += [Kt[gg]]
                ffT += [ffTwo[gg]]
        spline = interpolate.splrep(Kfft, np.real(ffT))
        if Klst is not None:
            value = np.exp(-self.r * self.T) * interpolate.splev(Klst, spline).real
        else:
            value = np.exp(-self.r * self.T) * interpolate.splev(K, spline).real

        tt = time.time()
        compTime = tt - t

        return value

    def strikeCalibration(self, size, strikesLst, K):
        x = np.zeros((len(size), len(strikesLst)))
        y = np.zeros((len(size), len(strikesLst)))
        a, b = np.meshgrid(size, strikesLst)

        for gg in range(len(size)):
            for pp in range(len(strikesLst)):
                Heston = self.heston(1, size[gg], strikesLst[pp], K)
                x[gg][pp] = Heston[0]
                y[gg][pp] = 1 / ((Heston[0]) ** 2 * Heston[1])

        return x, y, a, b

class breedenLitzenberger(FastFourierTransforms):

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
        value = base.euroCall(self,S, K, T, r, sigma)
        valuePlus = base.euroCall(self,S,K+h,T,r,sigma)
        valueDown = base.euroCall(self,S, K-h, T, r, sigma)
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

class hestonCalibration(breedenLitzenberger):

    def __init__(self,excel):
        self.excel = excel

    def __repr__(self):
        return f'The imported data frame is:\n  {self.excel}'

    def data(self):
        df = pd.DataFrame(self.excel)
        df['call_mid'] = (df.call_bid + df.call_ask) / 2
        df['put_mid'] = (df.put_bid + df.put_ask) / 2
        callDF = df[['expDays', 'expT', 'K','call_mid', 'call_ask', 'call_bid']]
        putDF =df[['expDays', 'expT', 'K', 'put_mid', 'put_ask', 'put_bid']]
        return df, putDF, callDF

    def arbitrage(self,df,type):
        mid = df.columns[df.columns.str.contains('mid')][0]
        if type == 'c':
            monotonic = any(df[mid].pct_change().dropna() >= 0)
        else:
            monotonic = any(df[mid].pct_change().dropna() <= 0)

        df['delta'] = (df[mid] - df[mid].shift(1)) / (df.K - df.K.shift(1))
        if type == 'c':
            dc = any(df.delta >= 0) or any(df.delta < -1)
        else:
            dc = any(df.delta > 1) or any(df.delta <= 0)

        df['convex'] = df[mid] - 2 * df[mid].shift(1) + df[mid].shift(2)
        convexity = any(df.convex < 0)

        # arb checks
        return pd.Series([monotonic, dc, convexity], index=['Monotonic','Delta','Convexity'])

    def helper2(self, K, Klst, alpha, T, lst):
        FFT = FastFourierTransforms(T, lst)
        value = FFT.heston(alpha, K, Klst, N=9, B=1000)
        return value[0]

    def squareSum(self, data, alpha, lst, weighted=False):
        #print(data)
        options = data.columns[3].split('_')[0]
        opt = 0
        if not weighted:
            for T in data.expT.unique():
                temp = data[data.expT == T]
                Klst = temp.K.values
                values = self.helper2(np.mean(Klst), Klst, alpha, T, lst)
                opt += np.sum((values - temp[options + '_mid'].values)**2)
        else:
            for T in data.expT.unique():
                temp = data[data.expT == T]
                Klst = temp.K.values
                v = 1 / (temp[options + '_ask'] - temp[options + '_bid'])
                v = v.values
                values = self.helper2(np.mean(Klst), Klst, alpha, T, lst)
                opt += v.dot((values - temp[options + '_mid'].values)**2)
        return opt

    def optimizer(self, lst, alpha, calls, puts, weighted=False):
        callValue = self.squareSum(calls, alpha, lst, weighted)
        putVlaue = self.squareSum(puts, alpha, lst, weighted)
        return callValue + putVlaue

    def cb1(self, x):
        global times
        if times % 5 == 0:
            print('{}: {}'.format(times, self.optimizer(x, alpha, calls, puts)))
            print(x)
        times += 1
        return

    def cb2(self, x):
        global times
        print('{}: {}'.format(times, self.optimizer(x, alpha, calls, puts, True)))
        print(x)
        times += 1
        return

class hedgingViaHeston(hestonCalibration):

    def __init__(self, K, S, r, T, h, alpha, lst):
        self.lst = lst
        self.alpha = alpha
        self.K = K
        self.S = S
        self.r = r
        self.T = T
        self.h = h

    def deltaHedge(self):
        value = FastFourierTransforms(self.T, lst,S=self.S).heston(self.alpha,self.K,None,10,1000)
        valuePlus = FastFourierTransforms(self.T, lst,S=self.S+self.h).heston(self.alpha,self.K,None,10,1000)
        valueDown = FastFourierTransforms(self.T, lst,S=self.S-self.h).heston(self.alpha,self.K,None,10,1000)
        delta = (valuePlus-valueDown)/2/self.h
        impliedVol = root(lambda x: base.euroCall(self, self.S,self.K,self.T,self.r,x) - value, 0.01).x
        return delta, impliedVol

    def vegaHedge(self,g):
        value = FastFourierTransforms(self.T, lst, S=self.S).heston(self.alpha, self.K, None, 10, 1000)
        lst1 = self.lst + np.array([0, g, 0, 0, g])
        valuePlus = FastFourierTransforms(self.T, lst1, S=self.S + self.h).heston(self.alpha, self.K, None, 10, 1000)
        lst2 = self.lst - np.array([0, g, 0, 0, g])
        valueDown = FastFourierTransforms(self.T, lst2, S=self.S - self.h).heston(self.alpha, self.K, None, 10, 1000)
        vega = (valuePlus-valueDown)/2/g
        impliedVol = root(lambda x: base.euroCall(self, self.S, self.K, self.T, self.r, x) - value, 0.01).x
        return vega, impliedVol

if __name__ == '__main__':

    # Volatility table-------------------------------------------------
    expiryStrike = ['10DP','25DP','40DP','50DP','40DC','25DC','10DC']
    vols = [[32.25,28.36],[24.73,21.78],[20.21,18.18],[18.24,16.45],[15.74,14.62],[13.70,12.56],[11.48,10.94]]
    volDictionary = dict(zip(expiryStrike, vols))
    volDF = pd.DataFrame.from_dict(volDictionary,orient='index',columns=['1M','3M'])
    #print(volDictionary)
    S = 100
    K = 0
    T = 0
    r = 0.0
    sigma = 0


    # part (a)
    BL = breedenLitzenberger(S, K, T, r, sigma)
    table = {}
    for row in volDictionary:
        delta = int(row[:2])/100
        type = row[-1]
        oneM = BL.strikeTransform(type,volDictionary[row][0]/100,1/12,delta)
        threeM = BL.strikeTransform(type,volDictionary[row][1]/100,3/12,delta)
        table[row] = [oneM,threeM]
    volTable = pd.DataFrame.from_dict(table,orient='index',columns=['1M','3M'])
    print(volTable)

    # part (b)
    strikeList = np.linspace(75, 110, 100)
    interp1M = np.polyfit(volTable['1M'], volDF['1M']/100,2)
    interp3M = np.polyfit(volTable['3M'], volDF['3M']/100,2)
    oneMvol = np.poly1d(interp1M)(strikeList)
    threeMvol = np.poly1d(interp3M)(strikeList)
    plt.plot(strikeList,oneMvol,color='r',label='1M vol')
    plt.plot(strikeList,threeMvol,color='b',label='3M vol')
    plt.xlabel('Strike Range')
    plt.ylabel('Volatilities')
    plt.title('Strike Against Volatility')
    plt.legend()
    plt.grid(linestyle='--', linewidth=0.75)
    plt.show()

    # part (c)
    pdf1 = BL.riskNeutral(S,strikeList,1/12,r,oneMvol,0.1)
    pdf2 = BL.riskNeutral(S,strikeList,3/12,r,threeMvol,0.1)
    plt.plot(strikeList, pdf1, label='1M volatility',linewidth=2,color='r')
    plt.plot(strikeList, pdf2, label='3M volatility',linewidth=2,color='b')
    plt.xlabel('Strike Range')
    plt.ylabel('Density')
    plt.title('Risk-Neutral Densities')
    plt.legend()
    plt.grid(linestyle='--', linewidth=0.75)
    plt.show()

    # part (d)
    cpdf1 = BL.constantVolatiltiy(S, 1/12, r, 0.1824, 0.1)
    cpdf2 = BL.constantVolatiltiy(S, 3/12, r, 0.1645, 0.1)
    plt.plot(cpdf1[1], cpdf1[0], label='1M volatility',linewidth=2,color='r')
    plt.plot(cpdf2[1], cpdf2[0], label='3M volatility',linewidth=2,color='b')
    plt.xlabel('Strike Range')
    plt.ylabel('Density')
    plt.title('Risk-Neutral Densities(const. vol)')
    plt.legend()
    plt.grid(linestyle='--', linewidth=0.75)
    plt.show()

    # part (e)
    S = np.linspace(75, 112.5, len(pdf1))
    p1 = BL.digitalPrice(pdf1, S, K=110,type='P')
    p2 = BL.digitalPrice(pdf2, S,K=105,type='C')
    v = (threeMvol+oneMvol)/2
    eupdf = BL.riskNeutral(100,strikeList,2/12,r,v,0.1)
    p3 = BL.euroPayoff(eupdf,S,100)
    print()
    print(f'1M European Digital Put Option with Strike 110:   {p1}')
    print(f'3M European Digital Call Option with Strike 105:  {p2}')
    print(f'2M European Call Option with Strike 100:          {p3}\n')


    # problem 2
    excel = pd.read_excel(r'C:\Users\kuntz\My Drive\Quant Stuff\MF 796 Computational Methods\MF796-repository\MF796-Computational-Methods\mf796-hw3-opt-data.xlsx')
    HC = hestonCalibration(excel)
    print(repr(HC))
    whole, puts, calls = HC.data()

    # part a
    callArb = calls.groupby('expDays').apply(HC.arbitrage, type = 'c')
    putArb = puts.groupby('expDays').apply(HC.arbitrage, type='p')
    print(f'\n   Arbitrage Checks for the Input Data')
    print(f'                Calls\n  {callArb}')
    print(f'                Puts\n   {putArb}\n')

    # part b
    sigma = 0.2
    alpha = 1.5
    nu = 0.2
    kappa = 0.5
    rho = 0.0
    theta = 0.2
    lst = [kappa, theta, sigma, rho, nu]

    lower = [0.01, 0.01, 0.0, -1, 0.0]
    upper = [2.5, 1, 1, 0.5, 0.5]
    bounds = tuple(zip(lower, upper))
    print(f'------Bounds and Initial Guesses------')
    print(f'Lower bound {lower}')
    print(f'Guess       {lst}')
    print(f'Upper bound {upper}')
    print()
    times = 1
    param = (alpha, calls, puts, False)
    minValues = minimize(HC.optimizer, np.array(lst), args=param, method='SLSQP', bounds=bounds, callback=HC.cb1)
    print('---------Minimized Outputs---------')
    print(f'success(True/False) {minValues.success}')
    print(f' kappa | theta | sigma | rho | nu')
    print(f' {minValues.x[0]}  |  {minValues.x[1]}  | {minValues.x[2]}   | {minValues.x[3]} |{minValues.x[4]}')
    print(f'minimized value  {minValues.fun}\n')

    # part c & d
    sigma = 1
    alpha = 1.5
    nu = 0.034
    kappa = 1.76
    rho = -0.81
    theta = 0.07
    lst1 = [kappa, theta, sigma, rho, nu]

    lower1 = [0.01, 0.01, 0.0, -1, 0.0]
    upper1 = [2.5, 1, 1, 0.5, 0.5]
    bounds1 = tuple(zip(lower, upper))
    print(f'-----Bounds and Initial Guesses-----')
    print(f'Lower bound {lower1}')
    print(f'Guess       {lst1}')
    print(f'Upper bound {upper1}\n')
    print()
    param1 = (alpha, calls, puts, True)
    minValues1 = minimize(HC.optimizer, np.array(lst1), args=param1, method='SLSQP', bounds=bounds1, callback=HC.cb2)
    print('---------Minimized Outputs---------')
    print(f'success(True/False) {minValues1.success}')
    print(f'  kappa |  theta | sigma |  rho   | nu')
    print(f' {minValues1.x[0]} | {minValues1.x[1]} | {minValues1.x[2]}  | {minValues1.x[3]}| {minValues1.x[4]}')
    print(f'minimized value  {minValues1.fun}\n')


    # problem 3
    # part a & b
    K = 275
    S = 267.15
    r = 0.015
    T = 0.25
    h =0.01
    alpha = 1.5
    lst = np.array([3.51,0.052,1.17,-0.77,0.034])
    HVH = hedgingViaHeston(K,S,r,T,h,alpha,lst)
    dh = HVH.deltaHedge()
    vh = HVH.vegaHedge(0.01)
    delta = euroGreeks(S, K, T, r, float(dh[1])).delta()
    print(f'Heston delta: {dh[0]}  implied vol: {float(dh[1])}')
    print(f'Black delta:  {delta[0]}\n')
    b = base(S, K, T, r, float(dh[1]))
    callv = b.euroCall(S, K, T, r, float(dh[1]))
    print(f'euro call {callv}')


    vega = euroGreeks(S,K,T,r,float(vh[1])).vega()
    print(f'Heston vega:  {vh[0]}  implied vol: {float(vh[1])}')
    print(f'Black vega:   {vega}')