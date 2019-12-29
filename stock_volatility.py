from arch.unitroot import PhillipsPerron
import statsmodels.tsa.api as smt
import os
import numpy as np 
import pandas as pd
from statsmodels.tsa.stattools import adfuller,kpss
import matplotlib.pyplot as plt
import pandas_datareader.data as pdr
import yfinance as yf
import arch
plt.style.use('fivethirtyeight')
#os.chdir(r"C:\Users\mehul\Pricing Models")

class stock_volatility(object):

#start = date from when you want to analyse stocks, "yyyy-mm-dd"
#end = date of final stock analysis (likely current date), "yyyy-mm-dd"
#tk = ticker label
    
    def __init__(self, tk, start, end):
        self.tk = tk
        self.start = start
        self.end = end
        all_data = pdr.get_data_yahoo(self.tk, start=self.start, end=self.end)
        self.stock_data = pd.DataFrame(all_data['Adj Close'], columns=["Adj Close"])
        self.stock_data["Log"] = np.log(self.stock_data)-np.log(self.stock_data.shift(1))
    
    def stationarity_test(self,stock_data,column):
        rolmean = self.stock_data.rolling(30).mean()
        rolstd = self.stock_data.rolling(30).std()
        plt.plot(self.stock_data,color='blue',label='Original')
        plt.plot(rolmean,color='red',label='Rolling Mean')
        plt.plot(rolstd,color='black',label='Rolling Std')
        plt.legend(loc='best')
        plt.title("Rolling Mean & Standard Deviation")
        plt.show()
        # Perform Dickey Fuller Test 
        print("Results of Dickey-Fuller Test")
        self.stock_data.dropna(inplace=True)
        dftest = adfuller(self.stock_data[column])
        print(dftest)
        dfoutput = pd.Series(dftest[0:4],index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
        for key,Value in dftest[4].items():
            dfoutput['Critical Value(%s)'%key] = Value
        print(dfoutput)
        print ('Results of KPSS Test:')
        print("------------------------------------------------------------------------------")
        kpsstest = kpss(self.stock_data[column], regression='c')
        kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
        for key,value in kpsstest[3].items():
            kpss_output['Critical Value (%s)'%key] = value
        print (kpss_output)
        print("------------------------------------------------------------------------------")
        print ('Results of Phillips-Perron Test:')
        pptest = PhillipsPerron(self.stock_data[column])
        print(pptest)
        print("------------------------------------------------------------------------------")
        
    def lags(self,column):
        from statsmodels.tsa.stattools import acf,pacf
        lag_acf = acf(self.stock_data[column],nlags=40,fft=False)
        lag_pacf = pacf(self.stock_data[column],nlags=40)
        # Plotting ACF and PACF plots
        plt.figure(figsize=(20,8))
        plt.subplot(121)
        plt.plot(lag_acf)
        plt.axhline(y=0,linestyle='--',color='gray')
        plt.axhline(y=-1.96/np.sqrt(len(self.stock_data[column])),linestyle='--',color='gray')
        plt.axhline(y=1.96/np.sqrt(len(self.stock_data[column])),linestyle='--',color='gray')
        plt.title("Autocorrelation Plot")
        plt.subplot(122)
        plt.plot(lag_pacf)
        plt.axhline(y=0,linestyle='--',color='gray')
        plt.axhline(y=-1.96/np.sqrt(len(self.stock_data[column])),linestyle='--',color='gray')
        plt.axhline(y=1.96/np.sqrt(len(self.stock_data[column])),linestyle='--',color='gray')
        plt.title("Partial Autocorrelation Plot")
        plt.show()
    
    def auto_graphics(self,column):
        fig,ax = plt.subplots(figsize=(14,7))
        acf = smt.graphics.plot_acf(self.stock_data[column], lags=40 , alpha=0.05,ax=ax)
        plt.title("Autocorrelation Plot")
        plt.show()
        
    def partial_graphics(self,column):
        fig,ax = plt.subplots(figsize=(14,7))
        acf = smt.graphics.plot_pacf(self.stock_data[column], lags=40 , alpha=0.05,ax=ax)
        plt.title("Partial Autocorrelation Plot")
        plt.show()
    
    def mean_sigma(self,column):
        st = self.stock_data[column].dropna().ewm(span=252).std()
        sigma = st.iloc[-1]
        return sigma

    def garch_sigma(self,column):
        model = arch.arch_model(self.stock_data[column].dropna(), mean='Zero', vol='GARCH', p=1, q=1)
        model_fit = model.fit()
        forecast = model_fit.forecast(horizon=1)
        var = forecast.variance.iloc[-1]
        sigma = float(np.sqrt(var))
        return sigma