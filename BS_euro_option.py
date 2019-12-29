import numpy as np
from stock_option import stockoption
import scipy.stats as si
import math


class BS_euro_option(stockoption):
    def euro_vanilla_call_incl_div(self,Stockopt):
        d1 = (np.log(Stockopt.S0/Stockopt.K) + (Stockopt.r+ ((Stockopt.sigma**2)*0.5))*Stockopt.T)/(Stockopt.sigma*(np.sqrt(Stockopt.T)))
        d2 = d1-(Stockopt.sigma*(np.sqrt(Stockopt.T)))
        if Stockopt.is_call:
            price = Stockopt.S0*(si.norm.cdf(d1,0.0,1.0))*(np.exp(-Stockopt.div*(Stockopt.T))) - (Stockopt.K*(np.exp(-Stockopt.r*Stockopt.T))*(si.norm.cdf(d2,0.0,1.0)))
        else:
            price = (Stockopt.K*(np.exp(-Stockopt.r*Stockopt.T))*(si.norm.cdf(-d2,0.0,1.0))) - Stockopt.S0*(si.norm.cdf(-d1,0.0,1.0))*(np.exp(-Stockopt.div*(Stockopt.T)))
        return price