import numpy as np
from stock_option import stockoption
import math


class euro_option(stockoption):
    '''
    calculate required preliminary parameters:
    u = factor change of upstate
    d = factor change of downstate
    pu = risk free upstate probability
    pd = risk free downstate probability
    M = number of nodes
    '''
    def __init__(self,stockopt):
        self.M = stockopt.N + 1 
        self.u = math.exp(stockopt.sigma*math.sqrt(stockopt.dt))
        self.d = 1./self.u
        self.pu = (math.exp((stockopt.r-stockopt.div)*stockopt.dt)-self.d)/(self.u-self.d)
        self.pd = 1-self.pu

    def stocktree(self,stockopt):
        stocktree = np.zeros([self.M, self.M])
        for i in range(self.M):
            for j in range(self.M):
                stocktree[j, i] = round(stockopt.S0*(self.u**(i-j))*(self.d**j),4)
        return stocktree

    def option_price(self, stocktree,stockopt):
        option = np.zeros([self.M, self.M])
        if stockopt.is_call:
            option[:,(self.M-1)] = np.maximum(np.zeros(self.M), (stocktree[:, stockopt.N] - stockopt.K))
        else:
            option[:,(self.M-1)] = np.maximum(np.zeros(self.M), (stockopt.K - stocktree[:, stockopt.N]))
        return option

    def optpricetree(self, option,stockopt):
        for i in np.arange(self.M-2, -1, -1):
            for j in range(0, i+1):
                option[j, i] = round(math.exp(-stockopt.r*stockopt.dt) * (self.pu*option[j, i+1]+self.pd*option[j+1, i+1]),4)
        return option

    def begin_tree(self,stockopt):
        stocktree = self.stocktree(stockopt)
        payoff = self.option_price(stocktree,stockopt)
        return self.optpricetree(payoff,stockopt)

    def price(self,stockopt):
        self.stocktree(stockopt)
        payoff = self.begin_tree(stockopt)
        return payoff[0, 0]
