import numpy as np
from stock_option import stockoption
import scipy.stats as si
import math


class Greeks(stockoption):
    def __init__(self,stockopt):
        pass
    
    def delta(self,Stockopt):
        self.d1 = (np.log(Stockopt.S0/Stockopt.K) + (Stockopt.r+ ((Stockopt.sigma**2)*0.5))*Stockopt.T)/(Stockopt.sigma*(np.sqrt(Stockopt.T)))
        if Stockopt.is_call:
            self.delta = np.exp(-Stockopt.div*(Stockopt.T))*(si.norm.cdf(self.d1))
        else:
            self.delta = np.exp(-Stockopt.div*(Stockopt.T))*((si.norm.cdf(self.d1))-1)
        return self.delta
        
    def vega(self,Stockopt):
        self.d1 = (np.log(Stockopt.S0/Stockopt.K) + (Stockopt.r+ ((Stockopt.sigma**2)*0.5))*Stockopt.T)/(Stockopt.sigma*(np.sqrt(Stockopt.T)))
        #self.N_dash = (np.exp(-(self.d1**2)/2))/np.sqrt(2*math.pi)
        self.vega = Stockopt.S0*(np.exp(-Stockopt.div*(Stockopt.T)))*np.sqrt(Stockopt.T)*si.norm.pdf(self.d1)
        return self.vega
    
    def Rho(self,Stockopt):
        self.d1 = (np.log(Stockopt.S0/Stockopt.K) + (Stockopt.r+ ((Stockopt.sigma**2)*0.5))*Stockopt.T)/(Stockopt.sigma*(np.sqrt(Stockopt.T)))
        self.d2 = self.d1-(Stockopt.sigma*(np.sqrt(Stockopt.T)))
        if Stockopt.is_call:
            self.Rho = Stockopt.K*Stockopt.T*(np.exp(-Stockopt.r*(Stockopt.T)))*(si.norm.cdf(self.d2))
        else:
            self.Rho = -Stockopt.K*Stockopt.T*(np.exp(-Stockopt.r*(Stockopt.T)))*(si.norm.cdf(-self.d2))
        
        return self.Rho
    
    def Gamma(self,Stockopt):
        self.d1 = (np.log(Stockopt.S0/Stockopt.K) + (Stockopt.r+ ((Stockopt.sigma**2)*0.5))*Stockopt.T)/(Stockopt.sigma*(np.sqrt(Stockopt.T)))
        #self.N_dash = (np.exp(-(self.d1**2)/2))/np.sqrt(2*math.pi)
        self.gamma = (np.exp(-Stockopt.div*(Stockopt.T))*si.norm.pdf(self.d1))/(Stockopt.S0*Stockopt.sigma*(np.sqrt(Stockopt.T)))
        return self.gamma
    
    def theta(self,Stockopt):
        self.d1 = (np.log(Stockopt.S0/Stockopt.K) + (Stockopt.r+ ((Stockopt.sigma**2)*0.5))*Stockopt.T)/(Stockopt.sigma*(np.sqrt(Stockopt.T)))
        #self.N_dash = (np.exp(-(self.d1**2)/2))/np.sqrt(2*math.pi)
        self.d2 = self.d1-(Stockopt.sigma*(np.sqrt(Stockopt.T)))
        if Stockopt.is_call:
            self.theta = -((Stockopt.S0*si.norm.pdf(self.d1)*Stockopt.sigma*np.exp(-Stockopt.div*(Stockopt.T)))/2*np.sqrt(Stockopt.T))-(Stockopt.r*Stockopt.K*(np.exp(-Stockopt.r*(Stockopt.T)))*(si.norm.cdf(self.d2)))+Stockopt.div*Stockopt.S0*np.exp(-Stockopt.div*(Stockopt.T))*(si.norm.cdf(self.d1))
        else:
            self.theta = -((Stockopt.S0*si.norm.pdf(self.d1)*Stockopt.sigma*np.exp(-Stockopt.div*(Stockopt.T)))/2*np.sqrt(Stockopt.T))+(Stockopt.r*Stockopt.K*(np.exp(-Stockopt.r*(Stockopt.T)))*(si.norm.cdf(-self.d2)))-Stockopt.div*Stockopt.S0*np.exp(-Stockopt.div*(Stockopt.T))*(si.norm.cdf(-self.d1))
        return self.theta
    
        
    
    
    
        