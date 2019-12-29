import math
from stock_volatility import stock_volatility


class stockoption():

    def __init__(self, S0, K, r, T, N,column, prm):
        """
        Initialise parameters
        :param S0: initial stock price
        :param K: strike price
        :param r: risk free interest rate per year
        :param T: length of option in years
        :param N: number of binomial iterations
        :param prm: dictionary with additional parameters
        """
        self.S0 = S0
        self.K = K
        self.r = r
        self.T = T
        self.N = N
        """
        prm parameters:
        start = date from when you want to analyse stocks, "yyyy-mm-dd"
        end = date of final stock analysis (likely current date), "yyyy-mm-dd"
        tk = ticker label
        div = dividend paid
        is_calc = is volatility calculated using stock price history, boolean
        use_garch = use GARCH model, boolean
        sigma = volatility of stock
        is_call = is it a call option, boolean
        eu_option = European or American option, boolean
        """
        self.tk = prm.get('tk', None)
        self.start = prm.get('start', None)
        self.end = prm.get('end', None)
        self.div = prm.get('div', 0)
        self.is_calc = prm.get('is_calc', False)
        self.use_garch = prm.get('use_garch', False)
        vol = stock_volatility(self.tk, self.start, self.end)
        if self.is_calc:
            if self.use_garch:
                self.sigma = vol.garch_sigma(column)
            else:
                self.sigma = vol.mean_sigma(column)
        else:
            self.sigma = prm.get('sigma', 0)
        self.is_call = prm.get('is_call', True)
        self.eu_option = prm.get('eu_option', True)
        '''
        derived values:
        dt = time per step, in years
        df = discount factor
        '''
        self.dt = T/float(N)
        self.df = math.exp(-(r-self.div)*self.dt)
