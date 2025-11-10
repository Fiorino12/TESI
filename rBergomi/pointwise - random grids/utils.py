import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

def g(x, a):
    """
    TBSS kernel applicable to the rBergomi variance process.
    """
    return x**a

def b(k, a):
    """
    Optimal discretisation of TBSS process for minimising hybrid scheme error.
    """
    return ((k**(a+1)-(k-1)**(a+1))/(a+1))**(1/a)

def cov(a, n):
    """
    Covariance matrix for given alpha and n, assuming kappa = 1 for
    tractability.
    """
    cov = np.array([[0.,0.],[0.,0.]])
    cov[0,0] = 1./n
    cov[0,1] = 1./((1.*a+1) * n**(1.*a+1))
    cov[1,1] = 1./((2.*a+1) * n**(2.*a+1))
    cov[1,0] = cov[0,1]
    return cov

def bs(F, K, V, o = 'call'):
    """
    Returns the Black call price for given forward, strike and integrated
    variance.
    """

    sv = np.sqrt(np.maximum(V, 1e-16))
    d1 = np.log(F/K) / sv + 0.5 * sv
    d2 = d1 - sv
    P = F * norm.cdf(d1) - K * norm.cdf(d2)
    return P

def bsinv(P, F, K, t, o = 'call'):
    """
    Returns implied Black vol from given call price, forward, strike and time
    to maturity.
    """
    # Set appropriate weight for option token o
    w = 1
    if o == 'put':
        w = -1
    elif o == 'otm':
        w = 2 * (K > 1.0) - 1

    # Ensure at least instrinsic value
    P = np.maximum(P, np.maximum(w * (F - K), 0))

    def error(s):
        return bs(F, K, s**2 * t, o) - P
    try:
        iv = brentq(error, 1e-10, 1.0, maxiter=4000)
        return  np.clip(iv, 1e-10, 1.0)
    except ValueError:
        # Handle cases where solution is not found
        return np.nan

def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    """
    Calculate Black-Scholes option price
    
    Parameters:
    - S: Spot price
    - K: Strike price
    - T: Time to maturity
    - r: Risk-free rate
    - sigma: Volatility
    - option_type: 'call' or 'put'
    
    Returns:
    - Option price
    """
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    if option_type == 'call':
        price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
    
    return price

def implied_volatility(market_price, S, K, T, r, option_type='call'):
    """
    Calculate implied volatility using Brent's method
    
    Parameters:
    - market_price: Observed market price of option
    - S: Spot price
    - K: Strike price
    - T: Time to maturity
    - r: Risk-free rate
    - option_type: 'call' or 'put'
    
    Returns:
    - Implied volatility
    """
    # Define the price difference function to find root of
    def price_diff(sigma):
        return black_scholes_price(S, K, T, r, sigma, option_type) - market_price
    
    # Brent's method bounds (1e-6% to 100%)
    try:
        iv = brentq(price_diff, 1e-10, 1.0, maxiter=4000)
        return  np.clip(iv, 1e-10, 1.0)
    except ValueError:
        # Handle cases where solution is not found
        return np.nan
