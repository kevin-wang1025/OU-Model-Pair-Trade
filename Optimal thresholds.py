# Computing optimal thresholds

import pandas as pd 
import numpy as np
from statsmodels.tsa.stattools import adfuller 
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm 
import math
import matplotlib.pyplot as py 
import seaborn as sns 
py.style.use('seaborn')
from sympy import symbols,pi,exp,log,summation,diff,simplify,solve,nsolve,factorial,gamma,hyper,sqrt
from sympy.stats import Probability, Normal 
from scipy.optimize import fsolve,minimize,dual_annealing
from sympy.utilities.lambdify import lambdify
from scipy.special import hyp2f1,hyp1f1

eth = pd.read_csv("C:\\Users\\10830\\Desktop\\TMBA專案\\OU_Model 數據\\PEP.csv")
sol = pd.read_csv("C:\\Users\\10830\\Desktop\\TMBA專案\\OU_Model 數據\\KO.csv")

def opt_thresholds(data1,data2,transaction_cost):

    """
    Parameters
    -------------------------
    data1 : Series
            One of our trading pairs
    data2 : Series
            One of our trading pairs
    transaction_cost : float
            Preset transaction cost per dollar invested
    """

    # # Engle & Granger Test
    # test = coint(data1,data2)
    # test2 = coint(data2,data1)
    # if test[0]<test[2][2]:
    #     print('具有共整合關係')
    # else:
    #     print('不具有共整合關係')
    #     return

    # Compute beta

    ln_data1 = np.log(data1)
    ln_data2 = np.log(data2)

    def ret(df):
        return df-df.iloc[0]
    
    log_ret_data1 = ret(ln_data1).dropna()
    log_ret_data2 = ret(ln_data2).dropna()

    X = sm.add_constant(log_ret_data2)
    Y = log_ret_data1
    res = sm.OLS(Y,X).fit()
    beta = res.params[1]
    print('beta值: ',beta)

    # Maximum likelihood function 
    # https://mlln.cn/2019/01/24/%E6%95%B0%E5%AD%A6%E5%B0%8F%E7%99%BD%E7%94%A8python%E5%81%9A%E6%9E%81%E5%A4%A7%E4%BC%BC%E7%84%B6%E4%BC%B0%E8%AE%A1MLE/
    # https://docs.scipy.org/doc/scipy/reference/optimize.html
    # https://stackoverflow.com/questions/34115233/python-optimization-using-sympy-lambdify-and-scipy
    Xt = ln_data1 - beta * ln_data2
    n = Xt.shape[0]
    θ = symbols('θ')
    σ = symbols('σ')
    x = symbols('x')
    μ = symbols('μ')
    print(Xt.mean())
    py.figure(figsize=(15,6))
    Xt.plot()
    py.show()

    part1 = 0
    for i in range(1,n-1):
        part1 = part1 + log(1-exp(-2*θ*1))
    
    part2 = 0
    for i in range(1,n-1):        
        part2 = part2 + (Xt.iloc[i]-Xt.mean()-(Xt.iloc[i-1]-Xt.mean())*exp(-θ*1))/(1-exp(-2*θ*1))
    
    likelihood_function = (-1*n/2-0.5*part1-(θ/σ**2)*part2)*(-1)
    likelihood_function = lambdify((θ,σ),likelihood_function)
    print('====================================================')

    def eqn(x):
        #μ = x[0]
        θ = x[0]
        σ = x[1]
        
        return abs(likelihood_function(θ,σ))

    initial_guess = [0.01,0.01]
    result = minimize(eqn,initial_guess,method='Nelder-Mead')
    print(result)

    mu = Xt.mean()
    theta = result.x[0]
    sigma = result.x[1]
    print('mu:',mu)
    print('theta:',theta)
    print('sigma:',sigma)
    print('====================================================')

    # Find optimal threshold a*,b*
    dimensionless_c = transaction_cost*(2*theta)**0.5/sigma

    a = symbols('a')
    n = symbols('n')
    infinite = float('inf')
    equation = 0.5*summation((2**0.5*a)**(2*n-1)/factorial(2*n+1)*gamma((2*n+1)/2),(n,0,infinite)) - (a-dimensionless_c)*2**0.5/2*summation((2**0.5*a)**(2*n)/factorial(2*n)*gamma((2*n+1)/2),(n,0,infinite))
    equation = simplify(equation)
    print(equation)

def solve(mu,sigma,theta):

    # print(hyp1f1(0.5,1.5,0.5*2**2))
    # print(hyper((0.5,), (3/2,), 0.5*2**2).evalf())
    
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.hyp1f1.html#scipy.special.hyp1f1
    # https://stackoverflow.com/questions/56225509/python-symbolic-use-of-hyp2f1-with-sympy
    def func(x):
        x = x[0]
        return (math.pi)**0.5*(-1*x*(0.707106781186548*x - 0.424762018355669)*exp(0.5*x**2) + 0.353553390593274*hyp1f1(0.5,1.5,0.5*x**2))/x
    
    ans = fsolve(func,[0.01])
    
    dimensionless_a = ans
    dimensionless_b = 0
    real_a = dimensionless_a*sigma/(2*theta)**0.5 + mu
    neg_real_a = -1*dimensionless_a*sigma/(2*theta)**0.5 + mu
    real_b = mu
    converge_stop_loss = real_a + 0.5 * (real_a-real_b)
    diverge_stop_loss = neg_real_a - 0.5 * (real_b-neg_real_a)
    print('Optimal thresholds: ',real_a[0],real_b,neg_real_a[0])
    print('Stop loss point: ',converge_stop_loss,diverge_stop_loss)

    # 畫圖
    ln_data1 = np.log(eth['Close'])
    ln_data2 = np.log(sol['Close'])

    def ret(df):
        return df - df.iloc[0]
    
    log_ret_data1 = ret(ln_data1).dropna()
    log_ret_data2 = ret(ln_data2).dropna()

    X = sm.add_constant(log_ret_data2)
    Y = log_ret_data1 
    res = sm.OLS(Y,X).fit()
    beta = res.params[1] 

    Xt = ln_data1 - beta * ln_data2 
    py.figure(figsize=(15,6))
    Xt.plot()
    py.axhline(y=real_a[0],c='r',ls='--')
    py.axhline(y=neg_real_a[0],c='r',ls='--')
    py.axhline(y=real_b,c='blue',ls='--')
    py.axhline(converge_stop_loss,c='purple',ls='--')
    py.axhline(diverge_stop_loss,c='purple',ls='--')
    py.ylabel('spread')
    py.show() 

#opt_thresholds(eth['Close'],sol['Close'],0.02)
solve(3.4257681088496805,0.005581081386597348,0.014049733020030538)


    
