import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
#from xbbg import blp
import datetime as dt
from statistics import mean
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from statsmodels.tsa.vector_ar.vecm import coint_johansen


def signal(curr_z_score,
    entry_signal,
    max_posn_size,
    current_posns):
    #Enter long
    if curr_z_score <=-entry_signal and -max_posn_size<current_posns<max_posn_size:
        posn = 1
    #Enter short
    elif curr_z_score >= entry_signal and -max_posn_size<current_posns<max_posn_size:
        posn =-1
    else:
        posn = 0
    return posn

def tp(prev_posn,
    curr_z_score,
    prev_z_score,
    take_profit_level,
    current_posns):

    #Exit long
    if curr_z_score >-take_profit_level and current_posns>0 and prev_z_score < curr_z_score:
        posn = -(current_posns) #neutralise position such that net = 0
        return posn
    #Exit short
    elif curr_z_score < take_profit_level and current_posns<0 and prev_z_score >curr_z_score:
        posn = -(current_posns) #neutralise position such that net = 0
        return posn
    
    else:
        return prev_posn

def drawdown_fx(current_posns,
    trade_history,
    portfolio_returns):
    if current_posns!=0:
        index_of_day_before_last_trade_taken = final_return_before_trade(trade_history) #Calculate the drawdown of the trade
        #Calculate drawdown
        # In terms of how much drawdown did this trade cause
        drawdown = max(portfolio_returns[index_of_day_before_last_trade_taken] - portfolio_returns[-1],0)
    else:
        drawdown = 0
    return drawdown


def high_water_mark(cumulative_returns_series:list)->list:
    prev = cumulative_returns_series[0]
    high_water = []
    for i in range(1,len(cumulative_returns_series)):
        curr = cumulative_returns_series[i]
        if  curr > prev:
            high_water.append(curr)
        else:
            high_water.append(prev)
        prev = max(curr,prev)
    return [0]+high_water

def final_return_before_trade(positions_list:list)->int:
    # Returns the first index from the back in a reversed list
    start = -1
    counter  =-2
    for i in positions_list[-2::-1]:
        if i != 0:
            counter-=1
        if i ==0:
            break
    return counter

def johansen_test(df):
    result = coint_johansen(df,0,1)
    trace_crit_value = result.cvt[:,0]
    if result.lr1[0]>trace_crit_value[0]:
        print('Time series is co-integrated using Johansen test')
    else:
        print('Time series is not cointegrated using Johansen test')
    
def adfuller_test(residuals):
    if adfuller(residuals)[1] <0.05:
        print('Time series is co-integrated using ADF test.')
    else:
        print('Time series is not co-integrated using ADF test.')




def portfolio_analytics(df,transaction_cost):
    df= df.dropna()
    results_dict = {}
    start = df.index[0].strftime('%Y')
    end = df.index[-1].strftime('%Y')
    i = int(start)
    txn_cost = 1/100
    while i <= int(end):
        date_str = str(i)
        returns = round(df[date_str]['daily_rets'].sum(),3)
        volatility = round(df[date_str]['daily_rets'].std(),3)*260**0.5
        daily_sharpe =  returns/volatility
        sharpe = round(daily_sharpe,3)
        trades_taken = int(abs(df[date_str]['signals'].values).sum())
        annual_drawdown = max(high_water_mark(df[date_str].portfolio_returns)-df[date_str].portfolio_returns)
        max_drawdown = max((pd.DataFrame(high_water_mark(df[:date_str].portfolio_returns)-df[:date_str].portfolio_returns)).dropna().portfolio_returns)
        transaction_cost = trades_taken * txn_cost
        results_dict[date_str] = [returns,volatility,sharpe,trades_taken,annual_drawdown,max_drawdown,transaction_cost]
        i+=1
    Analytics = pd.DataFrame(results_dict).T
    Analytics.columns = ['Returns','Volatility','Sharpe','Number of Trades Taken','Annual Max Drawdown','Total Max Drawdown','Transaction Costs']
    return Analytics
