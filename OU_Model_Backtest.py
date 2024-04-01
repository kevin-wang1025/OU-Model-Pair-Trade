import backtrader as bt 
import pandas as pd 
import datetime 
import matplotlib.pyplot as py 
import seaborn as sns 
from sympy import log 
import math

# 宣告全域便數紀錄權益變化
lst = []
balance = 10000

class stat_arb(bt.Strategy):

    author = 'Kevin Wang'

    params = (
        ('period',40),
    )

    def log(self,message,dt=None):
        dt = dt or self.datas[0].datetime.datetime(0)
        print('{} {}'.format(dt.isoformat(),message))
    
    def __init__(self):

        self.dataclose = self.datas[0].close
        self.data2close = self.datas[1].close
        self.spread = self.datas[0].high 

        # Condition (0.2432, 0.1757, 0.1081)
        self.converge = bt.indicators.CrossDown(self.spread,3.4583)
        self.diverge = bt.indicators.CrossOver(self.spread,3.3932)
        self.converge_tp = bt.indicators.CrossDown(self.spread,3/4258)
        self.diverge_tp = bt.indicators.CrossOver(self.spread,3.4258) 
        self.converge_sl = bt.indicators.CrossOver(self.spread,3.4746)
        self.diverge_sl = bt.indicators.CrossDown(self.spread,3.377)

        self.record = 0
        self.trading_time = 0
        self.order = None
        self.buyprice = None 
        self.buycomm = None 

    
    def notify_order(self,order):
        if order.status in [order.Submitted,order.Accepted]:
            return 
        
        if order.status in [order.Completed]:
            self.bar_executed = len(self)
            if order.isbuy():
                self.log('buy executed,price: %.2f, Cost: %.2f, comm %.2f, marketposition1 %.2f, marketposition2 %.2f,trades %.2f'%(order.executed.price,order.executed.value,order.executed.comm,self.getposition(data=self.datas[0]).size,self.getposition(data=self.datas[1]).size,self.trading_time))
                self.buyprice = order.executed.price 
                self.buycomm = order.executed.comm 
            
            else:
                self.log('sell executed, price: %.2f, Cost: %.2f, Comm %.2f, marketposition1 %.2f, marketposition2 %.2f,trades %.2f' %(order.executed.price,order.executed.value,order.executed.comm,self.getposition(data=self.datas[0]).size,self.getposition(data=self.datas[1]).size,self.trading_time))
            
            self.bar_executed = len(self)
        
        elif order.status in [order.Canceled,order.Margin,order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
        
        self.order = None
    
    def notify_trade(self,trade):
        global balance
        if not trade.isclosed:
            return 
        
        self.log('operation profit,gross %.2f, net %.2f' %(trade.pnl,trade.pnlcomm))
        balance = balance + trade.pnlcomm 
    
    def next(self):

        global lst 
        global balance 

        lst.append(balance)

        #self.log('year {}'.format(self.year[0]))

        if self.order:
            return 
        
        # 進場
        if not self.position:
            if self.converge:
                self.sell(data=self.datas[0],size=8200/self.dataclose[0])
                self.buy(data=self.datas[1],size=1800/self.data2close[0])
            
            if self.diverge:
                self.buy(data=self.datas[0],size=8200/self.dataclose[0])
                self.sell(data=self.datas[1],size=1800/self.data2close[0])

        # 出場
        if self.getposition(data=self.datas[0]).size>0:
            if self.converge_tp :#or self.converge_sl:
                self.close(data=self.datas[0])
                self.close(data=self.datas[1])
                self.trading_time = self.trading_time + 1
            
        if self.getposition(data=self.datas[0]).size<0:
            if self.diverge_tp :#or self.diverge_sl:
                self.close(data=self.datas[0])
                self.close(data=self.datas[1])
                self.trading_time = self.trading_time + 1

        # 固定天數出場
        if self.getposition(data=self.datas[0]).size!=0 and len(self)==(self.bar_executed + self.params.period):
            self.close(data=self.datas[0])
            self.close(data=self.datas[1])
            self.trading_time = self.trading_time + 1
    
if __name__=='__main__':

    cerebro = bt.Cerebro(cheat_on_open=True)
    cerebro.addstrategy(stat_arb)

    data = bt.feeds.GenericCSVData(
        dataname = "C:\\Users\\10830\\Desktop\\TMBA專案\\OU_Model 數據\\PEP.csv",
        fromdate = datetime.datetime(2009,11,30),
        todate = datetime.datetime(2012,10,31),

        nullvalue = 0.0,

        dtformat=('%Y/%m/%d'),

        datetime = 0,
        open = 1,
        high = 2,
        low = 3,
        close = 4,
        volume = 5,
        openinterest = -1
    )

    data2 = bt.feeds.GenericCSVData(
        dataname = "C:\\Users\\10830\\Desktop\\TMBA專案\\OU_Model 數據\\KO.csv",
        fromdate = datetime.datetime(2009,11,30),
        todate = datetime.datetime(2012,10,31),

        nullvalue = 0.0,

        dtformat=('%Y/%m/%d'),

        datetime = 0,
        open = 1,
        high = 2,
        low = 3,
        close = 4,
        volume = 5,
        openinterest = -1
    )

    cerebro.adddata(data)
    cerebro.adddata(data2)
    cerebro.broker.setcommission(0)
    cerebro.broker.setcash(10000)

    cerebro.addanalyzer(bt.analyzers.PyFolio,_name='PyFolio')
    cerebro.addanalyzer(bt.analyzers.AnnualReturn,_name='AnnualReturn')
    cerebro.addanalyzer(bt.analyzers.DrawDown,_name='DrawDown')

    print('Start {}',format(cerebro.broker.getvalue()))
    back = cerebro.run() 
    print('End {}',format(cerebro.broker.getvalue()))
    print("---------------AnnualReturn---------------")
    print(back[0].analyzers.AnnualReturn.get_analysis())
    print("---------------MaxDrawDown---------------")
    print(back[0].analyzers.DrawDown.get_analysis()['max']['drawdown'])

    # quantstats回測報表
    strat = back[0]
    portfolio_stats = strat.analyzers.getbyname('PyFolio')
    returns,positions,transaction,gross_lev = portfolio_stats.get_pf_items()
    print(returns)
    returns.index = returns.index.tz_convert(None)

    import quantstats
    quantstats.reports.full(returns)
    quantstats.reports.html(returns,output='stats.html',title='統計套利策略',download_filename='stat_arb.html')

    # 繪製equity curve
    py.style.use('seaborn')
    py.figure(figsize=(15,6))
    lst = pd.Series(lst)
    MDD_Series = lst.cummax()-lst 
    MDD_Percent = MDD_Series/lst.cummax()
    lst.plot()
    py.title('Equity Curve')
    py.show() 

    py.figure(figsize=(10,6))
    lst_return = lst/10000-1  
    lst_return.plot(color='red')
    py.fill_between(MDD_Percent.index,-MDD_Percent,0,facecolor='red',label='DD')
    high_index = lst_return[lst_return.cummax()==lst_return].index
    py.scatter(high_index,lst_return.loc[high_index],c='#02ff0f',label='High')
    py.title('Cumulative Return')
    py.show()

    print('Return: ',lst.iloc[-1]/10000-1)
    print('Max Draw Down: ',MDD_Percent.max())

    print('==========================回測完成============================')
    