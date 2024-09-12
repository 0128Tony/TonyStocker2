import pandas as pd
import numpy as np
import fbprophet
import matplotlib.pyplot as plt
import matplotlib

class Stocker():
    
    def __init__(self, price):
        self.symbol = 'the stock'
        stock = pd.DataFrame({'Date': price.index, 'y': price, 'ds': price.index, 'close': price, 'open': price})
        stock['Adj. Close'] = stock.get('close', stock['close'])
        stock['Adj. Open'] = stock.get('open', stock['open'])
        stock['y'] = stock['Adj. Close']
        stock['Daily Change'] = stock['Adj. Close'] - stock['Adj. Open']
        self.stock = stock
        self.min_date, self.max_date = min(stock['ds']), max(stock['ds'])
        self.max_price, self.min_price = stock['y'].max(), stock['y'].min()
        self.starting_price, self.most_recent_price = stock['Adj. Open'].iloc[0], stock['y'].iloc[-1]
        self.changepoint_prior_scale, self.training_years = 0.05, 3
        print(f'{self.symbol} Stocker Initialized. Data covers {self.min_date} to {self.max_date}.')
    
    def handle_dates(self, start_date=None, end_date=None):
        start_date = pd.to_datetime(start_date if start_date else self.min_date)
        end_date = pd.to_datetime(end_date if end_date else self.max_date)
        return start_date, end_date

    def make_df(self, start_date, end_date, df=None):
        df = df if df is not None else self.stock.copy()
        start_date, end_date = self.handle_dates(start_date, end_date)
        return df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

    def plot_stock(self, start_date=None, end_date=None, stats=['Adj. Close'], plot_type='basic'):
        stock_plot = self.make_df(start_date, end_date)
        colors = ['r', 'b', 'g', 'y', 'c', 'm']
        
        for i, stat in enumerate(stats):
            plt.plot(stock_plot['Date'], stock_plot[stat], color=colors[i], label=stat)
        plt.title(f'{self.symbol} Stock History'); plt.legend(); plt.show()

    @staticmethod
    def reset_plot():
        matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    
    def create_model(self):
        model = fbprophet.Prophet(
            daily_seasonality=False, 
            weekly_seasonality=False,
            yearly_seasonality=True,
            changepoint_prior_scale=self.changepoint_prior_scale
        )
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        return model

    def buy_and_hold(self, start_date=None, end_date=None, nshares=1):
        start_date, end_date = self.handle_dates(start_date, end_date)
        start_price, end_price = self.stock[self.stock['Date'] == start_date]['Adj. Open'].values[0], self.stock[self.stock['Date'] == end_date]['Adj. Close'].values[0]
        profits = self.make_df(start_date, end_date)
        total_hold_profit = nshares * (end_price - start_price)
        print(f'Total buy and hold profit from {start_date} to {end_date} for {nshares} shares = ${total_hold_profit:.2f}')
    
    def changepoint_prior_analysis(self, changepoint_priors=[0.001, 0.05, 0.1, 0.2]):
        train = self.stock[self.stock['Date'] > (self.max_date - pd.DateOffset(years=self.training_years))]
        colors = ['b', 'r', 'grey', 'gold']
        for prior, color in zip(changepoint_priors, colors):
            model = self.create_model()
            model.fit(train)
            future = model.make_future_dataframe(periods=180, freq='D')
            forecast = model.predict(future)
            plt.plot(forecast['ds'], forecast['yhat'], color=color, label=f'{prior} prior scale')
        plt.legend(); plt.show()

    def create_prophet_model(self, days=0, resample=False):
        model = self.create_model()
        stock_history = self.stock[self.stock['Date'] > (self.max_date - pd.DateOffset(years=self.training_years))]
        model.fit(stock_history)
        future = model.make_future_dataframe(periods=days, freq='D')
        future = model.predict(future)
        plt.plot(stock_history['ds'], stock_history['y'], 'ko-', label='Observations')
        plt.plot(future['ds'], future['yhat'], 'forestgreen', label='Modeled')
        plt.fill_between(future['ds'], future['yhat_upper'], future['yhat_lower'], color='g', alpha=0.3)
        plt.legend(); plt.show()
