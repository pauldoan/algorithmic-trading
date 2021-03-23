# Technical Analysis Library
# Allow to extract signals from talib tools, and plot indicators

'''
For all functions:
- prices should have the format of Token Metrics ico_daily_summaries table
- days is the number of days to output in the past

We usually return the following columns:
- signal: whether there is a buy or sell signal
- trends: binary output used for GA

For more information on the indicators, check
https://mrjbq7.github.io/ta-lib/
'''
import talib
from functools import reduce
import pandas as _pd
import numpy as _np
import quantstats as _qs
import matplotlib.pyplot as _plt


###########
# Helper functions
###########


def signals_stats(signals):
    '''
    Computing several stats on set of trading signals (accuracy, roi, mean, max, min)

    parameters
    ----------
    signals: dataframe with the trading signals

    output
    -------
    dictionnary containing the stats
    '''
    # Creating a dictionnary to contain all statistics about the Mama indicator
    stats = dict()

    # make sure there is at least 1 buy signal
    if len(signals[signals.signal == 1]) == 0:
        return None

    # First computing the holding ROI if we had bought on the first signal and hodl
    initial_price = signals[signals.signal == 1].iloc[0].close
    last_price = signals.iloc[-1].close
    holding_roi = (last_price - initial_price) / initial_price

    # Retrieving signals only
    signals_only = signals[signals.signal != 0]

    # First signal is a sell, we change to a buy and discard it
    if signals_only.iloc[0].signal == -1:
        signals_only = signals_only.drop(signals_only.index[0])

    # Getting all signals
    signal_buy = signals_only[signals_only.signal == 1][['date', 'close']].rename(columns={'date': 'date_buy', 'close': 'buy'}).reset_index(drop=True)
    signal_sell = signals_only[signals_only.signal == -1][['date', 'close']].rename(columns={'date': 'date_sell', 'close': 'sell'}).reset_index(drop=True)
    trades = _pd.concat((signal_buy, signal_sell), axis=1)

    # If last signal is a buy, we compute the unrealized ROI
    unrealized = False
    if _pd.isna(trades.sell.iloc[-1]):
        unrealized = True
        unrealized_roi = (last_price - trades.buy.iloc[-1]) / trades.buy.iloc[-1] + 1

    # dropping NAN now that we dealt with unrealized
    trades = trades.dropna()

    # ROI, included unrealized if needed
    trades['roi'] = (trades.sell - trades.buy) / trades.buy + 1

    if unrealized:
        rois = list(trades.roi)
        rois.append(unrealized_roi)
        rois = _np.array(rois)
        stats['total_roi'] = rois.prod() - 1

    else:
        stats['total_roi'] = (trades.roi.prod() - 1)

    # holding ROI
    stats['holding_roi'] = holding_roi

    # Accuracy
    trades['hit'] = trades.buy < trades.sell
    stats['accuracy'] = trades.hit.mean()

    # quantstats
    pnl = get_pnl(signals)
    daily_returns = pnl.set_index('date').strategy_daily_roi
    stats['sortino'] = _qs.stats.sortino(daily_returns, periods=365)
    stats['max_drawdown'] = _qs.stats.max_drawdown(daily_returns)
    stats['volatility'] = _qs.stats.volatility(daily_returns, periods=365)

    # other stats
    stats['avg_roi'] = (trades.roi - 1).mean()
    stats['best_roi'] = (trades.roi - 1).max()
    stats['win_avg_roi'] = (trades[trades.hit == True].roi - 1).mean()
    stats['worst_roi'] = (trades.roi - 1).min()
    stats['loss_avg_roi'] = (trades[trades.hit == False].roi - 1).mean()
    stats['nb_trades'] = len(trades)

    return stats


def holding_stats(prices):
    '''
    Computing stats on rough price

    parameters
    ----------
    prices: dataframe with all historical prices

    output
    -------
    dictionnary containing the stats
    '''
    # Creating a dictionnary to contain all statistics about the Mama indicator
    stats = dict()

    # First computing the holding ROI if we had bought on the first signal and hodl
    initial_price = prices.iloc[0].close
    last_price = prices.iloc[-1].close
    holding_roi = (last_price - initial_price) / initial_price
    stats['roi'] = holding_roi

    # quantstats
    returns = prices.close.pct_change()
    stats['sortino'] = _qs.stats.sortino(returns, periods=365)
    stats['max_drawdown'] = _qs.stats.max_drawdown(returns)
    stats['volatility'] = _qs.stats.volatility(returns, periods=365)

    return stats


def get_trades(signals):

    # Extracting signals only
    signals = signals[signals.signal != 0]
    if len(signals) == 0:
        return None

    # First signal is a sell
    if signals.iloc[0].signal == -1:
        signals = signals.drop(signals.index[0])

    # Getting all signals
    signal_buy = signals[signals.signal == 1][['date', 'close']].rename(columns={'date': 'date_buy', 'close': 'buy'}).reset_index(drop=True)
    signal_sell = signals[signals.signal == -1][['date', 'close']].rename(columns={'date': 'date_sell', 'close': 'sell'}).reset_index(drop=True)
    trades = _pd.concat((signal_buy, signal_sell), axis=1)
    trades = trades.dropna()

    # ROI
    trades['roi'] = (trades.sell - trades.buy) / trades.buy
    return trades


def get_pnl(signals):
    '''
    Plotting pnl of strategy vs holding.
    '''
    # Start the dataframe after the first buy signal
    pnl = signals[signals.index >= signals[signals.signal == 1].index[0]].copy()

    # create a IN/OUT of market column
    pnl['market'] = pnl.signal.replace(to_replace=0, method='ffill')

    # creating a strategy pct change column to compute strategy ROI
    pnl['strategy_daily_roi'] = _np.where((pnl.market == 1) & (pnl.signal != 1) | (pnl.signal == -1), pnl.close.pct_change(), 0)

    # computing strategy ROI
    pnl['strategy_cumulative_roi'] = (pnl.strategy_daily_roi + 1).cumprod().fillna(1) - 1

    return pnl


def plot_signals(signals, figsize=(17, 10), notebook=True):
    '''
    Plotting prices with signals.
    '''
    # _plt.style.use('fivethirtyeight')
    # _plt.style.use("dark_background")
    # _plt.rcParams['grid.color'] = '#3c4f6e'

    # Plotting main price line
    fig, ax = _plt.subplots(figsize=figsize)
    signals.plot(x='date', y='close', linewidth=1, label='price', color='slategray', ax=ax)

    # Creating a new dataframe only for signals
    signals_1 = signals[signals.signal == 1].copy()
    signals_min1 = signals[signals.signal == -1].copy()

    # Annotating signals for better visibility - Bearish signals
    ax.scatter(signals_min1.date.values, signals_min1.close.values, color='Red', s=50)
    for i in range(len(signals_min1)):
        ax.annotate('Bearish', xy=(signals_min1.date.values[i], signals_min1.close.values[i]),
                    textcoords='offset points', xytext=(0, 13), color='white',
                    bbox=dict(boxstyle="Round", fc="red", alpha=.8), horizontalalignment='center', verticalalignment='bottom')

    # Annotating signals for better visibility - Bullish signals
    ax.scatter(signals_1.date.values, signals_1.close.values, color='lime', s=50)
    for i in range(len(signals_1)):
        ax.annotate('Bullish', xy=(signals_1.date.values[i], signals_1.close.values[i]),
                    textcoords='offset points', xytext=(0, 13), color='white',
                    bbox=dict(boxstyle="Round", fc="green", alpha=.8), horizontalalignment='center', verticalalignment='bottom')

    # Titles and legend
    ax.legend(loc='best')
    _plt.title('Signals')

    if notebook:
        _plt.show()
    else:
        return fig


############################
# Overlap Indicators
############################


def mama(prices, days=None):

    # if not enough price data, we cannot compute mama
    if len(prices) <= 35:
        print('Not enough price data to compute MAMA')
        return None

    # Getting mama and fama values
    mama, fama = talib.MAMA(prices.close.values)
    # Creating dataframe
    df = _pd.DataFrame({'ico_id': prices.ico_id.values, 'symbol': prices.ico_symbol.values, 'date': prices.date.values, 'close': prices.close.values, 'mama': mama, 'fama': fama})

    # Getting lag values
    df['mama_lag'] = df.mama.shift()
    df['fama_lag'] = df.fama.shift()

    # Crossing conditions
    conditions = [((df.mama > df.fama) & (df.mama_lag > df.fama_lag)),
                  ((df.mama < df.fama) & (df.mama_lag < df.fama_lag)),
                  ((df.mama > df.fama) & (df.mama_lag < df.fama_lag)),
                  ((df.mama < df.fama) & (df.mama_lag > df.fama_lag))]

    # Creating signal, position and trend columns
    df['trend'] = _np.select(condlist=conditions, choicelist=[1, -1, 1, -1])
    df['signal'] = _np.select(condlist=conditions, choicelist=[0, 0, 1, -1])

    # Dropping lag features and NaN values and invalid values
    df.drop(['mama_lag', 'fama_lag'], axis=1, inplace=True)
    df.dropna(inplace=True)
    df = df.drop(df.index[0])

    # Returning dataframe
    if days is not None:
        if days >= len(df):
            pass
        else:
            df = df.iloc[len(df) - days:]
    return df.reset_index(drop=True)


def frama(prices, span=22, days=None):

    # if not enough price data, we cannot compute
    if len(prices) <= 2:
        print('Not enough price data to compute Frama')
        return None

    # df will be our final dataframe
    df = prices.reset_index(drop=True).copy()
    input_price = prices.reset_index(drop=True).close
    batch = 10

    # Initialize output before the algorithm
    Filt = _np.array(input_price)

    # sequencially calculate all variables and the output
    for i in range(len(prices)):

        # If there's not enough data, Filt is the price - whitch it already is, so just skip
        if i < 2 * batch:
            continue

        # take 2 batches of the input
        v1 = input_price[i-2*batch:i - batch]
        v2 = input_price[i - batch:i]

        # for the 1st batch calculate N1
        H1 = _np.max(v1)
        L1 = _np.min(v1)
        N1 = (H1 - L1) / batch

        # for the 2nd batch calculate N2
        H2 = _np.max(v2)
        L2 = _np.min(v2)
        N2 = (H2 - L2) / batch

        # for both batches calculate N3
        H = _np.max([H1, H2])
        L = _np.min([L1, L2])
        N3 = (H - L) / (2 * batch)

        # calculate fractal dimension
        Dimen = 0
        if N1 > 0 and N2 > 0 and N3 > 0:
            Dimen = (_np.log(N1 + N2) - _np.log(N3)) / _np.log(2)

        # calculate lowpass filter factor
        alpha = _np.exp(-4.6 * (Dimen) - 1)
        alpha = _np.max([alpha, 0.1])
        alpha = _np.min([alpha, 1])

        # filter the input data
        Filt[i] = alpha * input_price[i] + (1 - alpha) * Filt[i-1]

    # Getting MAs
    df['frama'] = Filt
    df['ema'] = df.close.ewm(span=span, adjust=False).mean()

    # Getting lag values
    df['frama_lag'] = df.frama.shift()
    df['ema_lag'] = df.ema.shift()

    # Crossing Conditions with EMA
    conditions = [((df.frama > df.ema) & (df.frama_lag > df.ema_lag)),  # Bullish
                  ((df.frama < df.ema) & (df.frama_lag < df.ema_lag)),  # Bearish
                  ((df.frama > df.ema) & (df.frama_lag < df.ema_lag)),  # Sell if ema crosses above frama
                  ((df.frama < df.ema) & (df.frama_lag > df.ema_lag))]  # Buy if ema crosses below frama

    # Creating signal, position and trend columns
    df['trend'] = _np.select(condlist=conditions, choicelist=[1, -1, 1, -1])
    df['signal'] = _np.select(condlist=conditions, choicelist=[0, 0, 1, -1])

    # Dropping lag features and NaN values and invalid values
    df = df[['ico_id', 'ico_symbol', 'date', 'close', 'frama', 'ema', 'trend', 'signal']].rename(columns={'ico_symbol': 'symbol'})
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = df.drop([df.index[0], df.index[1]])

    # Returning dataframe
    if days is not None:
        df = df.loc[len(df) - days:]
    return df.reset_index(drop=True)


def sar(prices, days=None):

    # if not enough price data, we cannot compute
    if len(prices) <= 2:
        print('Not enough price data to compute SAR')
        return None

    # Getting sar values
    sar = talib.SAR(prices.high.values, prices.low.values, acceleration=0.02, maximum=0.2)
    # Creating dataframe
    df = _pd.DataFrame({'ico_id': prices.ico_id.values, 'symbol': prices.ico_symbol.values, 'date': prices.date.values, 'close': prices.close.values, 'sar': sar})

    # Getting lag values
    df['sar_lag'] = df.sar.shift()
    df['close_lag'] = df.close.shift()

    # Crossing conditions
    conditions = [((df.sar > df.close) & (df.sar_lag > df.close_lag)),
                  ((df.sar < df.close) & (df.sar_lag < df.close_lag)),
                  ((df.sar > df.close) & (df.sar_lag < df.close_lag)),
                  ((df.sar < df.close) & (df.sar_lag > df.close_lag))]

    # Creating signal, position and trend columns
    df['trend'] = _np.select(condlist=conditions, choicelist=[-1, 1, -1, 1])
    df['signal'] = _np.select(condlist=conditions, choicelist=[0, 0, -1, 1])

    # Dropping lag features and NaN values and invalid values
    df.drop(['sar_lag', 'close_lag'], axis=1, inplace=True)
    df.dropna(inplace=True)
    df = df.drop(df.index[0])

    # Returning dataframe
    if days is not None:
        if days >= len(df):
            pass
        else:
            df = df.iloc[len(df) - days:]
    return df.reset_index(drop=True)


def sma(prices, SMA1=20, SMA2=50, days=None):

    # In case not in good order
    if SMA1 > SMA2:
        SMA1, SMA2 = SMA2, SMA1

    # if not enough price data, we cannot compute
    if len(prices) <= SMA2 + 1:
        print('Not enough price data to compute SMA')
        return None

    # Getting sma values
    sma1 = talib.SMA(prices['close'].values, SMA1)
    sma2 = talib.SMA(prices['close'].values, SMA2)

    # Creating dataframe
    df = _pd.DataFrame({'ico_id': prices.ico_id.values, 'symbol': prices.ico_symbol.values, 'date': prices.date.values, 'close': prices.close.values, 'sma1': sma1, 'sma2': sma2})

    # Getting lag values
    df['sma1_lag'] = df.sma1.shift()
    df['sma2_lag'] = df.sma2.shift()

    # Crossing conditions
    conditions = [((df.sma1 > df.sma2) & (df.sma1_lag > df.sma2_lag)),
                  ((df.sma1 < df.sma2) & (df.sma1_lag < df.sma2_lag)),
                  ((df.sma1 > df.sma2) & (df.sma1_lag < df.sma2_lag)),
                  ((df.sma1 < df.sma2) & (df.sma1_lag > df.sma2_lag))]

    # Creating signal, position and trend columns
    df['trend'] = _np.select(condlist=conditions, choicelist=[1, -1, 1, -1])
    df['signal'] = _np.select(condlist=conditions, choicelist=[0, 0, 1, -1])

    # Dropping lag features and NaN values and invalid values
    df.drop(['sma1_lag', 'sma2_lag'], axis=1, inplace=True)
    df.dropna(inplace=True)
    df = df.drop(df.index[0])

    # Returning dataframe
    if days is not None:
        if days >= len(df):
            pass
        else:
            df = df.iloc[len(df) - days:]
    return df.reset_index(drop=True)


def bollinger(prices, window=10, n_std=1, days=None):

    # if not enough price data, we cannot compute
    if len(prices) <= window + 1:
        print('Not enough price data to compute Bollinger')
        return None

    # Getting bands values
    upperband, middleband, lowerband = talib.BBANDS(prices.close.values, window, n_std, n_std)

    # Creating dataframe
    df = _pd.DataFrame({'ico_id': prices.ico_id.values, 'symbol': prices.ico_symbol.values, 'date': prices.date.values, 'close': prices.close.values, 'upperband': upperband, 'middleband': middleband, 'lowerband': lowerband})

    # Getting lag values
    df['upperband_lag'] = df.upperband.shift()
    df['lowerband_lag'] = df.lowerband.shift()
    df['close_lag'] = df.close.shift()

    # Crossing conditions
    conditions = [((df.close > df.upperband) & (df.close_lag < df.upperband_lag)),
                  ((df.close < df.lowerband) & (df.close_lag > df.lowerband_lag)),
                  ((df.close > df.upperband) & (df.close_lag > df.upperband_lag)),
                  ((df.close < df.lowerband) & (df.close_lag < df.lowerband_lag))]

    # Creating signal, position and trend columns
    df['trend'] = _np.select(condlist=conditions, choicelist=[-1, 1, -1, 1], default=0)
    df['signal'] = _np.select(condlist=conditions, choicelist=[-1, 1, 0, 0], default=0)

    # Dropping lag features and NaN values and invalid values
    df.drop(['upperband_lag', 'lowerband_lag', 'close_lag'], axis=1, inplace=True)
    df.dropna(inplace=True)
    df = df.drop(df.index[0])

    # Returning dataframe
    if days is not None:
        if days >= len(df):
            pass
        else:
            df = df.iloc[len(df) - days:]
    return df.reset_index(drop=True)


def kama(prices, window1=20, window2=50, days=None):

    # In case not in good order
    if window1 > window2:
        window1, window2 = window2, window1

    # if not enough price data, we cannot compute
    if len(prices) <= window2 + 2:
        print('Not enough price data to compute KAMA')
        return None

    # Getting kama values
    kama1 = talib.KAMA(prices.close.values, timeperiod=window1)
    kama2 = talib.KAMA(prices.close.values, timeperiod=window2)

    # Creating dataframe
    df = _pd.DataFrame({'ico_id': prices.ico_id.values, 'symbol': prices.ico_symbol.values, 'date': prices.date.values, 'close': prices.close.values, 'kama1': kama1, 'kama2': kama2})

    # Getting lag values
    df['kama1_lag'] = df.kama1.shift()
    df['kama2_lag'] = df.kama2.shift()

    # Crossing conditions
    conditions = [((df.kama1 > df.kama2) & (df.kama1_lag > df.kama2_lag)),
                  ((df.kama1 < df.kama2) & (df.kama1_lag < df.kama2_lag)),
                  ((df.kama1 > df.kama2) & (df.kama1_lag < df.kama2_lag)),
                  ((df.kama1 < df.kama2) & (df.kama1_lag > df.kama2_lag))]

    # Creating signal, position and trend columns
    df['trend'] = _np.select(condlist=conditions, choicelist=[1, -1, 1, -1])
    df['signal'] = _np.select(condlist=conditions, choicelist=[0, 0, 1, -1])

    # Dropping lag features and NaN values and invalid values
    df.drop(['kama1_lag', 'kama2_lag'], axis=1, inplace=True)
    df.dropna(inplace=True)
    df = df.drop(df.index[0])

    # Returning dataframe
    if days is not None:
        if days >= len(df):
            pass
        else:
            df = df.iloc[len(df) - days:]
    return df.reset_index(drop=True)


def trima(prices, window1=20, window2=50, days=None):

    # In case not in good order
    if window1 > window2:
        window1, window2 = window2, window1

    # if not enough price data, we cannot compute
    if len(prices) <= window2 + 1:
        print('Not enough price data to compute trima')
        return None

    # Getting trima values
    trima1 = talib.TRIMA(prices.close.values, timeperiod=window1)
    trima2 = talib.TRIMA(prices.close.values, timeperiod=window2)

    # Creating dataframe
    df = _pd.DataFrame({'ico_id': prices.ico_id.values, 'symbol': prices.ico_symbol.values, 'date': prices.date.values, 'close': prices.close.values, 'trima1': trima1, 'trima2': trima2})

    # Getting lag values
    df['trima1_lag'] = df.trima1.shift()
    df['trima2_lag'] = df.trima2.shift()

    # Crossing conditions
    conditions = [((df.trima1 > df.trima2) & (df.trima1_lag > df.trima2_lag)),
                  ((df.trima1 < df.trima2) & (df.trima1_lag < df.trima2_lag)),
                  ((df.trima1 > df.trima2) & (df.trima1_lag < df.trima2_lag)),
                  ((df.trima1 < df.trima2) & (df.trima1_lag > df.trima2_lag))]

    # Creating signal, position and trend columns
    df['trend'] = _np.select(condlist=conditions, choicelist=[1, -1, 1, -1])
    df['signal'] = _np.select(condlist=conditions, choicelist=[0, 0, 1, -1])

    # Dropping lag features and NaN values and invalid values
    df.drop(['trima1_lag', 'trima2_lag'], axis=1, inplace=True)
    df.dropna(inplace=True)
    df = df.drop(df.index[0])

    # Returning dataframe
    if days is not None:
        if days >= len(df):
            pass
        else:
            df = df.iloc[len(df) - days:]
    return df.reset_index(drop=True)


def wma(prices, WMA1=20, WMA2=50, days=None):

    # In case not in good order
    if WMA1 > WMA2:
        WMA1, WMA2 = WMA2, WMA1

    # if not enough price data, we cannot compute
    if len(prices) <= WMA2 + 1:
        print('Not enough price data to compute WMA')
        return None

    # Getting wma values
    wma1 = talib.WMA(prices['close'].values, WMA1)
    wma2 = talib.WMA(prices['close'].values, WMA2)

    # Creating dataframe
    df = _pd.DataFrame({'ico_id': prices.ico_id.values, 'symbol': prices.ico_symbol.values, 'date': prices.date.values, 'close': prices.close.values, 'wma1': wma1, 'wma2': wma2})

    # Getting lag values
    df['wma1_lag'] = df.wma1.shift()
    df['wma2_lag'] = df.wma2.shift()

    # Crossing conditions
    conditions = [((df.wma1 > df.wma2) & (df.wma1_lag > df.wma2_lag)),
                  ((df.wma1 < df.wma2) & (df.wma1_lag < df.wma2_lag)),
                  ((df.wma1 > df.wma2) & (df.wma1_lag < df.wma2_lag)),
                  ((df.wma1 < df.wma2) & (df.wma1_lag > df.wma2_lag))]

    # Creating signal, position and trend columns
    df['trend'] = _np.select(condlist=conditions, choicelist=[1, -1, 1, -1])
    df['signal'] = _np.select(condlist=conditions, choicelist=[0, 0, 1, -1])

    # Dropping lag features and NaN values and invalid values
    df.drop(['wma1_lag', 'wma2_lag'], axis=1, inplace=True)
    df.dropna(inplace=True)
    df = df.drop(df.index[0])

    # Returning dataframe
    if days is not None:
        if days >= len(df):
            pass
        else:
            df = df.iloc[len(df) - days:]
    return df.reset_index(drop=True)


############################
# Momentum Indicators
############################


def adxr(prices, window1=14, window2=10, days=None):

    # Getting adx and adxr values
    adx = talib.ADX(prices.high.values, prices.low.values, prices.close.values, timeperiod=window1)
    adxr = talib.ADXR(prices.high.values, prices.low.values, prices.close.values, timeperiod=window2)

    # Creating dataframe
    df = _pd.DataFrame({'ico_id': prices.ico_id.values, 'symbol': prices.ico_symbol.values, 'date': prices.date.values, 'close': prices.close.values, 'adx': adx, 'adxr': adxr})

    # Getting lag values
    df['adx_lag'] = df.adx.shift()
    df['adxr_lag'] = df.adxr.shift()

    # Crossing conditions
    conditions = [((df.adx > df.adxr) & (df.adx_lag > df.adxr_lag)),
                  ((df.adx < df.adxr) & (df.adx_lag < df.adxr_lag)),
                  ((df.adx > df.adxr) & (df.adx_lag < df.adxr_lag)),
                  ((df.adx < df.adxr) & (df.adx_lag > df.adxr_lag))]

    # Creating signal, position and trend columns
    df['trend'] = _np.select(condlist=conditions, choicelist=[1, -1, 1, -1])
    df['signal'] = _np.select(condlist=conditions, choicelist=[0, 0, 1, -1])

    # Dropping lag features and NaN values and invalid values
    df.drop(['adxr_lag', 'adx_lag'], axis=1, inplace=True)
    df.dropna(inplace=True)
    df = df.drop(df.index[0])

    # Returning dataframe
    if days is not None:
        if days >= len(df):
            pass
        else:
            df = df.iloc[len(df) - days:]
    return df.reset_index(drop=True)


def apo(prices, fastperiod=12, slowperiod=26, days=None):

    # if not enough price data, we cannot compute
    if len(prices) <= slowperiod + 1:
        print('Not enough price data to compute APO')
        return None

    # Getting apo values
    apo = talib.APO(prices.close.values, fastperiod=fastperiod, slowperiod=slowperiod, matype=0)

    # Creating dataframe
    df = _pd.DataFrame({'ico_id': prices.ico_id.values, 'symbol': prices.ico_symbol.values, 'date': prices.date.values, 'close': prices.close.values, 'apo': apo})

    # Getting lag values
    df['apo_lag'] = df.apo.shift()

    # Crossing conditions
    conditions = [((df.apo < 0) & (df.apo_lag < 0)),
                  ((df.apo >= 0) & (df.apo_lag >= 0)),
                  ((df.apo < 0) & (df.apo_lag > 0)),
                  ((df.apo > 0) & (df.apo_lag < 0))]

    # Creating signal, position and trend columns
    df['trend'] = _np.select(condlist=conditions, choicelist=[-1, 1, -1, 1])
    df['signal'] = _np.select(condlist=conditions, choicelist=[0, 0, -1, 1])

    # Dropping lag features and NaN values and invalid values
    df.drop(['apo_lag'], axis=1, inplace=True)
    df.dropna(inplace=True)
    df = df.drop(df.index[0])

    # Returning dataframe
    if days is not None:
        if days >= len(df):
            pass
        else:
            df = df.iloc[len(df) - days:]
    return df.reset_index(drop=True)


def bop(prices, window=14, days=None):

    # if not enough price data, we cannot compute
    if len(prices) <= window + 1:
        print('Not enough price data to compute BOP')
        return None

    # Getting bop values
    bop_1 = talib.BOP(prices.open.values, prices.high.values, prices.low.values, prices.close.values)
    bop = talib.SMA(bop_1, window)

    # Creating dataframe
    df = _pd.DataFrame({'ico_id': prices.ico_id.values, 'symbol': prices.ico_symbol.values, 'date': prices.date.values, 'close': prices.close.values, 'bop': bop})

    # Getting lag values
    df['bop_lag'] = df.bop.shift()

    # Crossing conditions
    conditions = [((df.bop < 0) & (df.bop_lag < 0)),
                  ((df.bop >= 0) & (df.bop_lag >= 0)),
                  ((df.bop < 0) & (df.bop_lag > 0)),
                  ((df.bop > 0) & (df.bop_lag < 0))]

    # Creating signal, position and trend columns
    df['trend'] = _np.select(condlist=conditions, choicelist=[-1, 1, -1, 1])
    df['signal'] = _np.select(condlist=conditions, choicelist=[0, 0, -1, 1])

    # Dropping lag features and NaN values and invalid values
    df.drop('bop_lag', axis=1, inplace=True)
    df.dropna(inplace=True)
    df = df.drop(df.index[0])

    # Returning dataframe
    if days is not None:
        if days >= len(df):
            pass
        else:
            df = df.iloc[len(df) - days:]
    return df.reset_index(drop=True)


def cmo(prices, window1=21, window2=50, days=None):

    # In case not in good order
    if window1 > window2:
        window1, window2 = window2, window1

    # if not enough price data, we cannot compute
    if len(prices) <= window2 + 2:
        print('Not enough price data to compute CMO')
        return None

    # Getting cmo values
    cmo1 = talib.CMO(prices['close'].values, window1)
    cmo2 = talib.CMO(prices['close'].values, window2)

    # Creating dataframe
    df = _pd.DataFrame({'ico_id': prices.ico_id.values, 'symbol': prices.ico_symbol.values, 'date': prices.date.values, 'close': prices.close.values, 'cmo1': cmo1, 'cmo2': cmo2})

    # Getting lag values
    df['cmo1_lag'] = df.cmo1.shift()
    df['cmo2_lag'] = df.cmo2.shift()

    # Crossing conditions
    conditions_crossing = [((df.cmo1 > df.cmo2) & (df.cmo1_lag > df.cmo2_lag)),
                           ((df.cmo1 < df.cmo2) & (df.cmo1_lag < df.cmo2_lag)),
                           ((df.cmo1 > df.cmo2) & (df.cmo1_lag < df.cmo2_lag)),
                           ((df.cmo1 < df.cmo2) & (df.cmo1_lag > df.cmo2_lag))]

    conditions_value = [(df.cmo1 > 50),
                        (df.cmo1 < -50)]

    # Creating signal, position and trend columns
    df['trend'] = _np.select(condlist=conditions_crossing, choicelist=[1, -1, 1, -1], default=None)
    df['signal'] = _np.select(condlist=conditions_crossing, choicelist=[0, 0, 1, -1], default=None)

    df['trend'] = _np.select(condlist=conditions_value, choicelist=[-1, 1], default=df.trend)
    df['signal'] = _np.select(condlist=conditions_value, choicelist=[0, 0], default=df.signal)

    # Dropping lag features and NaN values and invalid values
    df.drop(['cmo1_lag', 'cmo2_lag'], axis=1, inplace=True)
    df.dropna(inplace=True)
    df = df.drop(df.index[0])

    # Returning dataframe
    if days is not None:
        if days >= len(df):
            pass
        else:
            df = df.iloc[len(df) - days:]
    return df.reset_index(drop=True)


def cci(prices, window=14, days=None):

    # if not enough price data, we cannot compute
    if len(prices) <= window + 1:
        print('Not enough price data to compute CCI')
        return None

    # Getting cci values
    cci = talib.CCI(prices.high.values, prices.low.values, prices.close.values, timeperiod=window)

    # Creating dataframe
    df = _pd.DataFrame({'ico_id': prices.ico_id.values, 'symbol': prices.ico_symbol.values, 'date': prices.date.values, 'close': prices.close.values, 'cci': cci})

    # Getting lag values
    df['cci_lag'] = df.cci.shift()

    # Crossing conditions
    conditions = [((df.cci > 100) & (df.cci_lag < 20)),
                  ((df.cci < -100) & (df.cci_lag > -20)),
                  ((df.cci > 75) & (df.cci < 250)),
                  ((df.cci < -75) & (df.cci > -250)),
                  (df.cci > 250),
                  (df.cci < -250)]

    # Creating signal, position and trend columns
    df['cci_trends'] = _np.select(condlist=conditions, choicelist=['up', 'down', 'up', 'down', 'reversal', 'reversal'], default='neutral')
    df['trend'] = _np.select(condlist=conditions, choicelist=[1, -1, 1, -1, -1, 1], default=0)
    df['signal'] = _np.select(condlist=conditions, choicelist=[1, -1, 0, 0, 0, 0], default=0)

    # Dropping lag features and NaN values and invalid values
    df.drop('cci_lag', axis=1, inplace=True)
    df.dropna(inplace=True)
    df = df.drop(df.index[0])

    # Returning dataframe
    if days is not None:
        if days >= len(df):
            pass
        else:
            df = df.iloc[len(df) - days:]
    return df.reset_index(drop=True)


def mom(prices, window=26, days=None):

    # if not enough price data, we cannot compute
    if len(prices) <= window + 1:
        print('Not enough price data to compute MOM')
        return None

    # Getting mom values
    mom = talib.MOM(prices.close.values, timeperiod=window)

    # Creating dataframe
    df = _pd.DataFrame({'ico_id': prices.ico_id.values, 'symbol': prices.ico_symbol.values, 'date': prices.date.values, 'close': prices.close.values, 'mom': mom})

    # Getting lag values
    df['mom_lag'] = df.mom.shift()

    # Crossing conditions
    conditions = [((df.mom < 0) & (df.mom_lag < 0)),
                  ((df.mom >= 0) & (df.mom_lag >= 0)),
                  ((df.mom < 0) & (df.mom_lag > 0)),
                  ((df.mom > 0) & (df.mom_lag < 0))]

    # Creating signal, position and trend columns
    df['trend'] = _np.select(condlist=conditions, choicelist=[-1, 1, -1, 1])
    df['signal'] = _np.select(condlist=conditions, choicelist=[0, 0, -1, 1])

    # Dropping lag features and NaN values and invalid values
    df.drop('mom_lag', axis=1, inplace=True)
    df.dropna(inplace=True)
    df = df.drop(df.index[0])

    # Returning dataframe
    if days is not None:
        if days >= len(df):
            pass
        else:
            df = df.iloc[len(df) - days:]
    return df.reset_index(drop=True)


def ppo(prices, window1=12, window2=26, days=None):

    # In case not in good order
    if window1 > window2:
        window1, window2 = window2, window1

    # if not enough price data, we cannot compute
    if len(prices) <= window2 + 1:
        print('Not enough price data to compute PPO')
        return None

    # Getting ppo values
    ppo = talib.PPO(prices.close.values, fastperiod=window1, slowperiod=window2, matype=0)

    # Creating dataframe
    df = _pd.DataFrame({'ico_id': prices.ico_id.values, 'symbol': prices.ico_symbol.values, 'date': prices.date.values, 'close': prices.close.values, 'ppo': ppo})

    # Getting lag values
    df['ppo_lag'] = df.ppo.shift()

    # Crossing conditions
    conditions = [((df.ppo < 0) & (df.ppo_lag < 0)),
                  ((df.ppo >= 0) & (df.ppo_lag >= 0)),
                  ((df.ppo < 0) & (df.ppo_lag > 0)),
                  ((df.ppo > 0) & (df.ppo_lag < 0))]

    # Creating signal, position and trend columns
    df['trend'] = _np.select(condlist=conditions, choicelist=[-1, 1, -1, 1])
    df['signal'] = _np.select(condlist=conditions, choicelist=[0, 0, -1, 1])

    # Dropping lag features and NaN values and invalid values
    df.drop('ppo_lag', axis=1, inplace=True)
    df.dropna(inplace=True)
    df = df.drop(df.index[0])

    # Returning dataframe
    if days is not None:
        if days >= len(df):
            pass
        else:
            df = df.iloc[len(df) - days:]
    return df.reset_index(drop=True)


def roc(prices, window=26, days=None):

    # if not enough price data, we cannot compute
    if len(prices) <= window + 2:
        print('Not enough price data to compute ROC')
        return None

    # Getting roc values
    roc = talib.ROC(prices.close.values, timeperiod=window)

    # Creating dataframe
    df = _pd.DataFrame({'ico_id': prices.ico_id.values, 'symbol': prices.ico_symbol.values, 'date': prices.date.values, 'close': prices.close.values, 'roc': roc})

    # Getting lag values
    df['roc_lag'] = df.roc.shift()

    # Crossing conditions
    conditions = [((df.roc < 0) & (df.roc_lag < 0)),
                  ((df.roc >= 0) & (df.roc_lag >= 0)),
                  ((df.roc < 0) & (df.roc_lag > 0)),
                  ((df.roc > 0) & (df.roc_lag < 0))]

    # Creating signal, position and trend columns
    df['trend'] = _np.select(condlist=conditions, choicelist=[-1, 1, -1, 1])
    df['signal'] = _np.select(condlist=conditions, choicelist=[0, 0, -1, 1])

    # Dropping lag features and NaN values and invalid values
    df.drop('roc_lag', axis=1, inplace=True)
    df.dropna(inplace=True)
    df = df.drop(df.index[0])

    # Returning dataframe
    if days is not None:
        if days >= len(df):
            pass
        else:
            df = df.iloc[len(df) - days:]
    return df.reset_index(drop=True)


def rsi(prices, window=14, days=None):

    # if not enough price data, we cannot compute
    if len(prices) <= window + 2:
        print('Not enough price data to compute RSI')
        return None

    # Getting rsi values
    rsi = talib.RSI(prices.close.values, timeperiod=window)

    # Creating dataframe
    df = _pd.DataFrame({'ico_id': prices.ico_id.values, 'symbol': prices.ico_symbol.values, 'date': prices.date.values, 'close': prices.close.values, 'rsi': rsi})

    # Getting lag values
    df['rsi_lag'] = df.rsi.shift()

    # Crossing conditions
    conditions = [((df.rsi <= 30) & (df.rsi_lag <= 30)),
                  ((df.rsi >= 70) & (df.rsi_lag >= 70)),
                  ((df.rsi > 70) & (df.rsi_lag < 70)),
                  ((df.rsi < 30) & (df.rsi_lag > 30)),
                  ((df.rsi >= 30) & (df.rsi <= 70))]

    # Creating signal, position and trend columns
    df['trend'] = _np.select(condlist=conditions, choicelist=[1, -1, -1, 1, 0])
    df['signal'] = _np.select(condlist=conditions, choicelist=[0, 0, -1, 1, 0])

    # Dropping lag features and NaN values and invalid values
    df.drop('rsi_lag', axis=1, inplace=True)
    df.dropna(inplace=True)
    df = df.drop(df.index[0])

    # Returning dataframe
    if days is not None:
        if days >= len(df):
            pass
        else:
            df = df.iloc[len(df) - days:]
    return df.reset_index(drop=True)


def adosc(prices, window1=6, window2=20, days=None):

    # In case not in good order
    if window1 > window2:
        window1, window2 = window2, window1

    # if not enough price data, we cannot compute
    if len(prices) <= window2 + 1:
        print('Not enough price data to compute ADOSC')
        return None

    # Getting adosc values
    adosc = talib.ADOSC(prices.high.values, prices.low.values, prices.close.values, prices.volume.values, fastperiod=window1, slowperiod=window2)

    # Creating dataframe
    df = _pd.DataFrame({'ico_id': prices.ico_id.values, 'symbol': prices.ico_symbol.values, 'date': prices.date.values, 'close': prices.close.values, 'adosc': adosc})

    # Getting lag values
    df['adosc_lag'] = df.adosc.shift()

    # Crossing conditions
    conditions = [((df.adosc < 0) & (df.adosc_lag < 0)),
                  ((df.adosc >= 0) & (df.adosc_lag >= 0)),
                  ((df.adosc < 0) & (df.adosc_lag > 0)),
                  ((df.adosc > 0) & (df.adosc_lag < 0))]

    # Creating signal, position and trend columns
    df['trend'] = _np.select(condlist=conditions, choicelist=[-1, 1, -1, 1], default=0)
    df['signal'] = _np.select(condlist=conditions, choicelist=[0, 0, -1, 1])

    # Dropping lag features and NaN values and invalid values
    df.drop('adosc_lag', axis=1, inplace=True)
    df.dropna(inplace=True)
    df = df.drop(df.index[0])

    # Returning dataframe
    if days is not None:
        if days >= len(df):
            pass
        else:
            df = df.iloc[len(df) - days:]
    return df.reset_index(drop=True)


############################
# Overbought/sold Indicators
############################


def willr(prices, window=52, days=None):

    # if not enough price data, we cannot compute
    if len(prices) <= window + 1:
        print('Not enough price data to compute WILLR')
        return None

    # Getting willr values
    willr = talib.WILLR(prices.high.values, prices.low.values, prices.close.values, timeperiod=window)

    # Creating dataframe
    df = _pd.DataFrame({'ico_id': prices.ico_id.values, 'symbol': prices.ico_symbol.values, 'date': prices.date.values, 'close': prices.close.values, 'willr': willr})

    # Crossing conditions
    conditions = [(df.willr < -80),
                  (df.willr > -20)]

    # Creating signal, position and trend columns
    df['trend'] = _np.select(condlist=conditions, choicelist=[1, -1], default=0)
    df['signal'] = _np.select(condlist=conditions, choicelist=[0, 0], default=0)

    # Dropping lag features and NaN values and invalid values
    df.dropna(inplace=True)
    df = df.drop(df.index[0])

    # Returning dataframe
    if days is not None:
        if days >= len(df):
            pass
        else:
            df = df.iloc[len(df) - days:]
    return df.reset_index(drop=True)


############################
# Trend strength Indicators
############################


def adx(prices, window=14, days=None):

    # if not enough price data, we cannot compute
    if len(prices) <= window * 2 + 1:
        print('Not enough price data to compute ADX')
        return None

    # Getting adx values
    adx = talib.ADX(prices.high.values, prices.low.values, prices.close.values, timeperiod=window)

    # Creating dataframe
    df = _pd.DataFrame({'ico_id': prices.ico_id.values, 'symbol': prices.ico_symbol.values, 'date': prices.date.values, 'close': prices.close.values, 'adx': adx})

    # Getting lag values
    df['adx_lag'] = df.adx.shift()

    # Crossing conditions
    conditions_strengh = [(df.adx < 20),
                          ((df.adx > 20) & (df.adx < 50)),
                          (df.adx > 50)]

    conditions_trend = [(df.adx > df.adx_lag),
                        (df.adx < df.adx_lag)]

    # Creating strength and change columns
    df['trend_strength'] = _np.select(condlist=conditions_strengh, choicelist=['weak', 'strong', 'extreme'])
    df['trend_change'] = _np.select(condlist=conditions_trend, choicelist=['strenghening', 'weakening'])

    # Dropping lag features and NaN values and invalid values
    df.drop(['adx_lag'], axis=1, inplace=True)
    df.dropna(inplace=True)
    df = df[df.trend_change != '0']

    # Returning dataframe
    if days is not None:
        if days >= len(df):
            pass
        else:
            df = df.iloc[len(df) - days:]
    return df.reset_index(drop=True)


############################
# Volatility Indicators
############################

def atr(prices, window1=12, window2=52, days=None):

    # In case not in good order
    if window1 > window2:
        window1, window2 = window2, window1

    # if not enough price data, we cannot compute
    if len(prices) <= window2 + 2:
        print('Not enough price data to compute ATR')
        return None

    # Getting atr values
    atr1 = talib.ATR(prices.high.values, prices.low.values, prices.close.values, timeperiod=window1)
    atr2 = talib.ATR(prices.high.values, prices.low.values, prices.close.values, timeperiod=window2)

    # Creating dataframe
    df = _pd.DataFrame({'ico_id': prices.ico_id.values, 'symbol': prices.ico_symbol.values, 'date': prices.date.values, 'close': prices.close.values, 'atr1': atr1, 'atr2': atr2})

    # Getting lag values
    df['atr1_lag'] = df.atr1.shift()
    df['atr2_lag'] = df.atr2.shift()

    # Crossing conditions
    conditions = [((df.atr1 > df.atr2) & (df.atr1_lag > df.atr2_lag)),
                  ((df.atr1 < df.atr2) & (df.atr1_lag < df.atr2_lag)),
                  ((df.atr1 > df.atr2) & (df.atr1_lag < df.atr2_lag)),
                  ((df.atr1 < df.atr2) & (df.atr1_lag > df.atr2_lag))]

    # Creating signal, position and trend columns
    df['volatility_trend'] = _np.select(condlist=conditions, choicelist=['high', 'low', 'spike_up', 'decreasing'])
    df['volatility_breakout'] = _np.select(condlist=conditions, choicelist=[False, False, True, False])

    # Dropping lag features and NaN values and invalid values
    df.drop(['atr1_lag', 'atr2_lag'], axis=1, inplace=True)
    df.dropna(inplace=True)
    df = df[df.volatility_trend != '0']

    # Returning dataframe
    if days is not None:
        if days >= len(df):
            pass
        else:
            df = df.iloc[len(df) - days:]
    return df.reset_index(drop=True)


def plot_volatility(vol_df, figsize=(17, 10)):

    # _plt.style.use('fivethirtyeight')
    # _plt.style.use("dark_background")
    # _plt.rcParams['grid.color'] = '#3c4f6e'

    # Creating figures
    fig, axes = _plt.subplots(2, figsize=figsize, sharex=True)

    # Plotting main price line
    vol_df.plot(x='date', y='close', figsize=figsize, linewidth=1, label='price', color='slategray', ax=axes[0])

    # Annotating signals
    volatility_spikes = vol_df[vol_df.volatility_trend == 'spike_up'].copy()
    volatility_drops = vol_df[vol_df.volatility_trend == 'decreasing'].copy()

    # Annotating spikes
    axes[0].scatter(volatility_spikes.date.values, volatility_spikes.close.values, color='lemonchiffon', s=50)
    for i in range(len(volatility_spikes)):
        axes[0].annotate('Spike', xy=(volatility_spikes.date.values[i], volatility_spikes.close.values[i]),
                         textcoords='offset points', xytext=(0, 13), color='black',
                         bbox=dict(boxstyle="Round", fc="lemonchiffon", alpha=.8), horizontalalignment='center', verticalalignment='bottom')

    # Annotating drops
    axes[0].scatter(volatility_drops.date.values, volatility_drops.close.values, color='darkorange', s=50)
    for i in range(len(volatility_drops)):
        axes[0].annotate('Drop', xy=(volatility_drops.date.values[i], volatility_drops.close.values[i]),
                         textcoords='offset points', xytext=(0, -24), color='white',
                         bbox=dict(boxstyle="Round", fc="darkorange", alpha=.8), horizontalalignment='center', verticalalignment='bottom')

    # Plotting the atr clouds
    # Moving averages cloud - High vol
    axes[1].fill_between(vol_df.date.values, vol_df['atr1'].values, vol_df['atr2'].values,
                         color='lemonchiffon', alpha=.7, where=vol_df['atr1'] > vol_df['atr2'], label='high volatility')

    # Moving averages cloud - Low vol
    axes[1].fill_between(vol_df.date.values, vol_df['atr1'].values, vol_df['atr2'].values,
                         color='darkorange', alpha=.7, where=vol_df['atr1'] < vol_df['atr2'], label='low volatility')

    # Titles
    fig.suptitle('Volatility Indicator', fontsize=20)
    axes[0].set_title('Price and signals', fontsize=15)
    axes[1].set_title('ATR clouds', fontsize=15)
    axes[1].legend()
    _plt.show()


############################
# Candlestick indicators
############################

def signal_candle(indicator):
    """
    return signal based on indicator value
    """
    if indicator == 100:
        return 1
    if indicator == -100:
        return -1
    if indicator == 0:
        return 0


def CDL2CROWS(prices, days=None):
    CDL2CROWS = talib.CDL2CROWS(prices.open.values, prices.high.values, prices.low.values, prices.close.values)
    df = prices.copy()
    df['CDL2CROWS'] = CDL2CROWS
    df = df[['ico_id', 'ico_symbol', 'date', 'close', 'CDL2CROWS']]
    df['signal'] = df['CDL2CROWS'].apply(signal_candle)
    if days:
        return df.iloc[len(df)-days:]
    return df


def CDL3BLACKCROWS(prices, days=None):
    CDL3BLACKCROWS = talib.CDL3BLACKCROWS(prices.open.values, prices.high.values, prices.low.values, prices.close.values)
    df = prices.copy()
    df['CDL3BLACKCROWS'] = CDL3BLACKCROWS
    df = df[['ico_id', 'ico_symbol', 'date', 'close', 'CDL3BLACKCROWS']]
    df['signal'] = df['CDL3BLACKCROWS'].apply(signal_candle)
    if days:
        return df.iloc[len(df)-days:]
    return df


def CDL3INSIDE(prices, days=None):
    CDL3INSIDE = talib.CDL3INSIDE(prices.open.values, prices.high.values, prices.low.values, prices.close.values)
    df = prices.copy()
    df['CDL3INSIDE'] = CDL3INSIDE
    df = df[['ico_id', 'ico_symbol', 'date', 'close', 'CDL3INSIDE']]
    df['signal'] = df['CDL3INSIDE'].apply(signal_candle)
    if days:
        return df.iloc[len(df)-days:]
    return df


def CDL3LINESTRIKE(prices, days=None):
    CDL3LINESTRIKE = talib.CDL3LINESTRIKE(prices.open.values, prices.high.values, prices.low.values, prices.close.values)
    df = prices.copy()
    df['CDL3LINESTRIKE'] = CDL3LINESTRIKE
    df = df[['ico_id', 'ico_symbol', 'date', 'close', 'CDL3LINESTRIKE']]
    df['signal'] = df['CDL3LINESTRIKE'].apply(signal_candle)
    if days:
        return df.iloc[len(df)-days:]
    return df


def CDL3OUTSIDE(prices, days=None):
    CDL3OUTSIDE = talib.CDL3OUTSIDE(prices.open.values, prices.high.values, prices.low.values, prices.close.values)
    df = prices.copy()
    df['CDL3OUTSIDE'] = CDL3OUTSIDE
    df = df[['ico_id', 'ico_symbol', 'date', 'close', 'CDL3OUTSIDE']]
    df['signal'] = df['CDL3OUTSIDE'].apply(signal_candle)
    if days:
        return df.iloc[len(df)-days:]
    return df


def CDL3STARSINSOUTH(prices, days=None):
    CDL3STARSINSOUTH = talib.CDL3STARSINSOUTH(prices.open.values, prices.high.values, prices.low.values, prices.close.values)
    df = prices.copy()
    df['CDL3STARSINSOUTH'] = CDL3STARSINSOUTH
    df = df[['ico_id', 'ico_symbol', 'date', 'close', 'CDL3STARSINSOUTH']]
    df['signal'] = df['CDL3STARSINSOUTH'].apply(signal_candle)
    if days:
        return df.iloc[len(df)-days:]
    return df


def CDL3WHITESOLDIERS(prices, days=None):
    CDL3WHITESOLDIERS = talib.CDL3WHITESOLDIERS(prices.open.values, prices.high.values, prices.low.values, prices.close.values)
    df = prices.copy()
    df['CDL3WHITESOLDIERS'] = CDL3WHITESOLDIERS
    df = df[['ico_id', 'ico_symbol', 'date', 'close', 'CDL3WHITESOLDIERS']]
    df['signal'] = df['CDL3WHITESOLDIERS'].apply(signal_candle)
    if days:
        return df.iloc[len(df)-days:]
    return df


def CDLABANDONEDBABY(prices, days=None):
    CDLABANDONEDBABY = talib.CDLABANDONEDBABY(prices.open.values, prices.high.values, prices.low.values, prices.close.values, penetration=0)
    df = prices.copy()
    df['CDLABANDONEDBABY'] = CDLABANDONEDBABY
    df = df[['ico_id', 'ico_symbol', 'date', 'close', 'CDLABANDONEDBABY']]
    df['signal'] = df['CDLABANDONEDBABY'].apply(signal_candle)
    if days:
        return df.iloc[len(df)-days:]
    return df


def CDLADVANCEBLOCK(prices, days=None):
    CDLADVANCEBLOCK = talib.CDLADVANCEBLOCK(prices.open.values, prices.high.values, prices.low.values, prices.close.values)
    df = prices.copy()
    df['CDLADVANCEBLOCK'] = CDLADVANCEBLOCK
    df = df[['ico_id', 'ico_symbol', 'date', 'close', 'CDLADVANCEBLOCK']]
    df['signal'] = df['CDLADVANCEBLOCK'].apply(signal_candle)
    if days:
        return df.iloc[len(df)-days:]
    return df


def CDLBELTHOLD(prices, days=None):
    CDLBELTHOLD = talib.CDLBELTHOLD(prices.open.values, prices.high.values, prices.low.values, prices.close.values)
    df = prices.copy()
    df['CDLBELTHOLD'] = CDLBELTHOLD
    df = df[['ico_id', 'ico_symbol', 'date', 'close', 'CDLBELTHOLD']]
    df['signal'] = df['CDLBELTHOLD'].apply(signal_candle)
    if days:
        return df.iloc[len(df)-days:]
    return df


def CDLBREAKAWAY(prices, days=None):
    CDLBREAKAWAY = talib.CDLBREAKAWAY(prices.open.values, prices.high.values, prices.low.values, prices.close.values)
    df = prices.copy()
    df['CDLBREAKAWAY'] = CDLBREAKAWAY
    df = df[['ico_id', 'ico_symbol', 'date', 'close', 'CDLBREAKAWAY']]
    df['signal'] = df['CDLBREAKAWAY'].apply(signal_candle)
    if days:
        return df.iloc[len(df)-days:]
    return df


def CDLCLOSINGMARUBOZU(prices, days=None):
    CDLCLOSINGMARUBOZU = talib.CDLCLOSINGMARUBOZU(prices.open.values, prices.high.values, prices.low.values, prices.close.values)
    df = prices.copy()
    df['CDLCLOSINGMARUBOZU'] = CDLCLOSINGMARUBOZU
    df = df[['ico_id', 'ico_symbol', 'date', 'close', 'CDLCLOSINGMARUBOZU']]
    df['signal'] = df['CDLCLOSINGMARUBOZU'].apply(signal_candle)
    if days:
        return df.iloc[len(df)-days:]
    return df


def CDLCONCEALBABYSWALL(prices, days=None):
    CDLCONCEALBABYSWALL = talib.CDLCONCEALBABYSWALL(prices.open.values, prices.high.values, prices.low.values, prices.close.values)
    df = prices.copy()
    df['CDLCONCEALBABYSWALL'] = CDLCONCEALBABYSWALL
    df = df[['ico_id', 'ico_symbol', 'date', 'close', 'CDLCONCEALBABYSWALL']]
    df['signal'] = df['CDLCONCEALBABYSWALL'].apply(signal_candle)
    if days:
        return df.iloc[len(df)-days:]
    return df


def CDLCOUNTERATTACK(prices, days=None):
    CDLCOUNTERATTACK = talib.CDLCOUNTERATTACK(prices.open.values, prices.high.values, prices.low.values, prices.close.values)
    df = prices.copy()
    df['CDLCOUNTERATTACK'] = CDLCOUNTERATTACK
    df = df[['ico_id', 'ico_symbol', 'date', 'close', 'CDLCOUNTERATTACK']]
    df['signal'] = df['CDLCOUNTERATTACK'].apply(signal_candle)
    if days:
        return df.iloc[len(df)-days:]
    return df


def CDLDARKCLOUDCOVER(prices, days=None):
    CDLDARKCLOUDCOVER = talib.CDLDARKCLOUDCOVER(prices.open.values, prices.high.values, prices.low.values, prices.close.values)
    df = prices.copy()
    df['CDLDARKCLOUDCOVER'] = CDLDARKCLOUDCOVER
    df = df[['ico_id', 'ico_symbol', 'date', 'close', 'CDLDARKCLOUDCOVER']]
    df['signal'] = df['CDLDARKCLOUDCOVER'].apply(signal_candle)
    if days:
        return df.iloc[len(df)-days:]
    return df


def CDLDOJI(prices, days=None):
    CDLDOJI = talib.CDLDOJI(prices.open.values, prices.high.values, prices.low.values, prices.close.values)
    df = prices.copy()
    df['CDLDOJI'] = CDLDOJI
    df = df[['ico_id', 'ico_symbol', 'date', 'close', 'CDLDOJI']]
    df['signal'] = df['CDLDOJI'].apply(signal_candle)
    if days:
        return df.iloc[len(df)-days:]
    return df


def CDLDOJISTAR(prices, days=None):
    CDLDOJISTAR = talib.CDLDOJISTAR(prices.open.values, prices.high.values, prices.low.values, prices.close.values)
    df = prices.copy()
    df['CDLDOJISTAR'] = CDLDOJISTAR
    df = df[['ico_id', 'ico_symbol', 'date', 'close', 'CDLDOJISTAR']]
    df['signal'] = df['CDLDOJISTAR'].apply(signal_candle)
    if days:
        return df.iloc[len(df)-days:]
    return df


def CDLDRAGONFLYDOJI(prices, days=None):
    CDLDRAGONFLYDOJI = talib.CDLDRAGONFLYDOJI(prices.open.values, prices.high.values, prices.low.values, prices.close.values)
    df = prices.copy()
    df['CDLDRAGONFLYDOJI'] = CDLDRAGONFLYDOJI
    df = df[['ico_id', 'ico_symbol', 'date', 'close', 'CDLDRAGONFLYDOJI']]
    df['signal'] = df['CDLDRAGONFLYDOJI'].apply(signal_candle)
    if days:
        return df.iloc[len(df)-days:]
    return df


def CDLENGULFING(prices, days=None):
    CDLENGULFING = talib.CDLENGULFING(prices.open.values, prices.high.values, prices.low.values, prices.close.values)
    df = prices.copy()
    df['CDLENGULFING'] = CDLENGULFING
    df = df[['ico_id', 'ico_symbol', 'date', 'close', 'CDLENGULFING']]
    df['signal'] = df['CDLENGULFING'].apply(signal_candle)
    if days:
        return df.iloc[len(df)-days:]
    return df


def CDLEVENINGDOJISTAR(prices, days=None):
    CDLEVENINGDOJISTAR = talib.CDLEVENINGDOJISTAR(prices.open.values, prices.high.values, prices.low.values, prices.close.values, penetration=0)
    df = prices.copy()
    df['CDLEVENINGDOJISTAR'] = CDLEVENINGDOJISTAR
    df = df[['ico_id', 'ico_symbol', 'date', 'close', 'CDLEVENINGDOJISTAR']]
    df['signal'] = df['CDLEVENINGDOJISTAR'].apply(signal_candle)
    if days:
        return df.iloc[len(df)-days:]
    return df


def CDLEVENINGSTAR(prices, days=None):
    CDLEVENINGSTAR = talib.CDLEVENINGSTAR(prices.open.values, prices.high.values, prices.low.values, prices.close.values, penetration=0)
    df = prices.copy()
    df['CDLEVENINGSTAR'] = CDLEVENINGSTAR
    df = df[['ico_id', 'ico_symbol', 'date', 'close', 'CDLEVENINGSTAR']]
    df['signal'] = df['CDLEVENINGSTAR'].apply(signal_candle)
    if days:
        return df.iloc[len(df)-days:]
    return df


def CDLGAPSIDESIDEWHITE(prices, days=None):
    CDLGAPSIDESIDEWHITE = talib.CDLGAPSIDESIDEWHITE(prices.open.values, prices.high.values, prices.low.values, prices.close.values)
    df = prices.copy()
    df['CDLGAPSIDESIDEWHITE'] = CDLGAPSIDESIDEWHITE
    df = df[['ico_id', 'ico_symbol', 'date', 'close', 'CDLGAPSIDESIDEWHITE']]
    df['signal'] = df['CDLGAPSIDESIDEWHITE'].apply(signal_candle)
    if days:
        return df.iloc[len(df)-days:]
    return df


def CDLGRAVESTONEDOJI(prices, days=None):
    CDLGRAVESTONEDOJI = talib.CDLGRAVESTONEDOJI(prices.open.values, prices.high.values, prices.low.values, prices.close.values)
    df = prices.copy()
    df['CDLGRAVESTONEDOJI'] = CDLGRAVESTONEDOJI
    df = df[['ico_id', 'ico_symbol', 'date', 'close', 'CDLGRAVESTONEDOJI']]
    df['signal'] = df['CDLGRAVESTONEDOJI'].apply(signal_candle)
    if days:
        return df.iloc[len(df)-days:]
    return df


def CDLHAMMER(prices, days=None):
    CDLHAMMER = talib.CDLHAMMER(prices.open.values, prices.high.values, prices.low.values, prices.close.values)
    df = prices.copy()
    df['CDLHAMMER'] = CDLHAMMER
    df = df[['ico_id', 'ico_symbol', 'date', 'close', 'CDLHAMMER']]
    df['signal'] = df['CDLHAMMER'].apply(signal_candle)
    if days:
        return df.iloc[len(df)-days:]
    return df


def CDLHANGINGMAN(prices, days=None):
    CDLHANGINGMAN = talib.CDLHANGINGMAN(prices.open.values, prices.high.values, prices.low.values, prices.close.values)
    df = prices.copy()
    df['CDLHANGINGMAN'] = CDLHANGINGMAN
    df = df[['ico_id', 'ico_symbol', 'date', 'close', 'CDLHANGINGMAN']]
    df['signal'] = df['CDLHANGINGMAN'].apply(signal_candle)
    if days:
        return df.iloc[len(df)-days:]
    return df


def CDLHARAMI(prices, days=None):
    CDLHARAMI = talib.CDLHARAMI(prices.open.values, prices.high.values, prices.low.values, prices.close.values)
    df = prices.copy()
    df['CDLHARAMI'] = CDLHARAMI
    df = df[['ico_id', 'ico_symbol', 'date', 'close', 'CDLHARAMI']]
    df['signal'] = df['CDLHARAMI'].apply(signal_candle)
    if days:
        return df.iloc[len(df)-days:]
    return df


def CDLHARAMICROSS(prices, days=None):
    CDLHARAMICROSS = talib.CDLHARAMICROSS(prices.open.values, prices.high.values, prices.low.values, prices.close.values)
    df = prices.copy()
    df['CDLHARAMICROSS'] = CDLHARAMICROSS
    df = df[['ico_id', 'ico_symbol', 'date', 'close', 'CDLHARAMICROSS']]
    df['signal'] = df['CDLHARAMICROSS'].apply(signal_candle)
    if days:
        return df.iloc[len(df)-days:]
    return df


def CDLHIGHWAVE(prices, days=None):
    CDLHIGHWAVE = talib.CDLHIGHWAVE(prices.open.values, prices.high.values, prices.low.values, prices.close.values)
    df = prices.copy()
    df['CDLHIGHWAVE'] = CDLHIGHWAVE
    df = df[['ico_id', 'ico_symbol', 'date', 'close', 'CDLHIGHWAVE']]
    df['signal'] = df['CDLHIGHWAVE'].apply(signal_candle)
    if days:
        return df.iloc[len(df)-days:]
    return df


def CDLHIKKAKE(prices, days=None):
    CDLHIKKAKE = talib.CDLHIKKAKE(prices.open.values, prices.high.values, prices.low.values, prices.close.values)
    df = prices.copy()
    df['CDLHIKKAKE'] = CDLHIKKAKE
    df = df[['ico_id', 'ico_symbol', 'date', 'close', 'CDLHIKKAKE']]
    df['signal'] = df['CDLHIKKAKE'].apply(signal_candle)
    if days:
        return df.iloc[len(df)-days:]
    return df


def CDLHIKKAKEMOD(prices, days=None):
    CDLHIKKAKEMOD = talib.CDLHIKKAKEMOD(prices.open.values, prices.high.values, prices.low.values, prices.close.values)
    df = prices.copy()
    df['CDLHIKKAKEMOD'] = CDLHIKKAKEMOD
    df = df[['ico_id', 'ico_symbol', 'date', 'close', 'CDLHIKKAKEMOD']]
    df['signal'] = df['CDLHIKKAKEMOD'].apply(signal_candle)
    if days:
        return df.iloc[len(df)-days:]
    return df


def CDLHOMINGPIGEON(prices, days=None):
    CDLHOMINGPIGEON = talib.CDLHOMINGPIGEON(prices.open.values, prices.high.values, prices.low.values, prices.close.values)
    df = prices.copy()
    df['CDLHOMINGPIGEON'] = CDLHOMINGPIGEON
    df = df[['ico_id', 'ico_symbol', 'date', 'close', 'CDLHOMINGPIGEON']]
    df['signal'] = df['CDLHOMINGPIGEON'].apply(signal_candle)
    if days:
        return df.iloc[len(df)-days:]
    return df


def CDLIDENTICAL3CROWS(prices, days=None):
    CDLIDENTICAL3CROWS = talib.CDLIDENTICAL3CROWS(prices.open.values, prices.high.values, prices.low.values, prices.close.values)
    df = prices.copy()
    df['CDLIDENTICAL3CROWS'] = CDLIDENTICAL3CROWS
    df = df[['ico_id', 'ico_symbol', 'date', 'close', 'CDLIDENTICAL3CROWS']]
    df['signal'] = df['CDLIDENTICAL3CROWS'].apply(signal_candle)
    if days:
        return df.iloc[len(df)-days:]
    return df


def CDLINNECK(prices, days=None):
    CDLINNECK = talib.CDLINNECK(prices.open.values, prices.high.values, prices.low.values, prices.close.values)
    df = prices.copy()
    df['CDLINNECK'] = CDLINNECK
    df = df[['ico_id', 'ico_symbol', 'date', 'close', 'CDLINNECK']]
    df['signal'] = df['CDLINNECK'].apply(signal_candle)
    if days:
        return df.iloc[len(df)-days:]
    return df


def CDLINVERTEDHAMMER(prices, days=None):
    CDLINVERTEDHAMMER = talib.CDLINVERTEDHAMMER(prices.open.values, prices.high.values, prices.low.values, prices.close.values)
    df = prices.copy()
    df['CDLINVERTEDHAMMER'] = CDLINVERTEDHAMMER
    df = df[['ico_id', 'ico_symbol', 'date', 'close', 'CDLINVERTEDHAMMER']]
    df['signal'] = df['CDLINVERTEDHAMMER'].apply(signal_candle)
    if days:
        return df.iloc[len(df)-days:]
    return df


def CDLKICKING(prices, days=None):
    CDLKICKING = talib.CDLKICKING(prices.open.values, prices.high.values, prices.low.values, prices.close.values)
    df = prices.copy()
    df['CDLKICKING'] = CDLKICKING
    df = df[['ico_id', 'ico_symbol', 'date', 'close', 'CDLKICKING']]
    df['signal'] = df['CDLKICKING'].apply(signal_candle)
    if days:
        return df.iloc[len(df)-days:]
    return df


def CDLKICKINGBYLENGTH(prices, days=None):
    CDLKICKINGBYLENGTH = talib.CDLKICKINGBYLENGTH(prices.open.values, prices.high.values, prices.low.values, prices.close.values)
    df = prices.copy()
    df['CDLKICKINGBYLENGTH'] = CDLKICKINGBYLENGTH
    df = df[['ico_id', 'ico_symbol', 'date', 'close', 'CDLKICKINGBYLENGTH']]
    df['signal'] = df['CDLKICKINGBYLENGTH'].apply(signal_candle)
    if days:
        return df.iloc[len(df)-days:]
    return df


def CDLLADDERBOTTOM(prices, days=None):
    CDLLADDERBOTTOM = talib.CDLLADDERBOTTOM(prices.open.values, prices.high.values, prices.low.values, prices.close.values)
    df = prices.copy()
    df['CDLLADDERBOTTOM'] = CDLLADDERBOTTOM
    df = df[['ico_id', 'ico_symbol', 'date', 'close', 'CDLLADDERBOTTOM']]
    df['signal'] = df['CDLLADDERBOTTOM'].apply(signal_candle)
    if days:
        return df.iloc[len(df)-days:]
    return df


def CDLLONGLEGGEDDOJI(prices, days=None):
    CDLLONGLEGGEDDOJI = talib.CDLLONGLEGGEDDOJI(prices.open.values, prices.high.values, prices.low.values, prices.close.values)
    df = prices.copy()
    df['CDLLONGLEGGEDDOJI'] = CDLLONGLEGGEDDOJI
    df = df[['ico_id', 'ico_symbol', 'date', 'close', 'CDLLONGLEGGEDDOJI']]
    df['signal'] = df['CDLLONGLEGGEDDOJI'].apply(signal_candle)
    if days:
        return df.iloc[len(df)-days:]
    return df


def CDLLONGLINE(prices, days=None):
    CDLLONGLINE = talib.CDLLONGLINE(prices.open.values, prices.high.values, prices.low.values, prices.close.values)
    df = prices.copy()
    df['CDLLONGLINE'] = CDLLONGLINE
    df = df[['ico_id', 'ico_symbol', 'date', 'close', 'CDLLONGLINE']]
    df['signal'] = df['CDLLONGLINE'].apply(signal_candle)
    if days:
        return df.iloc[len(df)-days:]
    return df


def CDLMARUBOZU(prices, days=None):
    CDLMARUBOZU = talib.CDLMARUBOZU(prices.open.values, prices.high.values, prices.low.values, prices.close.values)
    df = prices.copy()
    df['CDLMARUBOZU'] = CDLMARUBOZU
    df = df[['ico_id', 'ico_symbol', 'date', 'close', 'CDLMARUBOZU']]
    df['signal'] = df['CDLMARUBOZU'].apply(signal_candle)
    if days:
        return df.iloc[len(df)-days:]
    return df


def CDLMATCHINGLOW(prices, days=None):
    CDLMATCHINGLOW = talib.CDLMATCHINGLOW(prices.open.values, prices.high.values, prices.low.values, prices.close.values)
    df = prices.copy()
    df['CDLMATCHINGLOW'] = CDLMATCHINGLOW
    df = df[['ico_id', 'ico_symbol', 'date', 'close', 'CDLMATCHINGLOW']]
    df['signal'] = df['CDLMATCHINGLOW'].apply(signal_candle)
    if days:
        return df.iloc[len(df)-days:]
    return df


def CDLMATHOLD(prices, days=None):
    CDLMATHOLD = talib.CDLMATHOLD(prices.open.values, prices.high.values, prices.low.values, prices.close.values, penetration=0)
    df = prices.copy()
    df['CDLMATHOLD'] = CDLMATHOLD
    df = df[['ico_id', 'ico_symbol', 'date', 'close', 'CDLMATHOLD']]
    df['signal'] = df['CDLMATHOLD'].apply(signal_candle)
    if days:
        return df.iloc[len(df)-days:]
    return df


def CDLMORNINGDOJISTAR(prices, days=None):
    CDLMORNINGDOJISTAR = talib.CDLMORNINGDOJISTAR(prices.open.values, prices.high.values, prices.low.values, prices.close.values, penetration=0)
    df = prices.copy()
    df['CDLMORNINGDOJISTAR'] = CDLMORNINGDOJISTAR
    df = df[['ico_id', 'ico_symbol', 'date', 'close', 'CDLMORNINGDOJISTAR']]
    df['signal'] = df['CDLMORNINGDOJISTAR'].apply(signal_candle)
    if days:
        return df.iloc[len(df)-days:]
    return df


def CDLMORNINGSTAR(prices, days=None):
    CDLMORNINGSTAR = talib.CDLMORNINGSTAR(prices.open.values, prices.high.values, prices.low.values, prices.close.values, penetration=0)
    df = prices.copy()
    df['CDLMORNINGSTAR'] = CDLMORNINGSTAR
    df = df[['ico_id', 'ico_symbol', 'date', 'close', 'CDLMORNINGSTAR']]
    df['signal'] = df['CDLMORNINGSTAR'].apply(signal_candle)
    if days:
        return df.iloc[len(df)-days:]
    return df


def CDLONNECK(prices, days=None):
    CDLONNECK = talib.CDLONNECK(prices.open.values, prices.high.values, prices.low.values, prices.close.values)
    df = prices.copy()
    df['CDLONNECK'] = CDLONNECK
    df = df[['ico_id', 'ico_symbol', 'date', 'close', 'CDLONNECK']]
    df['signal'] = df['CDLONNECK'].apply(signal_candle)
    if days:
        return df.iloc[len(df)-days:]
    return df


def CDLPIERCING(prices, days=None):
    CDLPIERCING = talib.CDLPIERCING(prices.open.values, prices.high.values, prices.low.values, prices.close.values)
    df = prices.copy()
    df['CDLPIERCING'] = CDLPIERCING
    df = df[['ico_id', 'ico_symbol', 'date', 'close', 'CDLPIERCING']]
    df['signal'] = df['CDLPIERCING'].apply(signal_candle)
    if days:
        return df.iloc[len(df)-days:]
    return df


def CDLRICKSHAWMAN(prices, days=None):
    CDLRICKSHAWMAN = talib.CDLRICKSHAWMAN(prices.open.values, prices.high.values, prices.low.values, prices.close.values)
    df = prices.copy()
    df['CDLRICKSHAWMAN'] = CDLRICKSHAWMAN
    df = df[['ico_id', 'ico_symbol', 'date', 'close', 'CDLRICKSHAWMAN']]
    df['signal'] = df['CDLRICKSHAWMAN'].apply(signal_candle)
    if days:
        return df.iloc[len(df)-days:]
    return df


def CDLRISEFALL3METHODS(prices, days=None):
    CDLRISEFALL3METHODS = talib.CDLRISEFALL3METHODS(prices.open.values, prices.high.values, prices.low.values, prices.close.values)
    df = prices.copy()
    df['CDLRISEFALL3METHODS'] = CDLRISEFALL3METHODS
    df = df[['ico_id', 'ico_symbol', 'date', 'close', 'CDLRISEFALL3METHODS']]
    df['signal'] = df['CDLRISEFALL3METHODS'].apply(signal_candle)
    if days:
        return df.iloc[len(df)-days:]
    return df


def CDLSEPARATINGLINES(prices, days=None):
    CDLSEPARATINGLINES = talib.CDLSEPARATINGLINES(prices.open.values, prices.high.values, prices.low.values, prices.close.values)
    df = prices.copy()
    df['CDLSEPARATINGLINES'] = CDLSEPARATINGLINES
    df = df[['ico_id', 'ico_symbol', 'date', 'close', 'CDLSEPARATINGLINES']]
    df['signal'] = df['CDLSEPARATINGLINES'].apply(signal_candle)
    if days:
        return df.iloc[len(df)-days:]
    return df


def CDLSHOOTINGSTAR(prices, days=None):
    CDLSHOOTINGSTAR = talib.CDLSHOOTINGSTAR(prices.open.values, prices.high.values, prices.low.values, prices.close.values)
    df = prices.copy()
    df['CDLSHOOTINGSTAR'] = CDLSHOOTINGSTAR
    df = df[['ico_id', 'ico_symbol', 'date', 'close', 'CDLSHOOTINGSTAR']]
    df['signal'] = df['CDLSHOOTINGSTAR'].apply(signal_candle)
    if days:
        return df.iloc[len(df)-days:]
    return df


def CDLSHORTLINE(prices, days=None):
    CDLSHORTLINE = talib.CDLSHORTLINE(prices.open.values, prices.high.values, prices.low.values, prices.close.values)
    df = prices.copy()
    df['CDLSHORTLINE'] = CDLSHORTLINE
    df = df[['ico_id', 'ico_symbol', 'date', 'close', 'CDLSHORTLINE']]
    df['signal'] = df['CDLSHORTLINE'].apply(signal_candle)
    if days:
        return df.iloc[len(df)-days:]
    return df


def CDLSPINNINGTOP(prices, days=None):
    CDLSPINNINGTOP = talib.CDLSPINNINGTOP(prices.open.values, prices.high.values, prices.low.values, prices.close.values)
    df = prices.copy()
    df['CDLSPINNINGTOP'] = CDLSPINNINGTOP
    df = df[['ico_id', 'ico_symbol', 'date', 'close', 'CDLSPINNINGTOP']]
    df['signal'] = df['CDLSPINNINGTOP'].apply(signal_candle)
    if days:
        return df.iloc[len(df)-days:]
    return df


def CDLSTALLEDPATTERN(prices, days=None):
    CDLSTALLEDPATTERN = talib.CDLSTALLEDPATTERN(prices.open.values, prices.high.values, prices.low.values, prices.close.values)
    df = prices.copy()
    df['CDLSTALLEDPATTERN'] = CDLSTALLEDPATTERN
    df = df[['ico_id', 'ico_symbol', 'date', 'close', 'CDLSTALLEDPATTERN']]
    df['signal'] = df['CDLSTALLEDPATTERN'].apply(signal_candle)
    if days:
        return df.iloc[len(df)-days:]
    return df


def CDLSTICKSANDWICH(prices, days=None):
    CDLSTICKSANDWICH = talib.CDLSTICKSANDWICH(prices.open.values, prices.high.values, prices.low.values, prices.close.values)
    df = prices.copy()
    df['CDLSTICKSANDWICH'] = CDLSTICKSANDWICH
    df = df[['ico_id', 'ico_symbol', 'date', 'close', 'CDLSTICKSANDWICH']]
    df['signal'] = df['CDLSTICKSANDWICH'].apply(signal_candle)
    if days:
        return df.iloc[len(df)-days:]
    return df


def CDLTAKURI(prices, days=None):
    CDLTAKURI = talib.CDLTAKURI(prices.open.values, prices.high.values, prices.low.values, prices.close.values)
    df = prices.copy()
    df['CDLTAKURI'] = CDLTAKURI
    df = df[['ico_id', 'ico_symbol', 'date', 'close', 'CDLTAKURI']]
    df['signal'] = df['CDLTAKURI'].apply(signal_candle)
    if days:
        return df.iloc[len(df)-days:]
    return df


def CDLTASUKIGAP(prices, days=None):
    CDLTASUKIGAP = talib.CDLTASUKIGAP(prices.open.values, prices.high.values, prices.low.values, prices.close.values)
    df = prices.copy()
    df['CDLTASUKIGAP'] = CDLTASUKIGAP
    df = df[['ico_id', 'ico_symbol', 'date', 'close', 'CDLTASUKIGAP']]
    df['signal'] = df['CDLTASUKIGAP'].apply(signal_candle)
    if days:
        return df.iloc[len(df)-days:]
    return df


def CDLTHRUSTING(prices, days=None):
    CDLTHRUSTING = talib.CDLTHRUSTING(prices.open.values, prices.high.values, prices.low.values, prices.close.values)
    df = prices.copy()
    df['CDLTHRUSTING'] = CDLTHRUSTING
    df = df[['ico_id', 'ico_symbol', 'date', 'close', 'CDLTHRUSTING']]
    df['signal'] = df['CDLTHRUSTING'].apply(signal_candle)
    if days:
        return df.iloc[len(df)-days:]
    return df


def CDLTRISTAR(prices, days=None):
    CDLTRISTAR = talib.CDLTRISTAR(prices.open.values, prices.high.values, prices.low.values, prices.close.values)
    df = prices.copy()
    df['CDLTRISTAR'] = CDLTRISTAR
    df = df[['ico_id', 'ico_symbol', 'date', 'close', 'CDLTRISTAR']]
    df['signal'] = df['CDLTRISTAR'].apply(signal_candle)
    if days:
        return df.iloc[len(df)-days:]
    return df


def CDLUNIQUE3RIVER(prices, days=None):
    CDLUNIQUE3RIVER = talib.CDLUNIQUE3RIVER(prices.open.values, prices.high.values, prices.low.values, prices.close.values)
    df = prices.copy()
    df['CDLUNIQUE3RIVER'] = CDLUNIQUE3RIVER
    df = df[['ico_id', 'ico_symbol', 'date', 'close', 'CDLUNIQUE3RIVER']]
    df['signal'] = df['CDLUNIQUE3RIVER'].apply(signal_candle)
    if days:
        return df.iloc[len(df)-days:]
    return df


def CDLUPSIDEGAP2CROWS(prices, days=None):
    CDLUPSIDEGAP2CROWS = talib.CDLUPSIDEGAP2CROWS(prices.open.values, prices.high.values, prices.low.values, prices.close.values)
    df = prices.copy()
    df['CDLUPSIDEGAP2CROWS'] = CDLUPSIDEGAP2CROWS
    df = df[['ico_id', 'ico_symbol', 'date', 'close', 'CDLUPSIDEGAP2CROWS']]
    df['signal'] = df['CDLUPSIDEGAP2CROWS'].apply(signal_candle)
    if days:
        return df.iloc[len(df)-days:]
    return df


def CDLXSIDEGAP3METHODS(prices, days=None):
    CDLXSIDEGAP3METHODS = talib.CDLXSIDEGAP3METHODS(prices.open.values, prices.high.values, prices.low.values, prices.close.values)
    df = prices.copy()
    df['CDLXSIDEGAP3METHODS'] = CDLXSIDEGAP3METHODS
    df = df[['ico_id', 'ico_symbol', 'date', 'close', 'CDLXSIDEGAP3METHODS']]
    df['signal'] = df['CDLXSIDEGAP3METHODS'].apply(signal_candle)
    if days:
        return df.iloc[len(df)-days:]
    return df


############################
# Combined indicator classes
############################

class CombinedIndicator:

    def __init__(self, indicators, weights, threshold):
        self.threshold = threshold
        indicators_dict = {}
        indicators_weight = {}
        for idx, indic in enumerate(indicators):
            indicators_dict[indic] = globals()[indic]
            indicators_weight[indic] = weights[idx]
        self.indicators = indicators_dict
        self.weights = indicators_weight

    def get_signals(self, prices, days=None):

        # building signals for each indicator
        indic_signals = {}
        for indic in self.indicators:
            indic_signals[indic] = self.indicators[indic](prices=prices)
            if indic_signals[indic] is None:
                print('Not enough price to compute Combined indicator')
                return None

        # Intersection of dates for all indicators
        common_dates = sorted(list(reduce(set.intersection, [set(item.date) for item in list(indic_signals.values())])))
        if len(common_dates) == 0:
            print("There are no common dates between the data source and the price data.")
            return None

        else:
            df = prices[['ico_id', 'ico_symbol', 'date', 'close']][prices.date.isin(common_dates)].reset_index(drop=True)
            for indicator in indic_signals:
                # Restrict available dates
                indic_signals[indicator] = indic_signals[indicator][indic_signals[indicator].date.isin(common_dates)].reset_index(drop=True)
                # Concatenate signals column
                df = _pd.concat([df, indic_signals[indicator][['trend']].rename(columns={'trend': indicator})], axis=1)

        weights = self.weights.values()
        df.iloc[:, 4:] = df.iloc[:, 4:] * weights
        df['score'] = df.iloc[:, 4:].sum(axis=1)
        df['trend'] = df.score.apply(lambda x: _np.select([x > self.threshold, x < - self.threshold], [1, -1], default=0))

        # Creating bullish or bearish signals based on trend
        df['signal'] = df.trend
        # Detecting all same adjacent trends
        df_signals = df[df.signal != 0].copy()
        df_signals['previous_signal'] = df_signals.signal.shift().fillna(0)
        indexes = df_signals[df_signals.signal == df_signals.previous_signal].index

        # setting the repeated signals to 0
        df.loc[indexes, 'signal'] = 0

        if days is not None:
            if days >= len(df):
                pass
            else:
                df = df.iloc[len(df) - days:]
        return df.reset_index(drop=True)

    def plot_weights(self, figsize=(15, 7)):

        _plt.figure(figsize=figsize)
        _sns.barplot(x=list(self.weights.keys()), y=list(self.weights.values()))
        _plt.xticks(rotation=45, fontsize=14)
        _plt.title(f'\nCombined indicator with threshold: {self.threshold}')
        _plt.show()

    def plot_signals(self, prices, days=None, figsize=(15, 7)):
        signals = self.get_signals(prices=prices, days=days)
        plot_signals(signals)

    def signals_stats(self, prices):
        signals = self.get_signals(prices=prices, days=None)
        return signals_stats(signals)
