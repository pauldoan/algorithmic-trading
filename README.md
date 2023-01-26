# algorithmic_trading

A comprehensive python script to perform automated crypto tradings on **CoinBase Pro** based on a selection of Trading stragies. 
The technicalanalysis script defines tens of different strategies based on technical analaysis (price movements), including simple moving averages, relative strengh index analysis, or more complex adaptive moving averages. It also includes a simple backtesting framework to analyze the performance of a strategy (not taking into account exchange fees).

The algo_trading script leverages the Coinbase Pro API to automate the market orders based on the signals provided by the trading. By default, a decision is made every 1 hour. 
