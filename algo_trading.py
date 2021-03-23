import datetime as dt
import cbpro
import time
import json
import pandas as pd
import technicalanalysis as ta

# CHOOSE HERE PRODUCT AND CURRENCY TO TRADE, AND INITIAL INVESTMENT
product = 'BTC-USD'
currency = 'BTC'
# Amount to initially invest
initial_investment = 10

################################################################################
# API STUFF

# with open('/Users/paul/Library/Mobile Documents/com~apple~CloudDocs/1 Personal/2 Finances/1 Bank/coinbasepro.json') as f:
with open('/home/ubuntu/coinbasepro.json') as f:
    api = json.load(f)

auth_client = cbpro.AuthenticatedClient(api['api_key'], api['api_secret'], api['passphrase'])

################################################################################
# Investment Details

# Amount that will be used for purchase starts at the initial amount
funding = initial_investment


# Will return the ID of your specific currency account
def getSpecificAccount(cur):
    x = auth_client.get_accounts()
    for account in x:
        if account['currency'] == cur:
            return account['id']


# Get the currency's specific ID
account_id = getSpecificAccount(currency)

# Granularity (in seconds). So 300 = data from every 5 min
period = 3600

# Start off by looking to buy
buy = True

################################################################################
# Begin Loop and get Historic Data

while True:

    # Here we get price for that last day
    # getting historical data
    prices = pd.DataFrame()
    end = dt.datetime.utcnow()
    start = end - dt.timedelta(days=1)
    historicData = auth_client.get_product_historic_rates(product, granularity=period, start=start, end=end)
    historicData = pd.DataFrame(historicData, columns=['date', 'low', 'high', 'open', 'close', 'volume']).sort_values('date', ascending=False).reset_index(drop=True)
    prices = pd.concat((prices, historicData))

    # adding two other days of data
    for i in range(2):
        time.sleep(30)
        end = start
        start = end - dt.timedelta(days=1)
        historicData = auth_client.get_product_historic_rates(product, granularity=period, start=start, end=end)
        historicData = pd.DataFrame(historicData, columns=['date', 'low', 'high', 'open', 'close', 'volume']).sort_values('date', ascending=False).reset_index(drop=True)
        prices = pd.concat((prices, historicData))

    # converting to readable date
    prices.date = pd.to_datetime(prices['date'], unit='s')
    prices = prices.sort_values('date').reset_index(drop=True)
    prices['ico_id'] = currency
    prices['ico_symbol'] = currency
    prices = prices[['ico_id', 'ico_symbol', 'date', 'low', 'high', 'open', 'close', 'volume']]

    # Wait for 1 second, to avoid API limit
    time.sleep(30)

    # Get latest data and show to the user for reference
    latest_data = auth_client.get_product_ticker(product_id=product)
    current_price = latest_data['price']

    # computing signals

    # we use Combined indicator
    # indicators = ['mama', 'frama', 'sar', 'bop', 'roc', 'adosc']
    # weights = [0.2768361581920904, 0.06779661016949153, 0.18361581920903955, 0.20056497175141244, 0.05649717514124294, 0.21468926553672316]
    # threshold = .5
    # comb = ta.CombinedIndicator(indicators, weights, threshold)
    # signals = comb.get_signals(prices)

    # we use SAR
    signals = ta.sar(prices)

    ################################################################################
    # Funds to Use

    # The maximum amount of Cryptocurrency that can be purchased with your funds
    possible_purchase = float(funding) / float(current_price)

    # The amount of currency owned
    owned = float(auth_client.get_account(account_id)['available'])

    # The value of my owned crypto in USD
    possible_sell = float(current_price) * owned

    ################################################################################
    # Decision Making

    # Buy Conditions: buy signal
    if buy == True and signals.signal.iloc[-1] == 1:

        # Place the order
        auth_client.place_market_order(product_id=product, side='buy', funds=str(funding))

        # Print message in the terminal for reference
        print('\n')
        print('*' * 30)
        print("BUYING " + str(possible_purchase) + " " + currency + " for " + str(current_price) + f"/Coin at {pd.to_datetime(latest_data['time']).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f'$ {funding} worth of {currency} bought')

        # Update funding level and Buy variable
        funding = 0
        buy = False

        print('Portfolio state')
        print('- - - - - - - -')
        print('Funding: $', funding)
        new_owned_currency = float(auth_client.get_account(account_id)['available'])
        new_owned_usd = float(auth_client.get_account(account_id)['available']) * float(current_price)
        print(f'Crypto: {currency}', new_owned_currency, f'| $', new_owned_usd)
        print('*' * 30)
        print('\n')

    # Sell Conditions: sell signal
    elif buy == False and signals.signal.iloc[-1] == - 1:

        # Place the order
        auth_client.place_market_order(product_id=product, side='sell', size=str(owned))

        # Print message in the terminal for reference
        print('\n')
        print('*' * 30)
        print("SELLING " + str(owned) + " " + currency + " for " + str(current_price) + f"/Coin at {pd.to_datetime(latest_data['time']).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f'$ {possible_sell} worth of {currency} sold')

        # Update funding level and Buy variable
        funding = int(possible_sell)
        buy = True

        print('Portfolio state')
        print('- - - - - - - -')
        print('Funding: ', funding)
        new_owned_currency = float(auth_client.get_account(account_id)['available'])
        new_owned_usd = float(auth_client.get_account(account_id)['available']) * float(current_price)
        print(f'Crypto: {currency} ', new_owned_currency, f'| $ ', new_owned_usd)
        print('*' * 30)
        print('\n')

    else:
        # Printing here to make the details easier to read in the terminal if no signals
        print("\n")
        print('time:', dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        print("current price: ", currency, current_price)

    # Stop loss: sell everything and stop trading if your value is
    # less than 80% of initial investment
    new_investment = funding + float(auth_client.get_account(account_id)['available']) * float(current_price)
    if new_investment <= 0.8 * initial_investment:
        # If there is any of the crypto owned, sell it all
        if float(auth_client.get_account(account_id)['available']) > 0.0:
            auth_client.place_market_order(product_id=product, side='sell', size=str(owned))
            print(f"STOP LOSS SOLD ALL at {pd.to_datetime(latest_data['time']).strftime('%Y-%m-%d %H:%M:%S')}")
        # Will break out of the while loop and the program will end
        break

    # Wait for 1 hour  before repeating
    time.sleep(3600)
