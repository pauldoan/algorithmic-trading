import datetime as dt
import cbpro
import time
import json
import pandas as pd
import technicalanalysis as ta

# Currency to trade
product = 'BTC-USD'
currency = 'BTC'

################################################################################
# API STUFF

with open('/Users/paul/Library/Mobile Documents/com~apple~CloudDocs/1 Personal/2 Finances/1 Bank/coinbasepro.json') as f:
    api = json.load(f)

auth_client = cbpro.AuthenticatedClient(api['api_key'], api['api_secret'], api['passphrase'])

################################################################################
# Investment Details

# Amount to initially invest
initial_investment = 5

# Amount that will be used for purchase starts at the initial amount
funding = initial_investment


# Will return the ID of your specific currency account
def getSpecificAccount(cur):
    x = auth_client.get_accounts()
    for account in x:
        if account['currency'] == cur:
            return account['id']


# Get the currency's specific ID
specificID = getSpecificAccount(currency)

# Granularity (in seconds). So 300 = data from every 5 min
period = 300

# We will keep track of how many iterations our bot has done
iteration = 1

# Start off by looking to buy
buy = True

################################################################################
# Begin Loop and get Historic Data

while True:

    # getting historical data
    prices = pd.DataFrame()
    end = dt.datetime.now()
    start = end - dt.timedelta(days=1)
    historicData = auth_client.get_product_historic_rates(product, granularity=period, start=start, end=end)
    historicData = pd.DataFrame(historicData, columns=['date', 'low', 'high', 'open', 'close', 'volume']).sort_values('date', ascending=False).reset_index(drop=True)
    prices = pd.concat((prices, historicData))

    # adding a another day of data
    time.sleep(1)
    end = start
    start = end - dt.timedelta(days=1)
    historicData = auth_client.get_product_historic_rates(product, granularity=period, start=start, end=end)
    historicData = pd.DataFrame(historicData, columns=['date', 'low', 'high', 'open', 'close', 'volume']).sort_values('date', ascending=False).reset_index(drop=True)
    prices = pd.concat((prices, historicData))

    # adding a another day of data
    time.sleep(1)
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
    time.sleep(1)

    # Get latest data and show to the user for reference
    newData = auth_client.get_product_ticker(product_id=product)
    print(newData)
    currentPrice = newData['price']

    # computing signals
    indicators = ['mama', 'frama', 'sar', 'bop', 'roc', 'adosc']
    weights = [0.2768361581920904, 0.06779661016949153, 0.18361581920903955, 0.20056497175141244, 0.05649717514124294, 0.21468926553672316]
    threshold = .5
    comb = ta.CombinedIndicator(indicators, weights, threshold)

    signals = comb.get_signals(prices)

    ################################################################################
    # Funds to Use

    # The maximum amount of Cryptocurrency that can be purchased with your funds
    possiblePurchase = (float(funding)) / float(currentPrice)

    # The amount of currency owned
    owned = float(auth_client.get_account(specificID)['available'])

    # The value of the cryptourrency in USD
    possibleIncome = float(currentPrice) * owned

    ################################################################################
    # Decision Making

    # Buy Conditions: buy signal
    if buy == True and signals.signal.iloc[-1] == 1:

        # Place the order
        auth_client.place_market_order(product_id=product, side='buy', funds=str(funding))

        # Print message in the terminal for reference
        message = "Buying Approximately " + str(possiblePurchase) + " " + currency + "  Now @ " + str(currentPrice) + "/Coin. TOTAL = " + str(funding)
        print(message)

        # Update funding level and Buy variable
        funding = 0
        buy = False

    # Sell Conditions: sell signal
    if buy == False and signals.signal.iloc[-1] == - 1:

        # Place the order
        auth_client.place_market_order(product_id=product, side='sell', size=str(owned))

        # Print message in the terminal for reference
        message = "Selling " + str(owned) + " " + currency + "Now @ " + str(currentPrice) + "/Coin. TOTAL = " + str(possibleIncome)
        print(message)

        # Update funding level and Buy variable
        funding = int(possibleIncome)
        buy = True

    # Stop loss: sell everything and stop trading if your value is
    # less than 80% of initial investment
    if (possibleIncome + funding) <= 0.8 * initial_investment:
        # If there is any of the crypto owned, sell it all
        if owned > 0.0:
            auth_client.place_market_order(product_id=product, side='sell', size=str(owned))
            print("STOP LOSS SOLD ALL")
        # Will break out of the while loop and the program will end
        break

    # Printing here to make the details easier to read in the terminal
    print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
    print("iteration number", iteration)

    # Print the details for reference
    print("Current Price: ", currentPrice)
    print("Your Funds = ", funding)
    print("You Own ", owned, currency)

    # Wait for 5 minutes before repeating
    time.sleep(300)
    iteration += 1
