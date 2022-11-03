"""# Data Analysis Plan: Foreign Exchange Analysis

# Team 6
- Mahek Aggarwal
- John Gemmell
- Jacob Kulik
- David Pogrebitskiy


# Project Goal:
This Project aims to use exchangerate.host, a free foreign exchange and crypto API,
to understand how different currencies change and in relation to others and if a movement in
one can help predict a movement in another. By analysing trends and volatility, we will be able
to understand which currencies trigger a global movement, which ones tend to follow afterwards, and
be able to predict a currency's direction if we see a movement in a currency that it tracks.

# Data:
## Overview:

We will request Foreign Exchange values for a variety of different currencies and cryptos
from the ExchangeRate API. Our data will include both major and minor currencies that are pegged to/track
the currencies of first-world countries. Because the data coming from the API is in a time-series format,
we will be able to look at a variety of different period lengths between observations to see
which length best suits our needs.
# Pipeline Overview:
##API and Formating Functions:
- `api_req()`
    - makes an initial request to the API that includes time-series data of all of our
    desired parameters using Python's kwargs feature
FUNCTIONS HERE

- merge_df()
    - Merge multiple years worth of data into one dataframe

##Analysis and Visualizations:
- moving_avg()
    - builds a moving function for a certain currency
FUNCTIONS HERE
"""
#%%
import requests
from pprint import pprint
import pandas as pd
#%%

def api_req(**kwargs):
    '''
    This function calls an exchange rate api and builds a df with the data
    A list of strings (currencies) is a parameter
    returns a transpose dataframe where the dates are the indices

    Params for API call kwargs:
        start_date [required] String format (YYYY-MM-DD)
        end_date [required] String format (YYYY-MM-DD)
        base. example:base=USD
        symbols	[optional] Enter a list of comma-separated currency codes to limit output currencies. example:symbols=USD,EUR,CZK
        amount	[optional] The amount to be converted. example:amount=1200
        places	[optional] Round numbers to decimal place. example:places=2
    '''

    params = kwargs
    url = 'https://api.exchangerate.host/timeseries?'
    response = requests.get(url, params=params)
    data = response.json()


    return  pd.DataFrame(data['rates']).T

#%%

def merge_df_by_years(start_year, end_year, currencies = ""):
    '''
    Creates a dataframe containing the exchange rates from the start year to the end year.
    :param start_year (int):
    :param end_year (int):
    :param currencies (str):
    :return: DataFrame
    '''

    df_output = pd.DataFrame()
    for year in range(start_year, end_year, 2):
        params = {
            'start_date': f'{year}-01-01',
            'end_date': f'{year + 1}-01-01',
        }
        df_year = api_req(params)
        df_output = pd.concat([df_output, df_year])
    return df_output

#Visualizations

def moving_avg(df, roll, *curs):
    '''
    Creates a moving average plot for a given number of currencies and their moving averages
    df - dataframe, roll - int and number of days to be smoothed, *curs - list of currencies
    returns an updated df and a plot
    '''
    fig, ax = plt.subplots()
    for cur in curs:
        cur_idx = cur + '_avg'
        df[cur_idx] = df[cur].rolling(roll).mean()
        df_usd[[cur, cur_idx]].plot(ax=ax, label='ROLLING AVERAGE',
                                  figsize=(16, 8))
    return df

'''
df_usd = moving_avg(df, 30, 'GBP','EUR')
'''

