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
- scale_cur()
    - scales the currencies in order for t
- moving_avg()
    - builds a moving function for a certain currency

"""
# %%
import requests
from pprint import pprint
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")
from sklearn.preprocessing import MinMaxScaler

# %%

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

    return pd.DataFrame(data['rates']).T


# %%

def merge_df_by_years(start_year=1800, end_year=2020, symbols=""):
    '''
    Creates a dataframe containing the exchange rates from the start year to the end year.
    Merge multiple years worth of data into one dataframe from the API call.
    :param start_year (int):
    :param end_year (int):
    :param symbols (str):
    :return: DataFrame
    '''

    df_output = pd.DataFrame()
    for year in range(start_year, end_year + 1):
        params = {
            'start_date': f'{year}-01-01',
            'end_date': f'{year}-12-31',
            'symbols': symbols
        }
        df_year = api_req(start_date=params['start_date'],
                          end_date=params['end_date'],
                          symbols=params['symbols'])
        df_output = pd.concat([df_output, df_year])
    df_output.dropna(inplace=True, axis=0)
    return df_output


# Visualizations
def scale_cur(df):
    '''
    Scales the exchange rates for a dataframe of currencies
    df- dataframe
    returns a scaled dataframe
    '''

    cols = df.columns
    # fitting a scaler
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df.to_numpy())
    df_scaled = pd.DataFrame(df_scaled, columns=cols)

    # updating indexes to be dates
    df_scaled.index = df.index
    return df_scaled
'''
df_scaled = scale_cur(df)
df_scaled
'''
def moving_avg(df, roll, y, *curs):
    '''
    Creates a moving average plot for a given number of currencies and their moving averages
    df - dataframe, roll - int and number of days to be smoothed, *curs - list of currencies
    returns an updated df and a plot
    '''
    fig, ax = plt.subplots()

    # Creating label based off graph type
    plt.xlabel('Date')
    if y == 'scale':
        plt.ylabel('Scaled Exchange Rate')
        plt.title('Scaled Currencies and Rolling Averages Time-Series')
    else:
        plt.ylabel('Exchange Rate')
        plt.title('Currencies and Rolling Averages Time-Series')

    # iterating across currencies
    for cur in curs:
        cur_idx = cur + '_avg'
        # creating a rolling mean column and plotting both
        df[cur_idx] = df[cur].rolling(roll).mean()
        df_usd[[cur, cur_idx]].plot(ax=ax, label='ROLLING AVERAGE',
                                    figsize=(16, 8))
    return df


'''
df_usd = moving_avg(df, 30, '', 'GBP','EUR')
df_usd = moving_avg(df_scaled, 30, 'scale', 'GBP')

'''

""""
##Analysis Plan

We plan to analyze our time-series data of the currencies using different regression models such
as linear regression, quadratic regression, and logistic regression and comparing these models to 
determine which one yields the best results. The time-series data will be converted to days since with the
first day starting at 0. 
"""
