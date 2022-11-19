"""
# Foreign Exchange Analysis Final Report

# Team 6
- Mahek Aggarwal
- John Gemmell
- Jacob Kulik
- David Pogrebitskiy


# Abstract:
WRITE ABSTRACT HERE

# Introduction: The Foreign Exchange market is a global market for the trade of currencies. In free economies,
the value of currencies are based off supply and demand. In some instances, countries peg their currency on another,
meaning their currency moves in line with another. The fluctuation of currencies can also give economic indicators,
such as which economies move in line with one another and the effect of current events. Further, there are many
factors that affect currency value, such as trade, investment, tourism, and geopolitics. Inflation is a very influential
economic phenomenon, and on top of influencing unemployment, it can have an effect on foreign exchange rates.

We are analyzing how the fluctuation of one currency can predict the fluctuation of another.
<img src="https://cdn.corporatefinanceinstitute.com/assets/foreign-exchange.jpeg" width=800px>

# Data Description:
To wrangle foreign exchange rates, request calls were made to [ExchangeRate API](
https://exchangerate.host/#/#docs). These calls provided time-series data for each of the specified currencies. Because
of the flexibility of the API, there were several customizable parameters to fine-tune the API request (date range,
 source, amount, base, etc). The API was limiting each request to 2 years of daily data, so we made functions to make
  multiple requests between our start and end dates and concatenating them together. The final result was about 13 years
   of daily exchange rates for multiple currencies (~4,700 rows).

# Pipeline Overview:
We accomplished this task with the following functions:
## API and Formating Functions:
- `api_req()`
    - Makes an initial request to the API that includes time-series data of all of our
    desired parameters using Python's kwargs feature.

- `merge_df_by_year()`
    - Merge multiple years worth of data into one dataframe because the API limits us
    to 2 years of data per request.

## Analysis and Visualizations:
- `scale_cur()`
    - Scales the currencies to be between 0 and 1 using MinMaxScaler, helping with plotting and analyzing/
- `moving_avg()`
    - Calculates a moving average of every currency of the dataframe using a specified window.
- `calc_pct_change()`
    - Calculates the percentage change between all values, helping to normalize and analyze.

## Machine Learning
- `r2_scoring()`
    - Calculates R2 of cross-validated simple linear regression model.
- `randomness_test()`
    - Checks variable independence, constant variance, and normality assumptions for linear regression.
- `get_mse()`
    - Calculate the Mean Squared Error between true and predicted values.
- `show_fit()`
    - Plot the fit of the linear regression with associated metrics.
- `disp_regress()`
    - Runs a multiple regression model and calculates the r2 of the model.
- `plot_feat_import()`
    - Plot importance of features in a multiple regression model.
- `disp_rfr_regress()`
    - Runs a random forest regression model and calculates the r2 of the model.
"""