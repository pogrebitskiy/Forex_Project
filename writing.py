"""
# Foreign Exchange Analysis Final Report

# Team 6
- Mahek Aggarwal
- John Gemmell
- Jacob Kulik
- David Pogrebitskiy


# Abstract:
This projects aims to examine, analyze, and explain seemingly random and unpredictable movements in foreign exchange rates, potentially informing future investment and asset allocation problems. We gathered years of daily exchange rates for numerous currencies and tried to find interesting relationships between them. By normalizing the exchange rates, we were able to put currencies side by side to find mutual changes that can later be generalized to provide useful and relevant information for prediction. By utilizing linear models and feature imporances, we found that aggregating multiple currencies and analyzing their behavior can help explain volatility in another currency. To more accuractley test and integrate our findings, we suggest simulating investment actions like buying and selling based on movements that we've addressed in our linear models. Additionally, more advanced forcasting models should be used to see more robust results.

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