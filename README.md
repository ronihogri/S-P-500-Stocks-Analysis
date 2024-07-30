# S&P 500 Stocks Analysis
S&amp;P 500 stocks analysis - RESTful API, SQLite, data preprocessing, data analysis, visualization, and ML-based predictions with python.  
Roni Hogri, March 2024.  

  
## ---*THIS PROJECT IS A WORK IN PROGRESS*---

  
The purpose of this project is to analyze historical data regarding the stocks included in the S&P 500 index, and make predictions regarding the future performance of these stocks using machine learning. The historical data is obtained using the free version of the RESTful API offered by [AlphaVantage](https://www.alphavantage.co/).  


## Installation

```console
# Clone this repository
$ git clone https://github.com/ronihogri/S-P-500-Stocks-Analysis.git

# Go to the appropriate directory
$ cd S-P-500-Stocks-Analysis

# Install requirements
$ python3 -m pip install -r requirements.txt
```
* Note that this project is a work in progress. Therefore, requirements may change as project develops. 

## Workflow
#### **See also figures below**

1. Obtain your own [key](https://www.alphavantage.co/support/#api-key) for the AlphaVantage API. 
2. Run populate_SQL.py:
```console
$ python3 src/populate_SQL.py --key=alphavantage_key
```
  This program will retrieve basic information regarding the 500+ stocks comprising the S&P 500 index, and start populating an SQLite database with the historical record for each stock for the selected time period. See screenshots in figures 1-4.

* Replace 'alphavantage_key' with the API key you obtained from AlphaVantage.  
* Alternatively, insert your key into alpha_vantage_key.py - this way you will not have to input it every run.
* (Optional) Before running the program, you may want to open 'populate_SQL.py' in an editor and modify the user-definable variables as required.  
* When using the free version of the API, you will be able to retrieve data for up to 25 stocks per day. This means that populating the SQLite file with data for all S&P 500 stocks will require running this script once per day for 21 days. You may want to set your local system to automatically run this script once a day (e.g., by using Task Scheduler in Windows).  


3. Obtain an [API key for custom Google searches](https://developers.google.com/custom-search/v1/introduction). 
4. Run get_baseURLs.py:
```console
$ python3 src/get_baseURLs.py --key=google_api_key
```
The purpose of this program is to use a custom Google search engine to retrieve a "base URL" for each stock from https://companiesmarketcap.com/. Base URLs will be stored in the 'Stocks' table of the SQLite database populated by 'populate_SQL.py', and will be used by subsequent programs to retrieve different kinds of information from companiesmarketcap.com. See screenshot in figure 5.

* Replace google_api_key with the key you got from Google. 
* Alternatively, paste this key into google_api_key.py - this way you will not have to input it every run. 
* The free version of the Google search API allows you to perform 100 requests per day. Therefore, when using the free version it will take 5 days to retrieve base URLs for all stocks in the Stocks table. 
* The "base URL" itself will not lead to a valid webpage (returns 404) - required URL slugs will be added by subsequent programs. 
* In order to avoid wasting API calls on unusable data, stocks with a too-short history (i.e., newly issued companies) will be excluded from further steps. You can adjust what is considered "too short" in the "user-definable variables" before running the program (optional).
5. Run data_adjustments.py:
```console
$ python3 src/data_adjustments.py
```
This program pre-processes the raw historical data obtained from AlphaVantage. Specifically, it performs stock price adjustments for stock splits, stock consolidations (AKA reverse splits), and dividend payments. See figures 6-8.

* The program builds on the base URLs obtained in the previous step to retrieve relevant info from https://companiesmarketcap.com/. 

* The program  creates/updates 3 new SQLite databases: i. A DB for recording adjustments; ii. A DB for storing data that was split-adjusted, or the raw data if no split adjustment was necessary; iii. A DB for storing data that was split-adjusted if required, and also dividend-adjusted. The underlying rational for DBs ii & iii is: It almost always makes sense to look at split-adjusted data. However, whether one should look at dividend-adjusted data or not depends on one's goals. 
Therefore, the split-adjusted DB will be used for most subsequent analyses, while the (split-and)-dividend-adjusted DB will be used in more specific cases. For more details, I recommend checking out [this video](https://www.youtube.com/watch?v=jIHjYrZoXxU&t=253s).

* The program generates figures showing 'raw' and 'clean' (when applicable) data for each stock - each stock shown in a subplot, up to 20 subplots per figure. For this, the program creates a new dir ./adjustment_figures/ , to which these figures are saved.  

6. Run stock_price_prediction_demo_LSTM.py:
```console
$ python3 src/stock_price_prediction_demo_LSTM.py
```  
This demo program attempts to predict stock price changes in the next OUTPUT_WINDOW_SIZE days based on the behavior of predictors during the previous INPUT_WINDOW_SIZE days. For this, a TensorFlow Sequential() model containing a long short-term memory layer (hereafter: 'LSTM model') is used. Figures 9-11 show screenshots of the program. Figures 12-14 show example results from one stock for which reasonably good predictions were obtained (Danaher Corporation, symbol:'DHR'). 

* With each run of the program, a prediction is made for a single stock (hereafter: 'master stock'). However, the prices of other stocks belonging to the same business subsector (see 'GICS_Sub_Industry' column in the 'Stocks' table of the sqlite DBs) are considered as potential predictors, together with the price and volume of the master stock itself. 

* The default behavior of the program is to automatically select the next stock to be analyzed, and to go subsector by subsector. However, the user can override this by specifying a stock for which predictions should be made. 

* The dataset is split into 3 parts: training, validation, and test. The training and validation splits are used to optimize the model, while the test split produces the final (single) prediction sequence with a length of OUTPUT_WINDOW_SIZE days (last days of the historical period for which data exists in the DB). 

* To optimize the performance of the LSTM model, the model goes through hyperparameter tuning (e.g., the number of LSTM units, the number of Dropout units, etc.). This is the most time-consuming part of the program. 

* Once the 'best hyperparameters' are set, the model is created with these hyperparameters. The model is then used to make predictions for each data split (training, validation, and test) separately. 

* The program creates 4 folders under the project folder (default: "./LSTM_demo/50_20/"): 'figures', 'results', 'tuner' and 'model'. Each of these folders is populated with a folder for each master stock (named after the stock's ticker symbol). The 'figures' folder contains plots showing how the master stock relates to its subsector, as well as figures showing prediction results. The 'results' folder contains CSV files containing results of subsector correlations, predictions, and prediction errors (as root mean squared errors, or RMSEs). The 'tuner' folder contains files used by the program for hyperparameter tuning. Once a model is fitted on the master stock's data, the 'model' folder contains files that potentially allow the user to reuse the trained model rather than creating a new one. Note that if you want to re-tune and/or re-fit the model, you would have to delete the master stock's folder within the 'tuner' and/or 'model' folders, respectively.  


**Note:** The package also contains a module ["ticker_to_company"](https://github.com/ronihogri/S-P-500-Stocks-Analysis/blob/main/src/ticker_to_company.py). This module contains a custom class "Company", which accepts a ticker symbol of a stock, and can be used to retrieve different kinds of information regarding the company issuing this stock (e.g., business sector, geographic location of HQ). This can be used, for example, to get features relevant for model training without having to explicitly retrieve them from the SQL database each time. 


## Figures
1. Output when running 'populate_SQL.py' for the first time - the program creates a database with 1 table ('Stocks') containing the basic information for all S&P 500 stocks, and then 1 table per stock for storing the historical data obtained from the API: 
<br><br>![](https://github.com/ronihogri/S-P-500-Stocks-Analysis/blob/main/images/get_symbols_and_make_tables.png)<br><br>  


2. Output after running 'populate_SQL.py' the Nth time (all tables already exist):  
<br><br>![](https://github.com/ronihogri/S-P-500-Stocks-Analysis/blob/main/images/API_requests_exceeded_2.png)<br><br>  

  
  
3. The 'Stocks' table containing basic info for all stocks:  
<br><br>![](https://github.com/ronihogri/S-P-500-Stocks-Analysis/blob/main/images/stocks_list_sql.png)<br><br>  

    
4. An example of a table containinig historical data (5 year period) for the stock with the symbol 'GOOGL' (table head and tail are shown on the left and right, respectively; DailyChange refers to change in % during the day - close vs open prices):  
<br><br>![](https://github.com/ronihogri/S-P-500-Stocks-Analysis/blob/main/images/example_GOOGL.png)<br><br>  

5. Example console output after running get_baseURLs.py:
<br><br>![](https://github.com/ronihogri/S-P-500-Stocks-Analysis/blob/main/images/fetching_base_urls.png)<br><br>  

6. Example console outputs after running data_adjustments.py:
<br><br>![](https://github.com/ronihogri/S-P-500-Stocks-Analysis/blob/main/images/end_data_adjustment_console.png)<br><br>

7. Content of SQLite file used to record stock price adjustments (left to right: heads of 'Adjustments', 'Splits', and 'Dividends' tables):
<br><br>![](https://github.com/ronihogri/S-P-500-Stocks-Analysis/blob/main/images/adjustment_sql.png)<br><br>

8. A selection of subplots showing raw stock price data (blue lines), split- and/or dividend-adjusted stock price (red lines), and adjustment events (dividends: green circles, circle size represents ratio between stock price and dividend amount; stock consolidation/split: purple/black vertical dashed lines, respectively):
<br><br>![](https://github.com/ronihogri/S-P-500-Stocks-Analysis/blob/main/images/example_adjustment_plots.png)<br><br>

9. Example console output when starting to run 'stock_price_prediction_demo_LSTM.py': 
<br><br>![](https://github.com/ronihogri/S-P-500-Stocks-Analysis/blob/main/images/lstm_start.png)<br><br>

10. Example console output showing the tuner searching for the best hyperparameters (i.e., those that minimize validation RMSE): 
<br><br>![](https://github.com/ronihogri/S-P-500-Stocks-Analysis/blob/main/images/tuning_losing_patience.png)<br><br>

11. Example console output showing the tuning summary, and the resulting model starting to be trained: 
<br><br>![](https://github.com/ronihogri/S-P-500-Stocks-Analysis/blob/main/images/tuning_end_model_summary.png)<br><br>

12. Figures showing how the example master stock ('DHR') relates to its subsector (top: scaled prices for all subsector stock + scaled trading volume for 'DHR'; bottom: scatter plots and histograms showing correlations between potential features and within-feature distributions, respectively): 
<br><br>![](https://github.com/ronihogri/S-P-500-Stocks-Analysis/blob/main/images/DHR%20example/DHR_subsector.png)  
![](https://github.com/ronihogri/S-P-500-Stocks-Analysis/blob/main/images/DHR%20example/DHR_feature_correlations.png)<br><br>

13. Figures showing model predictions vs actual prices for 'DHR' for all data splits (top: training; middle: validation; bottom: test): 
<br><br>![](https://github.com/ronihogri/S-P-500-Stocks-Analysis/blob/main/images/DHR%20example/DHR_predictions_training.png)  
![](https://github.com/ronihogri/S-P-500-Stocks-Analysis/blob/main/images/DHR%20example/DHR_predictions_validation.png)  
![](https://github.com/ronihogri/S-P-500-Stocks-Analysis/blob/main/images/DHR%20example/DHR_predictions_test.png)<br><br>

14. Figure showing prediction errors (RMSEs) for 'DHR' for all data splits: 
<br><br>![](https://github.com/ronihogri/S-P-500-Stocks-Analysis/blob/main/images/DHR%20example/DHR_RMSE_per_split.png)  
Here, n refers to the number of prediction vs actual comparisons. For the training and validation splits, the RMSE was calculated separately for each day in the predicted sequence (in this case, OUTPUT_WINDOW_SIZE = 5). For the test split, a single sequence is predicted; therefore, the test RMSE refers to the comparison between all predictions and all actual values in this sequence. 
<br><br>
