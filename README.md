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
#### **See also example screenshots below**

1. Obtain your own [key](https://www.alphavantage.co/support/#api-key) for the AlphaVantage API. 
2. Run populate_SQL.py:
```console
$ python3 src/populate_SQL.py --key=alphavantage_key
```
  This program will retrieve basic information regarding the 500+ stocks comprising the S&P 500 index, and start populating an SQLite database with the historical record for each stock for the selected time period. See screenshots 1-4.

* Replace 'alphavantage_key' with the API key you obtained from AlphaVantage.  
* Alternatively, insert your key into alpha_vantage_key.py - this way you will not have to input it every run.
* (Optional) Before running the program, you may want to open 'populate_SQL.py' in an editor and modify the user-definable variables as required.  
* When using the free version of the API, you will be able to retrieve data for up to 25 stocks per day. This means that populating the SQLite file with data for all S&P 500 stocks will require running this script once per day for 21 days. You may want to set your local system to automatically run this script once a day (e.g., by using Task Scheduler in Windows).  


3. Obtain an [API key for custom Google searches](https://developers.google.com/custom-search/v1/introduction). 
4. Run get_baseURLs.py:
```console
$ python3 src/get_baseURLs.py --key=google_api_key
```
The purpose of this program is to use a custom Google search engine to retrieve a "base URL" for each stock from https://companiesmarketcap.com/. Base URLs will be stored in the 'Stocks' table of the SQLite database populated by 'populate_SQL.py', and will be used by subsequent programs to retrieve different kinds of information from companiesmarketcap.com. See screenshot 5.

* Replace google_api_key with the key you got from Google. 
* Alternatively, paste this key into google_api_key.py - this way you will not have to input it every run. 
* The free version of the Google search API allows you to perform 100 requests per day. Therefore, when using the free version it will take 5 days to retrieve base URLs for all stocks in the Stocks table. 
* The "base URL" itself will not lead to a valid webpage (returns 404) - required URL slugs will be added by subsequent programs. 
* In order to avoid wasting API calls on unusable data, stocks with a too-short history (i.e., newly issued companies) will be excluded from further steps. You can adjust what is considered "too short" in the "user-definable variables" before running the program (optional).
5. Run data_adjustments.py:
```console
$ python3 src/data_adjustments.py
```
This program pre-processes the raw historical data obtained from AlphaVantage. Specifically, it performs stock price adjustments for stock splits, stock consolidations (AKA reverse splits), and dividend payments. See screenshots 6-8.

* The program builds on the base URLs obtained in the previous step to retrieve relevant info from https://companiesmarketcap.com/. 

* The program  creates/updates 3 new SQLite databases: i. A DB for recording adjustments; ii. A DB for storing data that was split-adjusted, or the raw data if no split adjustment was necessary; iii. A DB for storing data that was split-adjusted if required, and also dividend-adjusted. The underlying rational for DBs ii & iii is: It almost always makes sense to look at split-adjusted data. However, whether one should look at dividend-adjusted data or not depends on one's goals. 
Therefore, the split-adjusted DB will be used for most subsequent analyses, while the (split-and)-dividend-adjusted DB will be used in more specific cases. For more details, I recommend checking out [this video](https://www.youtube.com/watch?v=jIHjYrZoXxU&t=253s).

* The program generates figures showing 'raw' and 'clean' (when applicable) data for each stock - each stock shown in a subplot, up to 20 subplots per figure. For this, the program creates a new dir ./adjustment_figures/ , to which these figures are saved.  

6. *TO BE CONTINUED*  

**Note:** The package also contains a module ["ticker_to_company"](https://github.com/ronihogri/S-P-500-Stocks-Analysis/blob/main/src/ticker_to_company.py). This module contains a custom class "Company", which accepts a ticker symbol of a stock, and can be used to retrieve different kinds of information regarding the company issuing this stock (e.g., business sector, geographic location of HQ). This can be used, for example, to get features relevant for model training without having to explicitly retrieve them from the SQL database each time. 


## Example screenshots
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

