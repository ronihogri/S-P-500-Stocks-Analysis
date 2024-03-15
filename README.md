# S&P 500 Stocks Analysis
S&amp;P 500 stocks analysis - RESTful API, SQLite, data analysis, visualization and ML-based predictions with python.  
Roni Hogri, March 2024.  

  
## ---*THIS PROJECT IS A WORK IN PROGRESS*---

  
The purpose of this project is to analyze historical data regarding the stocks included in the S&P 500 index, and make predictions regarding the future performance of these stocks using machine learning. The historical data is obtained using the free version of the RESTful API offered by [AlphaVantage](https://www.alphavantage.co/).  

  
## Workflow
*See also example screenshots below.
1. Download the package from the [src folder](https://github.com/ronihogri/S-P-500-Stocks-Analysis/blob/main/src/).
2. Obtain your own [key](https://www.alphavantage.co/support/#api-key) for the AlphaVantage API. 
3. Open 'alpha_vantage_key.py' in an editor, and insert your key as a str instead of 'place_your_key_here'.
4. Make sure that all modules specified in [requirements.text](https://github.com/ronihogri/S-P-500-Stocks-Analysis/blob/main/requirements.txt) are installed.
5. (Optional) You may open 'populate_SQL.py' in an editor and modify the User-definable variables as required.  
6. Run 'populate_SQL.py' - this will retrieve basic information regarding the 500+ stocks comprising the S&P 500 index, and start populating an SQLite database with the historical record for each stock for the selected time period. *Note*: When using the free version of the API, you will be able to retrieve data for up to 25 stocks per day. This means that populating the SQLite file with data for all S&P 500 stocks will require running this script once per day for 21 days. You may want to set your local system to automatically run this script once a day (e.g., by using Task Scheduler in Windows).
7. Run the [preprocessing jupyter notebook](https://github.com/ronihogri/S-P-500-Stocks-Analysis/blob/main/src/preprocessing.ipynb) to check and clean data. This includes discarding datasets that are much shorter than others (new stocks), as well as price adjustment for stock splits and stock consolidation. Some example screenshots are shown below. See notebook for full documentation.  
8. *TO BE CONTINUED*  

The package also contains a module ["ticker_to_company"](https://github.com/ronihogri/S-P-500-Stocks-Analysis/blob/main/src/ticker_to_company.py). This module contains a custom class "Company", which accepts a ticker symbol of a stock, and can be used to retrieve different kinds of information regarding the company issuing this stock (e.g., business sector, geographic location of HQ). This can be used, for example, to get features relevant for model training without having to explicitly retrieve them from the SQL database each time. 


## Example screenshots
Output when running 'populate_SQL.py' for the first time - the program creates a database with 1 table ('Stocks') containing the basic information for all S&P 500 stocks, and then 1 table per stock for storing the historical data obtained from the API: 
<br><br>![](https://github.com/ronihogri/S-P-500-Stocks-Analysis/blob/main/images/get_symbols_and_make_tables.png)<br><br>  

  
  
  <br><br>Output after running 'populate_SQL.py' the Nth time (all tables already exist):  
<br><br>![](https://github.com/ronihogri/S-P-500-Stocks-Analysis/blob/main/images/API_requests_exceeded_2.png)<br><br>  

  
  
  <br><br>The 'Stocks' table containing basic info for all stocks:  
<br><br>![](https://github.com/ronihogri/S-P-500-Stocks-Analysis/blob/main/images/stocks_list_sql.png)<br><br>  

    
  <br><br>An example of a table containinig historical data (5 year period) for the stock with the symbol 'GOOGL' (table head and tail are shown on the left and right, respectively; DailyChange refers to change in % compared to the previous day):  
 <br><br>![](https://github.com/ronihogri/S-P-500-Stocks-Analysis/blob/main/images/example_GOOGL.png)<br><br>  

   <br><br>Raw price data from 4 stocks. The 2 stocks on the right ('GEHC' and 'KVUE') have been issued recently (in 2023), and therefore their datasets are much shorter. You can adjust your threshold to exclude or include data depending on how much shorter it is than the longest datasets (most stocks included in the S&P 500 have been traded for well over 5 years):
   <br><br>![](https://github.com/ronihogri/S-P-500-Stocks-Analysis/blob/main/images/KVUE%20GEHC%20too%20short.png)<br><br>  

  <br><br>Last lines of the table showing stock split/consolidation events automatically detected by the program:
     <br><br>![](https://github.com/ronihogri/S-P-500-Stocks-Analysis/blob/main/images/split_query2.png)<br><br>
     Below you can see a split event entered manually. This was an unusual 1281:1000 split, which would be very difficult to accurately detect automatically.<br><br>  

  <br><br>Plots showing stock prices before and after split adjustments (blue and red lines, respectively). The dashed vertical line shows the date(s) of split events per stock. 
       <br><br>![](https://github.com/ronihogri/S-P-500-Stocks-Analysis/blob/main/images/split_adjusted_plots.png)<br><br>
       Note that, for each stock, the blue and red lines converge after the last split adjustment - as would be expected.<br><br>

