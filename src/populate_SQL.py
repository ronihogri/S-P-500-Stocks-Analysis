'''
Roni Hogri, March 2024

The purpose of this program is to use the API provided by alphavantage.co to get historical data regarding the stocks comprising the S&P 500 index. 
This program gets basic information regarding these stocks and creates the 'Stocks' table in a SQLite file by calling the subprocess 'get_snp_symbols.py'.
The current program creates a separate table for each stock listed in the Stocks table - these tables are then populated with the historical data for each stock. 

This program uses the free version of the API "TIME_SERIES_DAILY". This means that up to 25 requests per day are possible.
Therefore, retrieving info on all 503 stocks comprising the S&P 500 index will require that the program is run once per day for 21 days. 
For additional information on the API, see:
https://www.alphavantage.co/documentation/
https://www.alphavantage.co/support/#support

**IMPORTANT NOTE:**
In order for the program to work, you have to get a free API key from: https://www.alphavantage.co/support/#api-key
This key should be inserted as a string in the file 'alpha_vantage_key.py' instead of 'place_your_key_here'.
'''


"""User-definable variables; modify as necessary:"""
latest_date_dict = {'year': 2024, 'month': 3, 'day': 5} #this is the latest date for which data will be stored. 
historical_period = {'years' : 5, 'months': 0, 'days': 0} #duration of historical period for which to retrieve data.
#Note1: For latest_date_dict and historical_period, the same values should be used in multiple runs - this will ensure consistency in subsequent analyses. 
#Note2: Remember that for each stock, data can only be retrieved starting from when this stock was issued.
sqlite_file_path = './S&P 500.sqlite' #path of SQLite file created by get_snp_symbols.py
wikipedia_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies' #URL of wikipedia page containing list of S&P 500 companies

"""End of user-defined variables."""


#import libraries for global use:
import requests
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import sqlite3
import time
import sys
import argparse
import subprocess


try :
    from alpha_vantage_key import ALPHA_VANTAGE_KEY as key
except Exception as e:
    print(f'\nError encountered: {e}\nPlease make sure to insert a valid key string as the value of ALPHA_VANTAGE_KEY in alpha_vantage_key.py in the same directory as this program.')


"""Globals:"""

#connection and cursor for SQLite file:
conn = sqlite3.connect(sqlite_file_path)
cur = conn.cursor()


"""Functions:"""


def get_stock_list_from_sql(scraping_script) :
    """Retrieves stock symbols from SQL file.
    
    Args:   
        scraping_script (str): Name of py script used to scrape info regarding S&P 500 stocks from wikipedia.
    
    Returns:
        symbols (list): A list containing stock symbols (str) extracted from the 'Stocks' table in the SQLite file.
           
    Raises: 
        Terminates program if the Stocks table cannot be accessed.        
    """    
    
    try :
        #check if 'Stocks' table with list of S&P 500 stocks already exists in SQLite file, and is not empty:
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='Stocks'")
        stock_table_exists = cur.fetchone()

        if stock_table_exists :
            cur.execute('SELECT COUNT(*) FROM Stocks')
            stock_list_row_count = cur.fetchone()[0]
            if stock_list_row_count == 0 :
                populate_table = True
            else : 
                populate_table = False
        else : 
            populate_table = True  
        
        if populate_table : #'Stocks' table doesn't exist, or is not yet populated - fill it  
            #run program to retrieve list of S&P stocks and populate the Stocks table:
            parser = argparse.ArgumentParser() #for passing vars to subprocess
            parser.add_argument('--sqlite_file_path', help='Path to the SQLite file', default=sqlite_file_path)
            parser.add_argument('--wikipedia_url', help='URL of the Wikipedia page', default=wikipedia_url)
            args = parser.parse_args()
            result = subprocess.run(['python', scraping_script, '--sqlite_file_path', sqlite_file_path, '--wikipedia_url', wikipedia_url], capture_output=True, text=True) 

            if result.returncode != 0 : #was an error encountered?
                print(f'Error occurred during execution:\n{result.stderr}\n\n******* Program terminated *******\n')
                sys.exit()

            else: #no error in running scraping_script
                print(result.stdout) #display output of scraping_script

        #import stock info list
        cur.execute("SELECT Symbol FROM Stocks")
        symbols = [row[0] for row in cur.fetchall()]       
        
        return symbols

    except sqlite3.Error as sqlite_error :
        print(f'SQLite error occurred: {sqlite_error}\n\n******* Program terminated *******\n')
        sys.exit()

    except subprocess.CalledProcessError as process_error :
        print(f'An error occurred when trying to run {scraping_script}:\n{process_error.stderr}\n\n******* Program terminated *******\n')
        sys.exit()

    except Exception as e :
        print(f'An unexpected error occurred: {e}\n\n******* Program terminated *******\n')
        sys.exit()
   


def create_sql_table_per_stock(symbols) :
    """For each stock in the S&P 500, creates a new SQL table (if does not yet exist) to hold historical information.
    
    Args:   
        symbols (list): A list containing stock symbols (str) - these will be the names of the new tables.
        
    Globals: 
        conn (sqlite3.Connection): Connection to SQL database.
        cur (sqlite3.Cursor): Handle of SQL file.
        
    Returns:
        None
        
    Raises: 
        Notifies user regarding tables that could not be created.
    """

    counter = 0 #for counting newly created tables
    for symbol in symbols: #for each stock
        try:
            cur.execute(f"SELECT 1 FROM sqlite_master WHERE type='table' AND name='{symbol}'") #check if table already exists in SQLite file
            if cur.fetchone() is None: #if table does not already exist, create it
                cur.execute(f'''CREATE TABLE "{symbol}" 
                (Date TEXT NOT NULL PRIMARY KEY UNIQUE,
                Open FLOAT, Close FLOAT, DailyChange FLOAT,
                Low FLOAT, High FLOAT, DailyRange FLOAT,
                Volume INTEGER)''')
                print(f'Table created for {symbol}')                
                counter += 1            
        except KeyboardInterrupt:
            print('\n****** Program aborted by user *******\n')
            conn.close()
            sys.exit()
        
        except:
            print(f'Failed to create table for {symbol}')
    
    cur.execute("SELECT COUNT(name) FROM sqlite_master WHERE type='table'") #number of tables in SQLite file
    num_tables = cur.fetchone()[0]
    
    if counter > 0: #were new tables created in this run?
        print(f'{counter} new tables created.')
    
    print(f'SQL database contains a total of {num_tables} tables, including the "Stocks" table + historical data for {len(symbols)} stocks.\n')

    

def period_to_retrieve(latest_date, historical_period) :
    """Determines the period for which historical information will be retrieved.
    
    Args:   
        lastest_date (datetime): The latest date for which data should be retrieved.
        historical_period (dict): Contains the number of years, months, and days to be included in the historical data.
                
    Returns:        
        earliest_date_str (str): Beginning of historical period (YYYY-MM-DD).
        latest_date_str (str): End of historical period (YYYY-MM-DD).
    """
    
    earliest_date = latest_date - relativedelta(years=historical_period['years'], months=historical_period['months'], days=historical_period['days'])
    
    earliest_date_str = earliest_date.strftime('%Y-%m-%d')
    latest_date_str = latest_date.strftime('%Y-%m-%d')

    print(f'Historical information will be retrieved for the time period (YYYY-MM-DD): {earliest_date_str} - {latest_date_str}')

    return earliest_date_str, latest_date_str
    


def get_stock_history_from_api(symbol) :
    """Requests data from the "TIME_SERIES_DAILY" API provided by alphavantage (up to 25 requests per day for the free version).

    Args:   
        symbol (str): Symbol of stock for which info is requested.
                
    Returns:        
        data (dict): Dict containing the data retrieved from the API.
    """

    function = 'TIME_SERIES_DAILY' #daily data, not adjusted for dividends and splits
    output_size = 'full' #full = 20+ year data (then trimmed according to defined period); compact = last 100 days
    
    #send request to API and retrieve JSON data as a dict :
    url = f'https://www.alphavantage.co/query?function={function}&symbol={symbol}&apikey={key}&outputsize={output_size}'
    r = requests.get(url)
    data = r.json()

    return data



def extract_data(data, earliest_date_str, latest_date_str):
    """Extract historical data for a particular stock from the data provided by the API, for later storage in the SQLite database.
    
    Args:   
        data (dict): Dict containing the data retrieved from the API.
        earliest_date_str (str): Beginning of historical period (YYYY-MM-DD).
        latest_date_str (str): End of historical period (YYYY-MM-DD).        
             
    Returns:
        dates (list): A list of dates for each trading day in data. 
        opens (list): A list of opening prices of a stock for each trading day in data. 
        closes (list): A list of closing prices of a stock for each trading day in data. 
        lows (list): A list of the lowest prices of a stock for each trading day in data. 
        highs (list): A list of the highest prices of a stock for each trading day in data. 
        volumes (list): A list of the trading volumes of a stock for each trading day in data.     
    """

    meta_data = data['Meta Data']
    timeseries = data['Time Series (Daily)']

    #lists of data
    dates = [date for date in timeseries if (date >= earliest_date_str and date <= latest_date_str)]
    opens = [float(timeseries[date]['1. open']) for date in dates]
    closes = [float(timeseries[date]['4. close']) for date in dates]
    lows = [float(timeseries[date]['3. low']) for date in dates]
    highs = [float(timeseries[date]['2. high']) for date in dates]
    volumes = [int(timeseries[date]['5. volume']) for date in dates]

    return dates, opens, closes, lows, highs, volumes
    
    
    
def get_tables_with_stock_data(symbols) :
    """Make a list of tables containing stock data.
    
    Args:   
        symbols (list): A list containing stock symbols (str) - i.e., names of the relevant tables.       
             
    Returns:
        stock_tables_with_data (list): A list of stock tables that already contain data (have been filled with data from API). 
    """
    
    stock_tables_with_data = []

    for symbol in symbols : #for each table containing stock info
        cur.execute(f"SELECT COUNT(*) FROM '{symbol}'")
        row_count = cur.fetchone()[0]
        
        if row_count > 0 : #if the table contains data
            stock_tables_with_data.append(symbol)

    return stock_tables_with_data



def calculate_from_data(tables_with_data) :
    """Calculate additional values from SQLite data and update SQLite file.
    
    Args:   
        tables_with_data (list): A list containing the names of tables with stock data (already filled with data from API).       
             
    Returns:
        None
    """
    
    digits_after_decimal = 3 #parameter for rounding results of calculations
    
    try:
        for table in tables_with_data :
            cur.execute(f"UPDATE '{table}' SET DailyChange = ROUND(100*Close/Open - 100, {digits_after_decimal})")
            cur.execute(f"UPDATE '{table}' SET DailyRange = ROUND(High - Low, {digits_after_decimal})")
    except Exception as e:
        if 'database is locked' in str(e) :
            prompt = input('\nCannot update tables while SQLite database is open - please close SQLite file and press ENTER.\t')
            calculate_from_data(tables_with_data)
        else :
            print(f'\nError encountered : {e}\n')
        

        
def main() :
    """Program for retrieving historical stock data for S&P 500 stocks for a specified period, and storing this data in an SQLite database."""
        
    global latest_date_dict, historical_period, sqlite_file_path, wikipedia_url, conn, cur

    scraping_script = 'get_snp_symbols.py' #script to scrape basic info for S&P 500 stocks from wikipedia and store it in the SQLite database
    inserted_count = 0 #counters for tables populated and tables that have already been previously filled, respectively
    play_nice = 5 #delay (s) to play nice with API

    symbols = get_stock_list_from_sql(scraping_script) #get symbols of stocks for which to get historical data from existing SQLite table produced by the scraping script
    create_sql_table_per_stock(symbols) #creates an individual table (if does not yet exist) for each stock, for storing its historical data 

    latest_date = datetime(latest_date_dict['year'], latest_date_dict['month'], latest_date_dict['day']) #timestamp of latest date for which data will be retrieved
    earliest_date_str, latest_date_str = period_to_retrieve(latest_date, historical_period) #gets the first and last dates of the historical period based on user preferences 
    
    for symbol in symbols : #for each stock
        
        cur.execute(f'SELECT COUNT(*) FROM "{symbol}"') #check if table contains any data
        row_count = cur.fetchone()[0]
        if row_count > 0 : continue #skip tables that have already been filled       

        data = get_stock_history_from_api(symbol) #get historical data for this stock from the API

        if 'Information' in data : #data not retrieved for this stock - API request limits exceeded - notify user and terminate this program. 
            print(f'\n{data["Information"]}\n')            
            break
        
        dates, opens, closes, lows, highs, volumes = extract_data(data, earliest_date_str, latest_date_str) #get API data as lists that can be unpacked into the SQLite file

        print(f'Historical data for stock with symbol "{symbol}" inserted into SQL database.')
        
        while True :
            try: 
                #unpack raw data from the API to the SQLite database, to be further processed later:
                cur.executemany(f'''INSERT INTO "{symbol}" 
                (Date, Open, Close, Low, High, Volume) VALUES ( ?, ?, ?, ?, ?, ? )''', 
                [(Date, Open, Close, Low, High, Volume) for Date, Open, Close, Low, High, Volume 
                in zip(dates, opens, closes, lows, highs, volumes)])
                inserted_count += 1 
                conn.commit() #save changes to SQLite file after updating each stock table
                break
            except Exception as e :
                if 'database is locked' in str(e) :
                    prompt = input('\nCannot update tables while SQLite database is open - please close SQLite file and press ENTER.\t')
                    continue
                else :
                    print(f'\nError encountered : {e}\n')
                    break

        time.sleep(play_nice) #play nice with API     

    print(f'\nInfo regarding {inserted_count} stocks inserted into SQL database during this run.')
    
    stock_tables_with_data = get_tables_with_stock_data(symbols)
    calculate_from_data(stock_tables_with_data)
    conn.commit() #save changes to SQLite database
    
    print(f'SQLite database now contains historical data on {len(stock_tables_with_data)} stocks.\n')
    conn.close() #close connection to SQLite file
    
    if len(stock_tables_with_data) == len(symbols) : #has data already been retrieved for all stocks in S&P 500 index?
        print('SQLite database contains info on all stocks in the S&P 500 index.\n')
    
    

"""Run program:"""
if __name__ == "__main__": 
    main() 
