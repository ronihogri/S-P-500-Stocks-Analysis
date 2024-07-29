"""
Roni Hogri, April 2024

This program should be run after 'get_baseURLs.py' was used to (at least partially) populate the Stocks table with base URLs
used to retrieve info about the stocks. 
The purpose of the current program is to adjust the raw historical data obtained from AlphaVantage to relevant events during 
this period, namely stock splits, stock consolidations (AKA reverse splits), and dividend payments. 
The program builds on the base URLs to retrieve this info from https://companiesmarketcap.com/. 

This program creates/updates 3 new SQLite databases:
1. A DB for recording adjustments. 
2. A DB for storing data that was split-adjusted, or the raw data if no split adjustment was necessary. 
3. A DB for storing data that was split-adjusted if required, and also dividend-adjusted.
The underlying rational for DBs 2 & 3 is: It almost always makes sense to look at split-adjusted data. 
However, whether one should look at dividend-adjusted data or not depends on one's goals. 
Therefore, the split-adjusted DB will be used for most subsequent analyses, while the 
(split-and)-dividend-adjusted DB will be used in more specific cases. 
For more details, I recommend checking out this video: https://www.youtube.com/watch?v=jIHjYrZoXxU&t=253s
"""

"""User-definable variables; modify as necessary:"""

batch_size = 60  # number of stocks to pre-process each time you run the program.
# set to None to go through all of your existing data in one batch.

overwrite_clean_tables = False  # default=False: this will prevent pre-processing of data that already exists in the clean SQLite file.
# set to True if you want to reanalyze and overwrite existing tables in the 'clean' SQLite databases.

# paths of SQLite DB files:
sqlite_file_name = "S&P 500.sqlite"  # file containing raw data (produced by 'populate_SQL.py' + 'get_base_URLs.py').
sqlite_file_name_split = "S&P 500_split_adj.sqlite"  # file that will contain only data that was adjusted for stock splits/consolidation.
sqlite_file_name_split_div = "S&P 500_split_div_adj.sqlite"  # file that will contain only data that was adjusted for stock splits/consolidation and/or dividends.
sqlite_file_name_record = "S&P 500_record.sqlite"  # file that will contain record on previous runs, including which stocks were adjusted for what

"""End of user-defined variables."""


# import required libraries for global use
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os
import requests
from bs4 import BeautifulSoup
import time
from ticker_to_company import Company 
import sys


"""Globals"""

current_dir = os.path.dirname(os.path.abspath(__file__))  # dir of this script
# pahts for sqlite files:
sqlite_file_path = os.path.join(current_dir, sqlite_file_name)
sqlite_file_path_split = os.path.join(current_dir, sqlite_file_name_split)
sqlite_file_path_split_div = os.path.join(current_dir, sqlite_file_name_split_div)
sqlite_file_path_record = os.path.join(current_dir, sqlite_file_name_record)

# connection and handle for existing SQLite DB
conn = sqlite3.connect(sqlite_file_path)
cur = conn.cursor()

# connection and handle for new SQLite files containing split- and split-and-dividend-adjusted data
conn_split = sqlite3.connect(sqlite_file_path_split)
cur_split = conn_split.cursor()
conn_split_div = sqlite3.connect(sqlite_file_path_split_div)
cur_split_div = conn_split_div.cursor()
conn_record = sqlite3.connect(sqlite_file_path_record)
cur_record = conn_record.cursor()


"""Functions"""


def get_original_symbols():
    """Retrieves the symbols of stocks from the original SQLite DB for which a base URL was already obtained.

    Returns:
       list: Symbols of all stocks to be checked and cleaned (including those that have been cleaned before)
    """

    cur.execute("SELECT Symbol FROM Stocks WHERE Base_URL_MC NOT NULL")
    return [symbol[0] for symbol in cur.fetchall()]


def copy_stocks_table():
    """If the split-adjusted SQLite DB doesn't contain our 'Stocks' table, add it.
        This will be the main DB used for subsequent data analyis.

    Returns:
        None
    """

    try:
        # check if stocks table exists in split DB, if yes then drop it (since original DB might still change, e.g. more URLs retrieved)
        cur_split.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='Stocks'"
        )
        if cur_split.fetchone():
            cur_split.execute("DROP TABLE Stocks")

        # get the schema of the source table
        cur.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='Stocks'"
        )
        create_table_sql = cur.fetchone()[0]

        # create the table in the clean database
        cur_split.execute(create_table_sql)

        # copy data from source table to destination table (skip class A stocks)
        cur.execute("SELECT * FROM Stocks WHERE Security NOT LIKE '%Class A%'")
        rows = cur.fetchall()
        cur_split.executemany(
            f"INSERT INTO Stocks VALUES ({','.join(['?'] * len(rows[0]))})", rows
        )
        conn_split.commit()

    except sqlite3.Error as e:
        conn_split.rollback()
        if "database is locked" in str(e):
            raise sqlite3.Error(
                f"SQLite DB at '{sqlite_file_path_split}' is locked, please close the file and try again."
            )
        raise sqlite3.Error(
            f"Error: could not copy 'Stocks' table to '{sqlite_file_path_split}', please check and try again. \nError details:\n{e}"
        )


def skip_already_cleaned(symbols):
    """Checks if data for a stock was already cleaned in a previous run.
    This is done so that these data are not processed over and over again in the next batches.

    Args:
        symbols (list): List of all stock symbols for which a base url was previously obtained.

    Returns:
        symbols (list): List of stock symbols after exclusion of stocks for which data was already adjusted as necessary.
    """

    # if records DB does not contain tables, create them:

    cur_record.execute(
        """CREATE TABLE IF NOT EXISTS Splits
                       (Symbol TEXT, 
                       Date DATE,
                       SplitRatio TEXT,
                       SR_Float FLOAT,
                       PRIMARY KEY (Symbol, Date))"""
    )

    cur_record.execute(
        """CREATE TABLE IF NOT EXISTS Dividends
                       (Symbol TEXT, 
                       Date DATE,                                     
                       Dividend TEXT,
                       DivPriceRatio FLOAT,
                       PRIMARY KEY (Symbol, Date))"""
    )

    cur_record.execute(
        """CREATE TABLE IF NOT EXISTS Adjustments
                       (Symbol TEXT PRIMARY KEY, 
                       Split INTEGER,
                       Dividend INTEGER)"""
    )

    # find which data was already processed
    cur_record.execute(
        "SELECT Symbol FROM Adjustments WHERE Split NOT NULL AND Dividend NOT NULL"
    )
    already_cleaned = [symbol[0] for symbol in cur_record.fetchall() if symbol]

    #update stocks list to be processed 
    symbols = [symbol for symbol in symbols if symbol not in already_cleaned]
    if not symbols:
        print(
            f"""\nNo more relevant data left to pre-process:
    Split- and/or dividend adjustments performed for all {len(already_cleaned)} stocks for which data is available in '{sqlite_file_name}'.
    Program completed successfully.\n\n"""
        )
        sys.exit(0)

    return symbols


def load_dfs_dict(symbols):
    """Imports SQL data to Pandas DataFrames, sorts them by df.Date (ascending, for plotting), and stores these dfs in a dictionary.

    Args:
        symbols (list): Symbols of stocks for which data needs to be checked and potentially adjusted.

    Returns:
        dfs (dict): A dictionary storing all datasets under consideration: key = stock symbol (str), value = stock data (pandas.DataFrame)

    Raises: ValueError if dfs doesn't contain any data
    """

    #populate the dfs dict with pandas DataFrames containing the trading data for each stock
    dfs = dict()
    for symbol in symbols:
        df = pd.read_sql(f"SELECT * FROM '{symbol}'", conn)
        df.sort_values("Date", inplace=True, ignore_index=True)
        dfs[symbol] = df

    if len(dfs) == 0:
        raise ValueError("No data selected for processing, please check and try again.")

    print(
        f"\nData will be checked and adjusted as necessary for {len(dfs)} stocks:\n{list(dfs.keys())}"
    )
    return dfs


def check_dtypes(dfs, flt_series_list, int_series):
    """Basic test to ensure that data types of dfs are as expected.

    Args:
        dfs (dict): A dictionary storing all datasets under consideration: key = stock symbol (str), value = stock data (pandas.DataFrame).
        flt_series_list (list): Names of df columns (strs) that are expected to contain float64 data.
        int_series (str): Name of df column that is expected to contain int64 data.

    Returns:
        None

    Raises:
        TypeError: If series dtype not as expected.
    """

    for (
        symbol,
        df,
    ) in dfs.items():
        for series in flt_series_list:
            if df[series].dtype != "float64":
                raise TypeError(
                    f"Unexpected value encountered, check series {series.name} for df of stock {symbol} - dtype should be float64."
                )
        if df[int_series].dtype != "int64":
            raise TypeError(
                f"Unexpected value encountered, check series {int_series.name} for df of stock {symbol} - dtype should be int64."
            )

    print("Data types OK.\n")


def get_splits(dfs, play_nice):
    """
    Extracts info about splits from the relevant webpage.

    Args:
        dfs (dict): A dictionary storing all datasets under consideration: key = stock symbol (str), value = stock data (pandas.DataFrame)
        play_nice (float): Initial waiting time (s) between requests - slightly prolonged with each request, some randomness added.

    Returns:
        split_df (pandas.DataFrame): A DataFrame holding info on retrieved split events
        splits_manual_input (list): List of tuples containing info regarding stocks for which a webpage couldn't be retrieved from companiesmarketcap.com.
            tuple elements: (company name, stock symbol, url, response status code)
    """

    #create df to populate with split events
    split_df = pd.DataFrame(columns=["Stock", "Date", "SplitRatio"])
    
    #create list to hold names of stocks for which split data could not be obtained using this function
    splits_manual_input = [] 
    
    for i, (symbol, df) in enumerate(dfs.items()): #for each stock

        company = Company(symbol) #get Company instance        
        url = company.base_url + "stock-splits" #expected url for stock-split/consolidation events for this stock
        
        if i > 0:
            time.sleep(play_nice + i*0.01+np.random.uniform(0.0, 0.1))  # wait between requests

        #make request to companiesmarketcap.com:
        response = requests.get(url)
        if i + 1 < len(dfs):
            print(
                f"Retrieving stock split/consolidation events - {i+1} requests sent to companiesmarketcap.com so far in the current batch...",
                end="\r",
            )
        else:
            print(
                f"Stock split/consolidation events retrieval completed - {i+1} requests sent to companiesmarketcap.com so far in the current batch..."
            )
        if not response.ok:
            splits_manual_input.append(
                (company.name, symbol, url, response.status_code)
            )  # mark need for manual input
            continue  # move on to next stock
        
        #if response ok, get content of webpage and look for split table:
        content = response.content
        soup = BeautifulSoup(content, "html.parser")
        tables = soup("table")
        if not tables:
            continue  # no table found, move on to next stock

        #if there are tables, locate the correct one:
        split_table = None
        for t, table in enumerate(tables):
            for header in table("th"):
                if "Split" in header.string:
                    split_table = tables[t]
                    break
        if not split_table:
            continue  # no table containing split events, move on to next stock

        #go through table rows and get split dates and split ratios:
        rows = split_table("tr")
        dates = []
        ratios = []
        for row in rows[
            1:
        ]:  # go through all rows (except headers) and get split dates and ratios
            cells = row("td")
            dates.append(cells[0].get_text())
            ratios.append(cells[1].get_text())

        # if split date in relevant historical period, add it to the splits df:
        for date, ratio in zip(dates, ratios):
            if df.Date.str.contains(date).any():
                split_df.loc[len(split_df)] = [symbol, date, ratio]

    return split_df, splits_manual_input


def split_query(split_df, splits_manual_input, dfs):
    """If could not access split info from website for specific stocks (request status not OK), get user input on these stocks.

    Args:
        split_df (pandas.DataFrame): A DataFrame holding info on detected split events.
        splits_manual_input (list): List of tuples containing info regarding stocks for which a webpage couldn't be retrieved from companiesmarketcap.com.
            tuple elements: (company name, stock symbol, url, response status code)

    Returns:
        split_df (pandas.DataFrame): The (user-accepted or modified) split events DataFrame.
    """

    print(f"Info could not be accessed for {len(splits_manual_input)} stocks:\n")

    #for each tuple in the list, extract info:
    for tup in splits_manual_input: 
        company_name, symbol, url, status_code = tup
        print(
            f"{company_name} (ticker symbol '{symbol}'): Webpage {url} could not be accessed - code : {status_code}"
        )

        while True:  # for each stock, you can add multiple events
            manual_input = input(
                "-To manually insert split data for this stock, input split date (YYYY-MM-DD) and split ratio (after:before), separated by space(s) - e.g., 2022-07-18	20:1.\n-To skip this stock enter 'S'.\n-To accept the current split data for all stocks enter 'C'.\n"
            ).strip()
            if manual_input.upper() == "C":
                # accept split_df in its present form
                split_df.sort_values(["Stock", "Date"], inplace=True, ignore_index=True)
                print(f"Split events table:\n{split_df}\n") #show current state of split_df
                return split_df
            if manual_input.upper() == "S":
                break  # get out of while loop - move to next stock
            
            addition = manual_input.split() #data of single split event added by user

            #check user input validity; if ok, input to split_df after user confirmation
            if (
                len(addition) != 2
                or len(addition[0]) != 10
                or ":" not in addition[1]
            ):
                print("***Invalid input format, please check and try again***")
                continue  # ask for new input

            date, ratio = addition #addition contains info on date and ratio of split

            if not dfs[symbol].Date.str.contains(date).any():
                print("***Inputted date not in range of historical period***")
                continue  # ask for new input
            if len(ratio.split(":")) == 2:
                try:
                    _, _ = int(ratio.split(":")[0]), int(
                        ratio.split(":")[1]
                    )
                    temp_df = pd.DataFrame(
                        {
                            "Stock": symbol,
                            "Date": date,
                            "SplitRatio": ratio,
                        },
                        index=[0],
                    )
                    confirm = input(
                        f"Are you sure you want to add the following split event?\n{temp_df}\n[Y/N]"
                    )
                    while True:
                        if confirm.upper() == "Y":
                            split_df.loc[len(split_df)] = [
                                symbol,
                                date,
                                ratio,
                            ]
                            break  # get out of confirmation while loop
                        elif confirm.upper() == "N":
                            break  # get out of confirmation while loop
                        else:
                            print("Invalid response, please enter 'Y' or 'N':\t")

                except:
                    print("***Invalid ratio inputted***")
                    continue  # ask for new input
            else:
                print("***Invalid ratio inputted***")
                continue  # ask for new input


def adjust_for_split(dfs, split_df):
    """Uses info found in split_df to split-adjust data.

    Args:
        dfs (dict): A dictionary storing all datasets under consideration: key = stock symbol (str), value = stock data (pandas.DataFrame).
        split_df (pandas.DataFrame): A DataFrame holding info on user-accepted split events.

    Returns:
        adjusted_dfs (dict): A dictionary storing all datasets in which a split adjustment was made:
            key = stock symbol (str), value = split-adjusted stock data (pandas.DataFrame).
    """

    adjusted_dfs = dict() #to hold info on stocks with split adjustments
    columns_to_divide = ["Open", "Close", "Low", "High", "DailyRange"]
    #this is the data that should be divided when there's a stock split

    # go through all rows of split_df and adjust trading data from dfs as necessary:
    for _, row in split_df.iterrows():
        symbol = row.Stock
        split_date = row.Date
        ratio_num1, ratio_num2 = row.SplitRatio.split(":")
        split_ratio = float(ratio_num1) / float(ratio_num2)

        #if there were multiple split events for this stock, multiple adjustments are required:
        if symbol not in adjusted_dfs:
            adj_df = dfs[symbol].copy()
        else:
            adj_df = adjusted_dfs[symbol]
        
        #adjust data for splits:
        adj_df.loc[adj_df.Date < split_date, columns_to_divide] /= split_ratio
        adj_df.loc[adj_df.Date < split_date, "Volume"] *= split_ratio

        adjusted_dfs[symbol] = adj_df  # store split-adjusted data

    print(
        f"Data has been split-adjusted for the following {len(adjusted_dfs)} stocks : {list(adjusted_dfs.keys())}\n"
    )

    return adjusted_dfs


def populate_split_records(dfs, split_df):
    """Populates SQLite records DB with info regarding split adjustment.

    Args:
        dfs (dict): A dictionary storing all datasets under consideration: key = stock symbol (str), value = stock data (pandas.DataFrame).
        split_df (pandas.DataFrame): A DataFrame holding info on split events.

    Returns:
        None

    Raises:
        sqlite3.OperationalError: If an error occurs while populating the SQL DB.
            If the DB is locked, the user is prompted to close the SQLite file or quit the program.
    """

    while True:

        try:
            # populate Splits table in SQLite record DB:
            for _, row in split_df.iterrows():
                sr_float = float(row.SplitRatio.split(":")[0]) / float(
                    row.SplitRatio.split(":")[1]
                )
                if overwrite_clean_tables: #if user wishes to overwrite, delete relevant rows if exist
                    cur_record.execute(
                        "DELETE FROM Splits WHERE Symbol = ?", (row.Stock,)
                    )
                cur_record.execute(
                    "INSERT OR IGNORE INTO Splits (Symbol, Date, SplitRatio, SR_Float) VALUES (?, ?, ?, ?)",
                    (row.Stock, row.Date, row.SplitRatio, sr_float),
                )

            # enter symbols of processed stocks to adjustments table if they do not already exist:
            for symbol in dfs:
                cur_record.execute(
                    "INSERT INTO Adjustments (Symbol) SELECT ? WHERE NOT EXISTS (SELECT 1 FROM Adjustments WHERE Symbol = ?)",
                    (symbol, symbol),
                )

                # check if stock data was adjusted for splits or not, and update the adjustments table as necessary:
                if symbol in split_df.Stock.values:
                    cur_record.execute(
                        "UPDATE Adjustments SET Split = 1 WHERE Symbol = ?", (symbol,)
                    )
                else:
                    cur_record.execute(
                        "UPDATE Adjustments SET Split = 0 WHERE Symbol = ?", (symbol,)
                    )

            conn_record.commit()
            break  # successfully populated records SQL, break out of while loop

        except sqlite3.OperationalError as e:
            if "database is locked" in str(e):
                retry = input(
                    "Database is locked - please close the SQLite file and press ENTER to try again, or enter 'Q' to quit program.\t"
                ).strip()
                if retry.upper() == "Q":
                    print("\n***Program terminated by user***\n")
                    sys.exit()
                continue  # try again
            else:
                raise sqlite3.OperationalError(
                    f"\nError encountered, program terminated:\n{e}\n"
                )


def get_dividends(dfs, play_nice):
    """
    Extracts info about dividends from the relevant webpage.

    Args:
        dfs (dict): A dictionary storing all datasets under consideration: key = stock symbol (str), value = stock data (pandas.DataFrame)
        play_nice (float): Initial waiting time (s) between requests - slightly prolonged with each request, some randomness added.

    Returns:
        dividend_df (pandas.DataFrame): A DataFrame holding info on retrieved dividend events
        dividend_manual_input (list): List of tuples containing info regarding stocks for which a webpage couldn't be retrieved from companiesmarketcap.com.
            tuple elements: (company name, stock symbol, url, response status code)
    """

    #initiate df for storing dividend events
    dividend_df = pd.DataFrame(columns=["Stock", "Date", "Dividend", "DivPriceRatio"])

    #initiate list for storing stocks for which dividend events could not be retrieved from companiesmarketcap.com
    dividend_manual_input = []

    # expected headers for dividend_table on webpage
    target_headers = [
        "Date",
        "Dividend (stock split adjusted)",
        "Change",
    ]  

    for i, (symbol, df) in enumerate(dfs.items()): #for each stock

        company = Company(symbol) #initiate Company instance
        url = company.base_url + "dividends" #expected url for dividend info
        dividend_table = None  # default value - no dividends table in page

        if i > 0:
            time.sleep(play_nice + i*0.01+np.random.uniform(0.0, 0.1))  # wait between requests
            
        #get response from webpage:
        response = requests.get(url)
        if i + 1 < len(dfs):
            print(
                f"Retrieving dividend payment events - {len(dfs)+i+1} requests sent to companiesmarketcap.com so far in the current batch...",
                end="\r",
            )
        else:
            print(
                f"Dividend events retrieval completed - {len(dfs)+i+1} requests sent to companiesmarketcap.com so far in the current batch..."
            )
        if not response.ok:
            dividend_manual_input.append(
                (company.name, symbol, url, response.status_code)
            )  # mark need for manual input
            continue  # move on to next stock

        #if response ok, get content and check for tables
        content = response.content
        soup = BeautifulSoup(content, "html.parser")
        tables = soup("table")

        if not tables:
            continue  # no table containing dividend events found, move on to next stock

        # first test to find correct table - look for table below suitable header
        h3_tags = soup("h3")
        for h3_tag in h3_tags:
            if "all dividend" in h3_tag.text:
                dividend_table = h3_tag.find_next_sibling("table")
                break
        else:
            continue  # no table containing dividend events found, move on to next stock

        # second test to find correct table - check that table headers are as expected
        rows = dividend_table("tr")
        table_headers = [header.text for header in rows[0]("th")]

        if target_headers != table_headers:
            continue  # no table containing dividend events found, move on to next stock

        #get dates and dividend values from dividend table
        dates = []
        dividends = []
        for row in rows[
            1:
        ]:  # go through all rows (except headers) and get split dates and ratios
            cells = row("td")
            dates.append(cells[0].get_text())
            dividends.append(cells[1].get_text())

        #add relevant dates and dividend values to dividend_df
        for date, div in zip(dates, dividends):
            if df.Date.str.contains(date).any():
                price = df.Close.loc[(df.Date == date)].values[0]
                dividend_df.loc[len(dividend_df)] = [
                    symbol,
                    date,
                    div,
                    np.round(float(div[1:]) / price, 5),
                ]

    return dividend_df, dividend_manual_input


def dividend_query(dividend_df, dividend_manual_input, dfs):
    """If could not access dividend info from website for specific stocks (request status not OK), get user input on these stocks.

    Args:
        dividend_df (pandas.DataFrame): A DataFrame holding info on detected dividend events.
        dividend_manual_input (list): List of tuples containing info regarding stocks for which a webpage couldn't be retrieved from companiesmarketcap.com.
            tuple elements: (company name, stock symbol, url, response status code)
        dfs (dict): A dictionary storing all datasets under consideration: key = stock symbol (str), value = stock data (pandas.DataFrame)
        split_adjusted_dfs (dict): A dictionary storing all datasets in which a split adjustment was made:
            key = stock symbol (str), value = split-adjusted stock data (pandas.DataFrame).

    Returns:
        split_df (pandas.DataFrame): The (user-accepted or modified) split events DataFrame.
    """

    print(f"\nInfo could not be accessed for {len(dividend_manual_input)} stocks:\n")

    #for each tuple in the list, extract info:
    for tup in dividend_manual_input:
        company_name, symbol, url, status_code = tup
        print(
            f"{company_name} (ticker symbol '{symbol}'): Webpage {url} could not be accessed - code : {status_code}"
        )

        while True:  # for each stock, you can add multiple events
            manual_input = input(
                "-To manually insert dividend data for this stock, input dividend date (YYYY-MM-DD) and amount ($D.cc), separated by space(s) - e.g., 2023-11-16	$1.50.\n-To skip this stock enter 'S'.\n-To accept the current dividend data for all stocks enter 'C'.\n"
            ).strip()
            if manual_input.upper() == "C":
                # accept dividend_df in its present form and print it
                dividend_df.sort_values(
                    ["Stock", "Date"], inplace=True, ignore_index=True
                )
                print(f"Dividend events table:\n{dividend_df}\n")
                return dividend_df
            if manual_input.upper() == "S":
                break  # get out of while loop - move to next stock

            addition = manual_input.split() #user input for this dividend event

            #check validity of user input; if ok, add to dividend_df after user confirmation:
            if (
                len(addition) != 2
                or len(addition[0]) != 10
                or "." not in addition[1]
                or not addition[1].startswith("$")
            ):
                print("***Invalid input format, please check and try again***\n")
                continue  # ask for new input

            date, div = addition  # addition contains valid date and dividend data

            if not dfs[symbol].Date.str.contains(date).any():
                print("***Inputted date not in range of historical period***\n")
                continue  # ask for new input
            if len(div.split(".")) == 2:
                try:
                    _, _ = int(div.split(".")[0][1:]), int(div.split(".")[1])
                    temp_df = pd.DataFrame(
                        {
                            "Stock": symbol,
                            "Date": date,
                            "Dividend": div,
                        },
                        index=[0],
                    )
                    confirm = input(
                        f"Are you sure you want to add the following dividend event?\n{temp_df}\n[Y/N]"
                    )
                    while True:
                        if confirm.upper() == "Y":
                            price = (
                                dfs[symbol]
                                .Close.loc[(dfs[symbol].Date == date)]
                                .values[0]
                            )
                            dividend_df.loc[len(dividend_df)] = [
                                symbol,
                                date,
                                div,
                                np.round(float(div[1:]) / price, 5),
                            ]
                            break  # get out of confirmation while loop
                        elif confirm.upper() == "N":
                            break  # get out of confirmation while loop without changing dividend_df
                        else:
                            print("Invalid response, please enter 'Y' or 'N':\t")

                except:
                    print("***Invalid dividend amount inputted***\n")
                    continue  # ask for new input
            else:
                print("***Invalid dividend amount inputted***")
                continue  # ask for new input


def adjust_for_dividends(dfs, split_adjusted_dfs, dividend_df):
    """Uses info found in dividends_df to dividend-adjust data.

    Args:
        dfs (dict): A dictionary storing all datasets under consideration: key = stock symbol (str), value = stock data (pandas.DataFrame).
        split_adjusted_dfs (dict): A dictionary storing all datasets in which a split adjustment was made:
            key = stock symbol (str), value = split-adjusted stock data (pandas.DataFrame).
        dividend_df (pandas.DataFrame): A DataFrame holding info on detected dividend events.

    Returns:
        adjusted_dfs (dict): A dictionary storing all datasets in which a split and/or dividend adjustment was made:
            key = stock symbol (str), value = split- and dividend-adjusted stock data (pandas.DataFrame).
    """

    
    dividend_adjusted_dfs = dict() #to hold info on stocks with dividend adjustments
    affected_columns = ["Open", "Close", "Low", "High"] #columns to adjuste for dividends

    # go through all rows of dividend_df and adjust trading data from dfs as necessary:
    for _, row in dividend_df.iterrows():
        symbol = row.Stock
        dividend_date = row.Date
        dividend_amount = float(row.Dividend[1:])
                
        if symbol in split_adjusted_dfs: # use split-adjusted df if exists
            df = split_adjusted_dfs[symbol]
        else:
            df = dfs[symbol]

        #if there were multiple dividend events for this stock, multiple adjustments are required:
        if symbol not in dividend_adjusted_dfs:
            adj_df = df.copy()
        else:
            adj_df = dividend_adjusted_dfs[symbol]

        # get original closing price on dividend day:
        og_close = float(adj_df.Close.loc[(adj_df.Date == dividend_date)].iloc[0])

        #calculate dividend adjustment ratio and adjust relevant data:
        adjustment_ratio = np.float64((og_close - dividend_amount) / og_close)
        adj_df.loc[adj_df.Date < dividend_date, affected_columns] *= adjustment_ratio

        # store split- and dividend-adjusted data:
        dividend_adjusted_dfs[symbol] = (
            adj_df  
        )
        dividend_adjusted_dfs[symbol].DailyChange = round(
            100
            * dividend_adjusted_dfs[symbol].Close
            / dividend_adjusted_dfs[symbol].Open
            - 100,
            4,
        )

    print(
        f"Data has been dividend-adjusted for the following {len(dividend_adjusted_dfs)} stocks : {list(dividend_adjusted_dfs.keys())}\n"
    )

    return dividend_adjusted_dfs


def populate_dividend_records(dfs, dividend_df):
    """Populates SQLite records DB with info regarding split adjustment.

    Args:
        dfs (dict): A dictionary storing all datasets under consideration: key = stock symbol (str), value = stock data (pandas.DataFrame).
        dividend_df (pandas.DataFrame): A DataFrame holding info on detected dividend events.

    Returns:
        None

    Raises:
        sqlite3.OperationalError: If an error occurs while populating the SQL DB.
            If the DB is locked, the user is prompted to close the SQLite file or quit the program.
    """

    while True: #to allow user to close DB file and continue

        try:
            # populate Dividends table:
            for _, row in dividend_df.iterrows():
                if overwrite_clean_tables: #if user wishes to overwrite, delete relevant rows if exist
                    cur_record.execute(
                        "DELETE FROM Dividends WHERE Symbol = ?", (row.Stock,)
                    )
                cur_record.execute(
                    "INSERT OR IGNORE INTO Dividends (Symbol, Date, Dividend, DivPriceRatio) VALUES (?, ?, ?, ?)",
                    (row.Stock, row.Date, row.Dividend, row.DivPriceRatio),
                )

            # enter symbols of processed stocks to adjustments table if they do not already exist:
            for symbol in dfs:
                cur_record.execute(
                    "INSERT INTO Adjustments (Symbol) SELECT ? WHERE NOT EXISTS (SELECT 1 FROM Adjustments WHERE Symbol = ?)",
                    (symbol, symbol),
                )

                # check if stock data was adjusted for dividends or not, and update the adjustments table as necessary
                if symbol in dividend_df.Stock.values:
                    cur_record.execute(
                        "UPDATE Adjustments SET Dividend = 1 WHERE Symbol = ?",
                        (symbol,),
                    )
                else:
                    cur_record.execute(
                        "UPDATE Adjustments SET Dividend = 0 WHERE Symbol = ?",
                        (symbol,),
                    )

            conn_record.commit()
            break  # successfully populated records SQL, break out of while loop

        except sqlite3.OperationalError as e:
            if "database is locked" in str(e):
                retry = input(
                    "Database is locked - please close the SQLite file and press ENTER to try again, or enter 'Q' to quit program.\t"
                ).strip()
                if retry.upper() == "Q":
                    print("\n***Program terminated by user***\n")
                    sys.exit()
                continue  # try again
            else:
                raise sqlite3.OperationalError(
                    f"\nError encountered, program terminated:\n{e}\n"
                )


def store_clean_data(dfs, split_adjusted_dfs, adjusted_dfs):
    """Store split- and (split-and-)dividend-adjusted price data in designated SQL DBs.
        The guiding principle here is: It almost always makes sense to look at split-adjusted data. 
        However, whether one should look at dividend-adjusted data or not depends on one's goals. 
        Therefore, the split-adjusted DB will be used for most subsequent analyses, while the 
        (split-and)-dividend-adjusted DB will be used in more specific cases. 

    Args:
        dfs (dict): A dictionary storing all datasets under consideration:
            key = stock symbol (str), value = stock data (pandas.DataFrame).
        split_adjusted_dfs (dict): A dictionary storing all datasets in which a split adjustment was made:
            key = stock symbol (str), value = split-adjusted stock data (pandas.DataFrame).
        adjusted_dfs (dict): A dictionary storing all datasets in which a split and/or dividend adjustment was made:
            key = stock symbol (str), value = split- and/or dividend-adjusted stock data (pandas.DataFrame).

    Returns:
        None

    Raises:
        sqlite3.OperationalError: If an error occurs while populating the SQL DB.
            If the DB is locked, the user is prompted to close the SQLite file or quit the program.
    """

    print("Storing data in SQLite files.....")
    
    while True: #to allow user to close DB file(s) and continue

        try:
            for symbol, df in dfs.items(): #for each stock
                
                # if stock data was split-adjusted, use the adjusted data; otherwise store the raw data:
                if symbol in split_adjusted_dfs:
                    df = split_adjusted_dfs[symbol]
                
                #sort values as they are in the original SQL data
                df.sort_values("Date", ascending=False).to_sql(
                    symbol, conn_split, index=False, if_exists="replace"
                )

            # only stocks for which dividend adjustment was made are stored in the designated SQL DB:
            for symbol, df in adjusted_dfs.items():
                df.sort_values("Date", ascending=False).to_sql(
                    symbol, conn_split_div, index=False, if_exists="replace"
                )

            break  # if successful, escape while loop

        except sqlite3.OperationalError as e:
            if "database is locked" in str(e):
                retry = input(
                    "Database is locked - please close SQLite file(s) and press ENTER to try again, or enter 'Q' to quit program.\t"
                ).strip()
                if retry.upper() == "Q":
                    print("\n***Program terminated by user***\n")
                    sys.exit()
                continue  # try again
            else:
                raise sqlite3.OperationalError(
                    f"\nError encountered, program terminated:\n{e}\n"
                )

    conn_split.commit()
    print(
        f"Inserted {len(dfs)} tables (including {len(split_adjusted_dfs)} tables contaning split-adjusted data) into '{sqlite_file_name_split}' in this run."
    )
    cur_split.execute(
        "SELECT name FROM sqlite_master WHERE type='table' and name!='Stocks'"
    )
    all_split_adjusted = [table[0] for table in cur_split.fetchall()]
    print(
        f"The SQLite DB storing split-adjusted prices now contains info on {len(all_split_adjusted)} stocks."
    )

    conn_split_div.commit()
    print(
        f"\nInserted {len(adjusted_dfs)} dividend-adjusted tables into '{sqlite_file_name_split_div}' in this run."
    )
    cur_split_div.execute("SELECT name FROM sqlite_master WHERE type='table'")
    all_split_div_adjusted = [table[0] for table in cur_split_div.fetchall()]
    print(
        f"The SQLite DB storing dividend-adjusted prices now contains info on {len(all_split_div_adjusted)} stocks.\n"
    )


def get_all_adjusted(dfs, split_adjusted_dfs, dividend_adjusted_dfs):
    """Combines all adjusted dfs into one dictionary for plotting.

    Args:
        dfs (dict): A dictionary storing all datasets under consideration:
            key = stock symbol (str), value = stock data (pandas.DataFrame).
        split_adjusted_dfs (dict): A dictionary storing all datasets in which a split adjustment was made:
            key = stock symbol (str), value = split-adjusted stock data (pandas.DataFrame).
        dividend_adjusted_dfs (dict): A dictionary storing all datasets in which a (split and) dividend adjustment was made:
            key = stock symbol (str), value = split- and/or dividend-adjusted stock data (pandas.DataFrame).

    Returns:
        all_adjusted_dfs (dict): A dictionary storing all datasets in which any type of adjustment was made:
            key = stock symbol (str), value = split-adjusted stock data (pandas.DataFrame).
    """

    #for each stock, find out if it was adjusted and store it in the new dict:
    all_adjusted_dfs = {}    
    for symbol in dfs: 
        # some stocks are split adjusted but not dividend adjusted
        if symbol in split_adjusted_dfs:
            all_adjusted_dfs[symbol] = split_adjusted_dfs[symbol]
        # by definition, dividend-adjusted dfs are also split adjusted if necessary
        if symbol in dividend_adjusted_dfs:
            all_adjusted_dfs[symbol] = dividend_adjusted_dfs[symbol]

    return all_adjusted_dfs


def plot_price_adjustments(adjusted_dfs, dfs, split_df, dividend_df):
    """Visualizes (close) prices for all data - adjusted for splits and dividends as required.

    Args:
        adjusted_dfs (dict): A dictionary storing all datasets in which a split and/or dividend adjustment has been made:
            key = stock symbol (str), value = adjusted stock data (pandas.DataFrame).
        dfs (dict): A dictionary storing all datasets under consideration: key = stock symbol (str), value = stock data (pandas.DataFrame).
            Used for showing raw price data.
        split_df (pandas.DataFrame): A DataFrame holding info on split events.
        dividend_df (pandas.DataFrame): A DataFrame holding info on dividend events.

    Returns:
        None
    """

    print("Producing figures....")

    stocks_per_fig = 20  # maximum number of stocks to be displayed in a single figure
    width, height = 7, 5 # width and height of each subplot
    max_col_num = 5  # maximum number of columns per fig
    figs = []  # list for storing figures in case several are produced
    symbol_list = list(dfs.keys()) #list of stock ticker symbols
    first_last_stock = []  # list for storing names of stocks for figure names

    # find the minimum and maximum dividend/price ratio across all stocks in batch for adjusting marker size:
    min_dividend_price_ratio = dividend_df.DivPriceRatio.min()
    max_dividend_price_ratio = dividend_df.DivPriceRatio.max()
    # min and max marker sizes:
    min_marker_size = 10
    max_marker_size = 100

    lw = 2 #line width for line plots

    for i, symbol in enumerate(dfs): #for each stock

        # create a new figure for every stocks_per_fig subplots:
        if (
            i % stocks_per_fig == 0
        ):  
            last_stock_in_fig = min(len(dfs) - 1, (i + stocks_per_fig - 1))
            first_last_stock.append(
                f"{symbol_list[i]}-{symbol_list[last_stock_in_fig]}"
            )

            j = 0 #reset axes index 
            num_plots = min(
                stocks_per_fig, len(dfs) - i
            )  # number of subplots per fig - one subplot per df

            # number of columns and rows in fig:
            num_cols = min(len(dfs) - i, max_col_num)
            num_rows = int(np.ceil(num_plots / num_cols))  

            #initialize subplots for this figure:
            fig, axes = plt.subplots(
                num_rows,
                num_cols,
                figsize=(width * num_cols, height * num_rows),
                layout="constrained",
                gridspec_kw={"wspace": 0.1, "hspace": 0.1},
            )

            figs.append(fig)  # store the created figure in a list

            # flatten axes if there's more than one subplot; otherwise put axes in a list
            if num_plots > 1:
                axes = axes.flatten()
            else:
                axes = [axes]

        # access the appropriate subplot and increment counter j
        ax = axes[j]  
        j += 1

        #determine min and max values of subplot x and y axes:
        max_x = dfs[symbol].Date.max()
        min_x = dfs[symbol].Date.min()
        if symbol in adjusted_dfs:
            max_y = max(dfs[symbol].Close.max(), adjusted_dfs[symbol].Close.max()) * 1.1
            min_y = min(dfs[symbol].Close.min(), adjusted_dfs[symbol].Close.min()) * 0.9
        else:
            max_y = dfs[symbol].Close.max() * 1.1
            min_y = dfs[symbol].Close.min() * 0.9

        #plot unadjusted stock price history:
        ax.plot(
            dfs[symbol].Date,
            dfs[symbol].Close,
            color="tab:blue",
            lw=lw,
            label="Stock Price",
        )

        #plot adjusted stock price history when relevant:
        if symbol in adjusted_dfs:
            ax.plot(
                adjusted_dfs[symbol].Date,
                adjusted_dfs[symbol].Close,
                color="tab:red",
                lw=lw,
                label="Adjusted Price",
                alpha=0.8,
            )

        # mark stock split/consolidation event(s) as vertical line(s):        
        split_dates = [
            split_date for split_date in split_df.loc[split_df.Stock == symbol, "Date"]
        ] #get split date(s) 
        event_type_plotted = {}  # for preventing multiple legends for same event type
        
        #for each split event in period, find date and whether it was a split or consolidation
        for date in split_dates: 
            after, before = (
                split_df.loc[
                    (split_df.Date == date) & (split_df.Stock == symbol), "SplitRatio"
                ]
                .values[0]
                .split(":")
            )
            if after > before:
                event_type = "Split"
                color = "black"
            else:
                event_type = "Consolidation"
                color = "tab:purple"

            # plot vertical line to indicate split event; add event type to legend if it's not already there:
            if event_type not in event_type_plotted:
                ax.plot(
                    [date, date],
                    [0, max_y],
                    color=color,
                    linestyle="--",
                    alpha=0.5,
                    lw=lw,
                    label=event_type,
                )  
                event_type_plotted[event_type] = True
            else:
                ax.plot(
                    [date, date],
                    [0, max_y],
                    color=color,
                    linestyle="--",
                    alpha=0.5,
                    lw=lw,
                )

        # mark dividend event(s) as green circles - circle size indicates normalized dividend value:
        dividend_dates = [
            dividend_date
            for dividend_date in dividend_df.loc[dividend_df.Stock == symbol, "Date"]
        ] # dividend date(s) in period

        prices_on_div_dates = [] #used to place dividend marker on adjusted price line

        #calculate normalized dividend value and set marker size accordingly:
        normalized_div_ratios = (
            dividend_df.DivPriceRatio.loc[(dividend_df.Stock == symbol)]
            - min_dividend_price_ratio
        ) / (max_dividend_price_ratio - min_dividend_price_ratio)
        sizes = (
            min_marker_size
            + normalized_div_ratios * (max_marker_size - min_marker_size)
        ).tolist()

        #if there are dividend events for this stock, pad dates to align with price data:
        if dividend_dates: 

            if dividend_dates[0] > min_x:
                dividend_dates.insert(0, min_x)
                sizes.insert(0, 0.0)

            if dividend_dates[-1] < max_x:
                dividend_dates.append(max_x)
                sizes.append(0.0)

            #place dividend marker(s) on adjusted price line:
            for date in dividend_dates:
                prices_on_div_dates.append(
                    float(
                        adjusted_dfs[symbol]
                        .Close.loc[(adjusted_dfs[symbol].Date == date)]
                        .iloc[0]
                    )
                )

            #mark dividends as green circles
            ax.scatter(
                dividend_dates,
                prices_on_div_dates,
                s=sizes,
                color="tab:green",
                alpha=0.8,
                zorder=5,
                label="Dividend Payment",
            )

        #set subplot axes, titles, etc.:
        ax.set_title(symbol)
        ax.set_xlabel("Date")
        ax.set_ylabel("Stock Price (USD)")
        ax.tick_params(axis="x", labelrotation=45)
        ax.set_ylim(top=max_y, bottom=min_y)
        ax.xaxis.set_major_locator(plt.MultipleLocator(len(dfs[symbol]) // 5))
        ax.legend()
        #the size of the dividend marker in the legend should be the same for all subplots:
        for handle in ax.get_legend().legend_handles: 
            if handle.get_label() == "Dividend Payment":
                handle.set_sizes([(max_marker_size + min_marker_size) / 2])

    #set dir for storing figures
    fig_dir = os.path.join(current_dir, "adjustment_figures")
    os.makedirs(fig_dir, exist_ok=True)

    #save each figure using the symbols for the first and last stocks shown in it
    for i, fig in enumerate(figs):
        fig.savefig(
            f"{fig_dir}/figure_{first_last_stock[i]}.png"
        )  
        print(f"Figure saved: ./adjustment_figures/figure_{first_last_stock[i]}.png")
    print("\n")


def main():
    """Program for adjusting data to stock split/consolidation and dividend events."""

    #get relevant stock symbols from original SQLite database (produced by populate_SQL.py)
    symbols = get_original_symbols()

    try: #allows keyboard interruptions without losing existing data

        #copy Stocks table from original DB to split-adjusted DB
        copy_stocks_table()

        #skip data that has already been adjusted in a previous run, unless user chooses to overwrite
        if not overwrite_clean_tables:
            symbols = skip_already_cleaned(symbols)

        #if batch_size is not None, shorten list to batch_size
        if batch_size and len(symbols) > batch_size:
            symbols = symbols[:batch_size]

        #populate dfs dict with stock price history data obtained from the original SQLite database
        dfs = load_dfs_dict(symbols)

        #check that datatypes of 
        check_dtypes(
            dfs, ["Open", "Close", "DailyChange", "Low", "High", "DailyRange"], "Volume"
        )

        #set initial waiting time (s) between requests to companiesmarketcap.com (increases with number of requests)
        play_nice = 1.0 

        #get split data for stocks in batch from companiesmarketcap.com
        split_df, splits_manual_input = get_splits(dfs, play_nice)

        #if program failed to retrieve split data for specific stocks, ask user to enter it manually
        if splits_manual_input:
            split_df = split_query(split_df, splits_manual_input, dfs)

        #adjust data for stock split/consolidation events as needed
        split_adjusted_dfs = adjust_for_split(dfs, split_df)

        #indicate split adjustment in SQL DB used to track adjustments
        populate_split_records(dfs, split_df)

        #get dividend data for stocks in batch from companiesmarketcap.com
        dividend_df, dividend_manual_input = get_dividends(dfs, play_nice)

        #if program failed to retrieve dividend data for specific stocks, ask user to enter it manually
        if dividend_manual_input:
            dividend_df = dividend_query(dividend_df, dividend_manual_input, dfs)

        #adjust (split-adjusted) data for dividends as needed
        dividend_adjusted_dfs = adjust_for_dividends(
            dfs, split_adjusted_dfs, dividend_df
        )

        #indicate dividend adjustment in SQL DB used to track adjustments
        populate_dividend_records(dfs, dividend_df)

        #store (adjusted) data in appropriate SQLite databases
        store_clean_data(dfs, split_adjusted_dfs, dividend_adjusted_dfs)

        #pool all data that was adjusted in any way for plotting
        all_adjusted_dfs = get_all_adjusted(
            dfs, split_adjusted_dfs, dividend_adjusted_dfs
        )

        #create and save figures showing raw stock prices and prices after split- and/or dividend-adjustment if adjusted
        plot_price_adjustments(all_adjusted_dfs, dfs, split_df, dividend_df)

    except KeyboardInterrupt:
        raise KeyboardInterrupt("\n****Program terminated by user****\n")

    except Exception as e:
        raise Exception(f"{type(e).__name__} encountered:\n{e}")

    finally:
        # close connection to SQLite files
        conn.close()
        conn_split.close()
        conn_split_div.close()


"""Run program:"""
if __name__ == "__main__":
    main()
