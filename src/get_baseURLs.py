"""
Roni Hogri, March 2024

This program should be run after 'populate_SQL.py' was used to (at least partially) populate the SQLite file with trading history data.
The purpose of the current program is to use a custom Google search engine (requires a Google API key) to retrieve a "base URL" 
for each stock from https://companiesmarketcap.com/. 
Base URLs will be stored in the 'Stocks' table of the SQLite database populated by 'populate_SQL.py', 
and will be used by subsequent programs to retrieve different kinds of information from companiesmarketcap.com. 

**IMPORTANT NOTES:**
1. In order for the program to work, you have to get a Google API key: https://developers.google.com/custom-search/v1/introduction .
For the free version, the limit is 100 calls per day. This program will allow you to continue from where you stopped last time. 
Your key should be inserted as a string in the file 'google_api_key.py' instead of 'place_your_key_here'. 
Alternatively, you can add the key as an argument every time you run the program, e.g.: 'python3 get_baseURLs.py --key=mykey' .
2. In order to avoid wasting API calls on unusable data, stocks with a too-short history (i.e., newly issued companies)
will be excluded from further steps. You can adjust what is considered "too short" in the "user-definable variables" section below.
3. The "base URL" itself will not lead to a valid webpage (returns 404)! Required URL slugs will be added by subsequent programs.
"""

"""User-definable variables; modify as necessary:"""

batch_size = 60  # number of stocks to handle each time you run the program.
# set batch_size to None to go through all of your existing data in one batch (as much as possible depending on the type of your Google API key).

min_duration_ratio = 0.5 # minimal allowed length of timeseries as a fraction of the maximal timeseries.
# (for discarding stocks that have been traded for less than a certain duration from further consideration).

sqlite_file_name = "S&P 500.sqlite"  # name of SQLite DB file containing raw data (produced by 'populate_SQL.py')

"""End of user-defined variables."""


# import required libraries for global use
import sqlite3
import os
import requests
import time
import sys
import re
import argparse


"""Globals :"""

SEARCH_ENGINE_ID = "32a7e87106fe44a88"  # custom search engine for finding info on stocks on companiesmarketcap.com
play_nice = 1  # duration to wait between API requests (s)

try:
    from ticker_to_company import Company  # for getting company name for query
except:
    raise Exception(
        "Could not import the 'Company' class. Please ensure that 'ticker_to_company.py' exists in the current working directory and try again."
    )

#dir of current program
current_dir = os.path.dirname(os.path.abspath(__file__))

# connection and handle for SQLite DB created by 'populate_SQL.py'
sqlite_file_path = os.path.join(current_dir, sqlite_file_name)
conn = sqlite3.connect(sqlite_file_path)
cur = conn.cursor()


"""Functions:"""


def check_db():
    """Checks that SQLite DB exists in the expected location, and that it contains a 'Stocks' table.

    Returns:
        None

    Raises:
        Exception if SQLite DB does not exist in the specified path or does not contain a 'Stocks' table
    """

    if not os.path.exists(sqlite_file_path):
        raise Exception(
            "Could not find SQLite database. Make sure you already ran 'populate_SQL.py', and that all package files are stored in the same folder.\n***Program terminated***"
        )

    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='Stocks'")
    stock_table_exists = cur.fetchone()

    if not stock_table_exists:
        raise Exception(
            "Could not find SQLite database. Make sure you already ran 'populate_SQL.py', and that all package files are stored in the same folder.\n***Program terminated***"
        )


def add_columns():
    """Adds the 'Base_URL_MC' and 'Tot_Trade_Days' columns to the SQLite DB if they do not yet exist.

    Returns:
        None
    """

    # get existing columns from Stocks table:
    cur.execute("PRAGMA table_info(Stocks)")
    existing_columns = [column[1] for column in cur.fetchall()]

    new_cols = [
        ("Tot_Trade_Days", "INTEGER"),
        ("Base_URL_MC", "TEXT"),
    ]  # names and datatypes of new columns to be added to Stocks table

    for col in new_cols:
        while True:
            if col[0] not in existing_columns:  # if column doesn't already exist
                try:
                    # create a new column
                    cur.execute(f"ALTER TABLE 'Stocks' ADD COLUMN {col[0]} {col[1]}")
                    print(f"Column '{col[0]}' added to 'Stocks' table.")
                    break  # break out of while loop

                except sqlite3.Error as e:
                    if "database is locked" in str(e):
                        close_db = input(
                            "Cannot write URL to open SQLite file, please close file(s) and press ENTER, or enter 'Q' to interrupt program.\t"
                        ).strip()
                        if close_db == "":
                            continue
                        if close_db.upper() == "Q":
                            raise Exception("*****Program interrupted by user*****")
                    else:
                        raise sqlite3.Error(f"Error encountered:\n{e}")

            else:
                break  # column already exists, move on

    conn.commit()


def get_trading_duration():
    """For each stock, adds the total number of trading days in the analyzed period.
    This will be used to exclude stocks that have not been traded long enough ("too-short history") from further steps.

    Returns:
        list: A list of symbols for stocks with too-short history, to be excluded from further steps.
    """

    #get all stock symbols from Stocks table
    cur.execute("SELECT Symbol FROM Stocks")
    symbols = [symbol[0] for symbol in cur.fetchall()]

    #count trading days per stock
    for symbol in symbols:
        #check if table exists for this stock, otherwise move on to the next one
        cur.execute(f"SELECT name FROM sqlite_master WHERE name = '{symbol}'")
        if not cur.fetchone(): continue

        #count the number of trading days in stock history and update Stocks table
        cur.execute(f"SELECT COUNT(*) FROM '{symbol}'")
        cnt = cur.fetchone()[0]
        if cnt != 0:
            cur.execute(
                "UPDATE Stocks SET Tot_Trade_Days = ? WHERE Symbol = ?", (cnt, symbol)
            )

    conn.commit()

    #get a list of stocks with too-short history
    cur.execute(
        f"SELECT Symbol FROM Stocks WHERE Tot_Trade_Days < ((SELECT MAX(Tot_Trade_Days) FROM Stocks) * {min_duration_ratio})"
    )
    return [symbol[0] for symbol in cur.fetchall()]


def stocks_wo_url(too_short):
    """Retrieves stock symbols for which the base URL is not stored in the SQL file.

    Args:
        too_short (list): A list of symbols for stocks with too-short history, to be excluded from further steps

    Returns:
        list: A list (len=batch_size) containing stock symbols (str) for which base url is not stored in SQLite file
    """

    #get symbols of stocks for which trading data exists but base URL does not
    cur.execute(
        "SELECT Symbol FROM Stocks WHERE (Base_URL_MC IS NULL AND Tot_Trade_Days NOT NULL)"
    )
    no_url = [symbol[0] for symbol in cur.fetchall() if symbol[0] not in too_short]

    if not no_url:  # if list is empty
        cur.execute("SELECT COUNT(*) FROM Stocks WHERE Tot_Trade_Days IS NULL")
        num_stocks_wo_data = cur.fetchone()[0]
        if num_stocks_wo_data == 0:
            print(
                f"Base URL has been obtained for all stocks with valid data.\nURLs have not been retrieved for {len(too_short)} stocks for which not enough data exists :\n{too_short}\n\nProgram completed successfully.\n"
            )
            sys.exit(0)
        else:
            print(
                f"Base URL has been obtained for all stocks with valid data existing so far.\nData is currently missing for {num_stocks_wo_data} stocks.\n\nPopulate DB by running 'populate_SQL.py' before re-running the current program.\n"
            )
            sys.exit(0)

    if batch_size is None:
        return no_url

    else:
        return no_url[:batch_size]


def build_payload(query, start=1, num=10):
    """Used for sending requests to the custom Google search engine.

    Args:
        query (str): Search query
        start (int): Index of the first result to return
        num (int): Number of results to return

    Returns:
        payload (dict): Dictionary containing the API request parameters
    """

    payload = {
        "key": GOOGLE_API_KEY,
        "q": query,
        "cx": SEARCH_ENGINE_ID,
        "start": start,
        "num": num,
    }

    return payload


def make_request(payload):
    """Makes the request to the google API to get the URL of a marketcap.com webpage containing info for a specific stock.

    Args:
        payload (dict): Dictionary containing the API request parameters

    Returns:
        dict: JSON response from the API
        None if the API returns a 429 code

    Raises: Exception if status code is not 200 or 429
    """

    #make request to google search API
    response = requests.get(
        "https://www.googleapis.com/customsearch/v1", params=payload
    )

    #check response from API, if you obtain json data - return it
    if response.status_code != 200:
        if response.status_code == 429:
            return None
        raise Exception(
            f"Request to google search API failed, status code: {response.status_code}"
        )
    return response.json()


def base_url_from_google(symbols):
    """
    Calls the Google API to get the base URL of a marketcap.com webpage containing info for a specific stock.
    Note that the base URL itself will actually return a 404; it requires a suitable suffix to get a certain type of info.

    Returns:
        url_not_found (list): Symbols of stocks for which a base URL could not be located
    """

    #retreived links must include this address:
    domain = "https://companiesmarketcap.com" 
    #retreived links must end with one of the following:
    link_suffixes = ["/marketcap/", "/shares-outstanding/", "/total-assets/", "/ps-ratio/", "/pe-ratio/", "/cash-on-hand/", "/earnings/", "/total-liabilities/"]
    #remove suffix of company names to facilitate search:
    company_suffixes = [' inc.', ' ltd.', ' incorporated', ' limited']
    
    pattern = r"(https://companiesmarketcap.com/[^/]+/).*"  # pattern for retrieving base_url using re

    url_added = [] #list of stocks for which base URL was added in this run
    url_not_found = [] #list of stocks for which base URL could not be found
    api_calls = 0 #initialize counter for API calls

    for symbol in symbols:

        #initiate an instance of Company, and input company name to payload for API request
        company = Company(symbol) 
        company_name = company.name
        matching_suffix = next((suffix for suffix in company_suffixes if company_name.lower().endswith(suffix)), None)

        # Remove the suffix from the company name
        if matching_suffix:
            company_name = company_name[:company_name.lower().find(matching_suffix)]
            company_name = company_name.strip(',')
            
        payload = build_payload(f"{company_name}") 
        
        #make request to API
        data = make_request(payload) 
        api_calls += 1
        print(f"{api_calls} calls to Google API made so far....", end="\r")
        time.sleep(play_nice)

        if data is None:  # 429 code returned - API limit reached
            print("You may have exceeded your daily query limit from Google.")
            print(
                f"Base URLs retrieved for the following {len(url_added)} stocks :\n",
                url_added,
            )
            conn.commit()
            return url_not_found

        if "items" not in data.keys():
            url_not_found.append(symbol)
            continue  # no relevant search results, move to next stock

        # look for the relevant link in the API results
        for item in data["items"]:  
            if domain in item["link"] and any(item["link"].endswith(suffix) for suffix in link_suffixes):
                link = item["link"]

                # use re.sub() to perform the substitution:
                base_url = re.sub(pattern, r"\1", link)

                #get all stock tickers per company
                # a single company may have stocks with different symbols (e.g. Class A and Class C stocks)
                cur.execute(
                    f"SELECT Symbol FROM Stocks WHERE Security LIKE '%{company.name}%'"
                )
                company_symbols = [cmp_symb[0] for cmp_symb in cur.fetchall()]
                for cmp_symb in company_symbols:
                    cur.execute(
                        "UPDATE Stocks SET Base_URL_MC = ? WHERE Symbol = ?",
                        (base_url, cmp_symb),
                    )
                    url_added.append(cmp_symb)
                break

        else:  # for loop ended w/o breaking
            url_not_found.append(symbol)

    conn.commit()
    print(
        f"Base URLs retrieved for the following {len(url_added)} stocks :\n", url_added
    )

    return url_not_found


def manual_url_input(url_not_found):
    """Asks the user to input base URLs that could not be found automatically by the program.

    Args:
        url_not_found (list): Symbols (str) of stocks for which a base URL was not found by the program.

    Returns:
        None
    """

    for symbol in url_not_found:
        base_url = input(
            f"Insert URL for stock symbol {symbol}, or press ENTER to keep NULL.\t"
        ).strip()
        if base_url != "":
            cur.execute(
                "UPDATE Stocks SET Base_URL_MC = ? WHERE Symbol = ?", (base_url, symbol)
            )
            conn.commit()


def report_sql_url_num(too_short):
    """Reports the number of base URLs that currently exist in the SQLite database.

    Returns:
        None
    """

    cur.execute("SELECT COUNT(*) FROM Stocks WHERE Base_URL_MC NOT NULL")
    print(f"A total of {cur.fetchone()[0]} URLs exist in 'Stocks' table.\n")

    # Check and report if base URL already exists for all relevant stocks
    global batch_size
    batch_size = None
    stocks_wo_url(too_short)


def main(key):
    """Program for retrieving the base URLs from companiesmarketcap.com for S&P 500 stocks, and storing these URLs in the SQLite database.
    
    Args:
        key (str or None): API key as defined by --key, or None if key was not inputted in command line.
         
    Returns:
        None

    Raises: 
        Exception: If Google API key is not available.
        sqlite3.OperationalError: If there's an error when working with the SQLite DB.
        KeyboardInterrupt: If user quits.
     """

    global GOOGLE_API_KEY
    GOOGLE_API_KEY = key

    if not GOOGLE_API_KEY:
        try:
            from google_api_key import GOOGLE_API_KEY

            if "your_key" in GOOGLE_API_KEY:
                raise Exception("Google API key must be provided.")
        except:
            raise Exception("Google API key must be provided.")

    try:
        # check that the SQLite DB exists in the expected location, and that it contains a 'Stocks' table
        check_db()

        # add Base_URL_MC and Tot_Trade_Days columns to the Stocks table
        add_columns()

        # identify stocks with too-short trading history, to be excluded from further consideration
        too_short = get_trading_duration()

        # identify the stocks for which a base URL has not yet been obtained
        symbols = stocks_wo_url(too_short)

        # use the Google search API to obtain base URLs, identify stocks for which base URL could not be found
        url_not_found = base_url_from_google(symbols)

        # if the search API failed to retrieve any URLs, ask the user for input
        if url_not_found:
            manual_url_input(url_not_found)

        # report how many base URLs already exist in the Stocks table
        report_sql_url_num(too_short)

    except sqlite3.OperationalError as e:
        if "database is locked" in str(e):
            print("Database is locked - please close the SQLite file and try again.")
        else:
            raise sqlite3.OperationalError(f"Error encountered: {e}")

    except KeyboardInterrupt:
        print("\n****Program terminated by user****\n")
        try:
            #try to report current state of SQLite DB before quitting
            report_sql_url_num(too_short)
        except:
            pass

    finally:
        # close connection to SQLite file
        conn.close()


"""Run program:"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Retrieve base URLs for stocks with sufficient trading data from companiesmarketcap.com; 
    populate SQLite DB with number of trading days and base URL per stock."""
    )

    #add argument for Google API key
    parser.add_argument(
        "--key", help="Google API key for accessing the custom search engine."
    )

    # parse command-line arguments
    args = parser.parse_args()

    main(args.key)