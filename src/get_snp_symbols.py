'''
Roni Hogri, March 2024

This script scrapes the wikipedia page containng a list of S&P 500 companies to get some basic info on these companies. 
This info is then exported to an SQLite file, which is then used by other programs in this package. 
'''

#import libraries:
import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3
import argparse
import re

#get vars from main program:
parser = argparse.ArgumentParser() 
parser.add_argument('--sqlite_file_path', help='Path to the SQLite file')
parser.add_argument('--wikipedia_url', help='URL of the Wikipedia page')
args = parser.parse_args()
sqlite_file_path = args.sqlite_file_path #path to sqlite file to be created by this program (passed from the main program)
url = args.wikipedia_url #URL of wikipedia page containing list of S&P 500 companies

#get content from wikipedia page and parse it:
content = requests.get(url).content
soup = BeautifulSoup(content, 'html.parser')

#get relevant table from scraping:
tables = soup('table')
stocks_table = [table for table in tables if 'Date added' in table.get_text()] 
assert len(stocks_table) == 1, "More than one table fits criterion - check!!"
stocks_table = stocks_table[0]

#extract table rows:
rows = stocks_table('tr')

#get column titles of table:
column_titles = []
title_cells = rows[0]('th')
for cell in title_cells:
    column_title = re.sub(r"[ -]", "_", cell.get_text().strip()) #replace spaces and hyphens of column titles w underscore to be more SQL-friendly
    column_titles.append(column_title)
    
#make pandas DataFrame with columns named after column titles:
df = pd.DataFrame(columns=column_titles)

#for each row (except for column titles), get values and add them to the relevant cells in df
for row in rows[1:] : #for each row
    values = [value.get_text().strip() for value in row('td')] #values per row 
    try:
        df.loc[len(df)] = values #if data fits, fill this row in the df
    except: pass
        
df.Symbol = df.Symbol.str.replace('.', '-') #for stock symbols, '.' is replaced by '-' to conform w API       
        
#export df to sqlite file:
with sqlite3.connect(sqlite_file_path) as conn :
    try:
        df.to_sql('Stocks', conn, index=False, if_exists='replace')
        print(f'\nSQL table with basic info on stocks included in the S&P 500 successfully created - table includes {len(df)} stocks.')
    except Exception as e:
        print(f'Error encountered : {e}')
