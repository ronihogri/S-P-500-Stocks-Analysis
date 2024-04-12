"""
Module containing the custom class "Company" for providing details regarding the company issuing a specific stock on-the-fly, 
without having to check the SQL database.

***Note: that this class refers to a specific table ('Stocks') within SQLite file. 
This is so that you don't have to specify it each time.
However, you can specify a different path and/or table name and pass them to the class if you'd like 
(just make sure that the table you specify is arranged in a similar way to 'Stocks'.*** 

Example:
    from ticker_to_company import Company

    # Create a Company instance:
    company = Company("GOOGL")
    company_differet_DB = Company("GOOGL", './different_sqlite_file.sqlite', 'different_table_name')

    # Retrieve company name and main market in which the stock is traded:
    name, market = company.name, company.market
    
    # Retrieve city and country of company's HQ:
    city, country = company.hq().city, company.hq().country    
    *Note the 'hq' must be used as a method.
    
    # Check if company HQ is located in the USA:
    in_usa = company.inusa        
  
"""

sqlite_file_name = "S&P 500.sqlite"  # default sqlite database
table_name = "Stocks"

# required libraries
import sqlite3
import requests
from bs4 import BeautifulSoup
from geopy.geocoders import Nominatim
import os
import time

geolocator = Nominatim(
    user_agent="hq_locator"
)  # geolocator object for getting geographic data about the company

current_dir = os.path.dirname(os.path.abspath(__file__))
sqlite_file_path = os.path.join(current_dir, sqlite_file_name)
play_nice = 0.5  # duration (s) to wait after requests from online resources, in case multiple requests are made in succession


class Company:
    """A class representing a company with methods to retrieve various attributes and information."""

    def __init__(self, symbol, sql_path=sqlite_file_path, table=table_name):
        """Initializes a Company object with the provided symbol and connects to the SQLite database.

        Args:
            symbol (str): The company's ticker.
            sql_path (str, optional): The path to the SQLite database file. Defaults to './S&P 500.sqlite'.

        Returns:
            None

        Raises:
            TypeError: If one of the input variables is not a str.
            ValueError: If the inputted symbol does not exist in the specified SQLite table.
        """

        for input_var in (
            symbol,
            sql_path,
            table,
        ):  # make sure inputs are of the correct type (str)
            if not isinstance(input_var, str):
                raise TypeError(
                    f"Input must be of type 'str', but input of type {type(input_var)} encountered."
                )

        self.symbol = symbol.upper()
        self.conn = sqlite3.connect(sql_path)
        self.cur = self.conn.cursor()
        self.cur.execute(
            f"SELECT Symbol FROM {table_name} WHERE Symbol = '{self.symbol}'"
        )
        symbol_exists = self.cur.fetchone()
        if symbol_exists is None:
            raise ValueError(
                f"Symbol '{self.symbol}' does not exist in the '{table_name}' table in {sql_path}."
            )

    def fetch_attribute(self, attribute_name, query):
        """Fetches an attribute of the company from the SQLite database using the provided SQL query.

        Args:
            attribute_name (str): The name of the attribute to fetch.
            query (str): The SQL query to execute.

        Returns:
            str: The fetched attribute value.
        """

        self.cur.execute(query)
        result = self.cur.fetchone()
        if result is not None:
            setattr(self, attribute_name, result[0])
        else:
            setattr(self, attribute_name, "NA")
        return getattr(self, attribute_name)

    @property
    def name(self):
        """Returns the name of the company issuing the stock.

        Returns:
            str: Company name.
        """

        query = f"SELECT Security FROM {table_name} WHERE Symbol = '{self.symbol}'"
        self._name = self.fetch_attribute("_name", query).split("(")[0].strip()
        return self._name

    @property
    def sector(self):
        """Returns the business sector in which this company operates.

        Returns:
            str: Company sector.
        """

        query = f"SELECT GICS_Sector FROM {table_name} WHERE Symbol = '{self.symbol}'"
        return self.fetch_attribute("_sector", query).strip()

    @property
    def subsector(self):
        """Returns the sub-sector in which this company operates.

        Returns:
            str: Company sub-sector.
        """

        query = (
            f"SELECT GICS_Sub_Industry FROM {table_name} WHERE Symbol = '{self.symbol}'"
        )
        return self.fetch_attribute("_subsector", query).strip()

    @property
    def CIK(self):
        """Returns the company's CIK (Central Index Key), as issued by the SEC.

        Returns:
            str: The company's CIK.
        """

        query = f"SELECT CIK FROM {table_name} WHERE Symbol = '{self.symbol}'"
        return self.fetch_attribute("_CIK", query).strip()

    @property
    def founded_str(self):
        """Returns the company's founding info as a str, including comments.

        Returns:
            str: The company's founding year + comments if there are any.
        """

        query = f"SELECT FOUNDED FROM Stocks WHERE Symbol = '{self.symbol}'"
        return self.fetch_attribute("_founded_str", query).strip()

    @property
    def founded(self):
        """Returns the company's founding year as an int.

        Returns:
            int: The company's founding year.
        """

        query = f"SELECT Founded FROM Stocks WHERE Symbol = '{self.symbol}'"
        return int(self.fetch_attribute("_founded", query).split("(")[0].strip())

    @property
    def base_url(self):
        """Returns the base URL for this stock from 'companiesmarketcap.com' without any slugs (by itself will return 404).

        Returns:
            str: base URL for this stock from 'companiesmarketcap.com'
        """

        query = f"SELECT Base_URL_MC FROM Stocks WHERE Symbol = '{self.symbol}'"
        return self.fetch_attribute("_base_url", query).strip()

    @property
    def days_traded(self):
        """Returns the number of trading days existing for this stock in the investigated historical period.

        Returns:
            int: Number of days traded within the historical period defined in 'populate_SQL.py'
        """

        query = f"SELECT Tot_Trade_Days FROM Stocks WHERE Symbol = '{self.symbol}'"
        return self.fetch_attribute("_days_traded", query)

    @property
    def first_entry_date(self):
        """Returns the date of the first data entry for this stock.

        Returns:
            str: Date (YYYY-MM-DD) of first data entry for this stock.
        """
        query = f"SELECT MIN(Date) FROM '{self.symbol}'"
        return self.fetch_attribute("_first_entry_date", query)

    def wiki_soup(self, url=None):
        """Retrieves information about the company from Wikipedia.

        This method constructs a Wikipedia URL based on the company's name,
        fetches the content of the webpage, and parses it using BeautifulSoup
        to extract relevant information.

        Returns:
            BeautifulSoup object: Parsed HTML content of the Wikipedia page,
            or 'NA' if the page cannot be accessed.
        """

        name = self.name.replace(" ", "_")
        if url is None:
            url = "https://en.wikipedia.org/wiki/" + name
        try:
            r = requests.get(url)
            time.sleep(play_nice)
            if r.status_code != 200:
                return "NA"
            self._soup = BeautifulSoup(r.content, "html.parser")

            # first, check if this is a disambiguation page:
            disambiguation_template = self._soup.find("div", {"id": "disambigbox"})
            if disambiguation_template is None:
                # if not, return soup as is
                return self._soup

            links = self._soup("a", href=True)
            for link in links:
                href = link["href"]
                title = link.get_text()
                if self.name.lower() in title.lower() and "company" in title.lower():
                    new_url = "https://en.wikipedia.org/" + href
                    return self.wiki_soup(url=new_url)
        except:
            return "NA"

    @property
    def market(self):  # subdef of wiki_soup
        """Returns the main market in which this stock is traded.

        This method extracts the main market information from the Wikipedia page
        obtained by calling the wiki_soup() method. It searches for the 'Traded as'
        label in the infobox vcard table on the page and returns the corresponding
        market name.

        Returns:
            str: The main market in which this stock is traded,
            or 'NA' if the information is not available or cannot be retrieved.
        """
        soup = self.wiki_soup()
        if (soup == "NA") or (soup is None):
            return "NA"
        results = soup("table", attrs={"class": "infobox vcard"})
        for result in results:
            hrefs = result(href=True)
            for i, href in enumerate(hrefs):
                if "Traded as" in hrefs[i - 1]:
                    self._market = href.text
                    return self._market.strip()

    def hq(self):
        """Retrieves the headquarters location of the company from the database.

        This method should be called before accessing other methods that depend on
        the headquarters location attribute.

        Returns:
            Company: The Company object with updated headquarters location attribute.

        Note:
            This method does not return the headquarters location directly!
            It is intended to be called as a prerequisite for other methods that depend on the HQ location.
        """

        query = f"SELECT Headquarters_Location FROM {table_name} WHERE Symbol = '{self.symbol}'"
        self._hq_location = self.fetch_attribute("_hq_location", query)
        return self

    """Subdefs of hq():"""

    @property
    def location(self):
        """Retrieves the company's HQ location as specified in the SQLite database.

        Returns:
            str: HQ location.
        """
        return self._hq_location.strip()

    @property
    def city(self):
        """Retrieves the city in which the company's HQ is located.

        Returns:
            str: HQ city.
        """
        return self._hq_location.split(",")[0].strip()

    @property
    def inusa(self):
        """Determines if the company's headquarters is located in the United States.

        The 'Stocks' table contains the HQ location as "city, state" for US-based companies (and "city, country" for companies with HQ outside of the USA).
        Therefore, the geolocator module is used to check if a given location refers to the USA.

        Returns:
            bool: True if the headquarters is in the United States, False otherwise.
        """

        location = geolocator.geocode(self._hq_location)
        time.sleep(play_nice)
        country = location.raw["display_name"].split(",")[-1].strip()
        return country == "United States"

    @property
    def state(self):
        """For US- based companies, returns the state where the company's headquarters is located.

        Returns:
            str: The state name if the headquarters is in the United States, otherwise 'NA'.
        *I am aware that many other countries have states too, this is just how the wikipedia table (and therefore the 'Stocks' table) is organized...
        """

        if self.inusa():
            return self._hq_location.split(",")[-1].strip()
        else:
            return "NA"

    @property
    def country(self):
        """Retrieves the country in which the company's HQ is located.

        Returns:
            str: HQ country.
        """

        if self.inusa():
            return "United States"
        else:
            return self._hq_location.split(",")[-1].strip()

    @property
    def coords(self):
        """Retrieves the geographic coordinates (latitude, longitude) of the company's HQ

        Returns:
            A tuple containing 2 float elements: (latitude, longitude), as retrieved by the geolocator.

        """

        location = geolocator.geocode(self._hq_location)
        time.sleep(play_nice)
        self._coords = (location.latitude, location.longitude)
        return self._coords
