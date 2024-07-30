"""
Roni Hogri, July 2024

This program should be run after 'data_adjustments.py'. It attempts to predict stock price changes in the next OUTPUT_WINDOW_SIZE days
based on the behavior of predictors during the previous INPUT_WINDOW_SIZE days. For this, a TensorFlow Sequential() model containing 
a long short-term memory (LSTM) layer (hereafter: 'LSTM model') is used. 

Workflow:
1. Check and alter user-definable variables as needed. 
2. The program finds the next stock to analyze, based on a CSV file it creates ('stock_prediction_tracker.csv') in the project's
main folder. Stocks are bunched together based on industry subsectors (see 'GICS_Sub_Industry' column in the 'Stocks' table 
of the sqlite DBs). The rationale is that price changes of other stocks in the subsector can contain useful information for predicting 
the price of the stock of interest (hereafter: 'master stock').
3. To minimize the introduction of unhelpful predictors into the LSTM model, the correlation between each suspected predictor
(master stock trade volume, and prices of other subsector stocks) and the master stock price is calculated; only features which
are strongly correlated with the master stock price are then used as predictors. 
4. The dataset is scaled, transformed and split into 3 splits (training, validation, test). The test split data is not available to 
the LSTM model - it is meant to test the model's performance by producing a single sequence of predicted prices 
(for OUTPUT_WINDOW_SIZE days) based on the previous INPUT_WINDOW_SIZE days.
5. Hyperparameter tuning: This is the most time-consuming part of the program. Depending on the context, the program will either
prompt the user to choose the initial hyperparameters or use the hyperparameters that worked best for the previous stock as the 
starting point for tuning. The 'best' hyperparameters are considered those that are expected to result in the lowest validation 
root mean squared error (RMSE). Each tuning trial consists of NUM_EXECUTIONS executions (for averaging purposes), and each execution
consists of up to MAX_TUNER_EPOCHS - however, executions (and sometimes trials) are cut short if the validation RMSE has stabilized
and/or the results are unpromising. 
6. Once tuning is complete (due to stabilization of validation RMSE values or reaching MAX_TUNE_TRIALS trials), the LSTM model runs 
using the best hyperparameters.
7. The LSTM model is used to make predictions for all three splits. Predictions and actual values are compared, and the RMSE values for 
each split are calculated.
8. The tracker file is updated. If RUN_CONTINUOUSLY is set to True (and CONTINUOUS_LIMIT has not been reached, or has been set to None), 
the program continues to the next stock in the subsector, or the next alphabetically (and its associated subsector) if predictions 
have already been made for all stocks in the previous subsector.    

Additional notes:
* As the program runs, it reports the paths of different files it creates. These include pkl files for storing model-related 
information, CSV files reporting on different aspects of program performance, and figure files. 
* The user may choose to terminate the program (Ctrl-C), and continue running it later - model tuning and other time-consuming
processes will be continued from the point of previous termination. 
* This program is a demo, and is limited by the fact that for each stock only a single test sequence is predicted for a very specific
point in the stocks' history. 
"""


"""User-definable variables; modify as necessary:"""

master_stock = None #The stock to be analyzed in each run is intended to be set automatically by the program. 
#However, if you want to make predictions for a specific stock, replace None with stock symbol. 

RUN_CONTINUOUSLY = True #if set to False, this program will run for a single stock (master stock) and then stop.
#Set to True if you want to run the program continuously, until predictions have been made for all stocks in DB (or until limit reached, see below).
#Ignored if user defines a specific master_stock to analyze.

CONTINUOUS_LIMIT = 15 #if RUN_CONTINUOUSLY is True, sets the limit of stocks to be analyzed per run; otherwise ignored.
#Set to None to run all remaining stocks at once.

OVERWRITE_FIGURES = True #set to False if you want existing figures to not be overwritten (only relevant when working on the same stock more than once).
#Note that, if the model is fit again (not loaded from file), then all model-related figures will be overwritten anyway.

#model and tuner preferences:
INPUT_WINDOW_SIZE = 50 #num of previous trading days to consider
OUTPUT_WINDOW_SIZE = 5 #num of next trading days to predict
TRAIN_RATIO = 0.7 #percent of trials in dataset to be used for model training

#thresholds (window) for correlation of sector features with master stock price - determine which features will be used as predictors
#by the LSTM model:
CORRELATION_TRHESHOLD_LOW = 0.5 #lower edge of window, correlations below this threshold will not be considers as predictors 
CORRELATION_TRHESHOLD_HIGH = 0.9 #higher edge of window, correlations above this threshold will not be considered as predictors
#This determines which features are used as predictors by the LSTM model.
#Set to 1 if you only want to use the price of the master stock

INITIAL_TUNER_TRIALS = 30 #initial num of trials per tuning before deciding if tunning should continue
ADDITIONAL_TUNER_TRIALS = 20 #num of additional trials (after initial) before next decision whether to continue tuning
MAX_TUNE_TRIALS = 300 #if tuner has run for more trials than this value, stop tuning 
NUM_EXECUTIONS = 3 #number of executions per tuning trial - used for averaging trial result
MAX_TUNER_EPOCHS = 100 #maximum num of tuner epochs per tuning trial, if not cut short due to high/stabilized validation RMSE

MAX_TUNER_STD = 0.01 #if the std of the validation RMSE for the last ADDITIONAL_TUNER_TRIALS is lower than this value, stop tuning
PATIENCE = 7 #number of epochs without improvement of >= MIN_DELTA that will lead to early stopping of a tuning trial
MIN_DELTA = 0.01 #minimal improvement from previous best epoch required to reset patience
VAL_RMSE_CHECKPOINT = int(MAX_TUNER_EPOCHS * 0.75) #number of tuning epochs, after which the trial is terminated if the validation RMSE > MAX_VAL_RMSE
MAX_VAL_RMSE_RATIO = 0.2 #Used to calculate a threshold value for validation RMSE during tuning.
#If validation RMSE is higher than this thresholod after VAL_RMSE_CHECKPOINT epochs, or when progress is too slow, the tuning trial is terminated.
#The actual value that is considered to high is set as MAX_VAL_RMSE_RATIO * max value of the scaled training split of the target data (master stock price)

MODEL_EPOCHS = 100 #num of epochs in tuned model (for actual predictions)

#paths of relevant sqlite DBs (created by 'data_adjustments.py')
MAIN_DB = "S&P 500_split_adj.sqlite" #split-adjusted data (when required)
DIV_ADJ_DB = "S&P 500_split_div_adj.sqlite" #dividend- (and split-) adjusted data 

"""End of user-defined variables"""


# import required libraries for global use:

print('\nImporting libraries....\n') #this takes some time
import warnings
warnings.filterwarnings("ignore") #suppress warnings
import os
import sys
import re
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoLocator
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from ticker_to_company import Company
from datetime import datetime
import pickle 
from sklearn.metrics import root_mean_squared_error

#tensorflow/keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #partially suppress TensorFlow info and warning messages
#import libraries for modeling
from tensorflow.keras.callbacks import EarlyStopping, Callback, ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import *
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt



"""Globals"""

#dirs and paths:
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # dir of this script
PROJECT_DIR = os.path.join(CURRENT_DIR, 'LSTM_demo', f'{INPUT_WINDOW_SIZE}_{OUTPUT_WINDOW_SIZE}')
STOCK_TRACKER_PATH = os.path.join(PROJECT_DIR, 'stock_prediction_tracker.csv') #CSV file for storing info on stocks for which predictions have already been made   
STOCK_TRACKER_COLUMNS = ['Symbol', 'Company Name', 'Subsector', 'Last Training', 'Mean Training RMSE (scaled)', 
    'Mean Validation RMSE (scaled)', 'Test RMSE (scaled)'] #columns in stock tracker
TUNER_DIR = os.path.join(PROJECT_DIR, 'tuner')
MODEL_DIR = os.path.join(PROJECT_DIR, 'model')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
FIG_DIR = os.path.join(PROJECT_DIR, 'figures')
#paths to sqlite files:
MAIN_DB_FP = os.path.join(CURRENT_DIR, MAIN_DB)
DIV_ADJ_FP = os.path.join(CURRENT_DIR, DIV_ADJ_DB)
if master_stock is not None: #if user defines a specific stock, the program will only run for this stock
    RUN_CONTINUOUSLY = False



"""Functions"""


def get_stocks_subsector():
    """Scans the tracker csv file (if exists) and chooses the next dataset to work on. 
    Goes one subsector at a time. Assumes that you want to model each stock only once (on the existing history). 

    Globals:
        master_stock (str): Symbol of the stock for which the price will be predicted during this run. 
                            Initially None (unless altered by user), then assigned a value by this function.
        MAIN_DB_FP (str): File path to the main database.
        STOCK_TRACKER_PATH (str): File path to the stock tracker CSV file.
        STOCK_TRACKER_COLUMNS (list): Headers (str) of stock tracker CSV file. 

    Returns:
        already_done (list): List containing 2 lists [stock symbols, subsectors] of previously-modeled stocks.
        company (ticker_to_company.Company): Instance of the custom class Company.
        subsector (list): List of stock symbols (str) in the subsector.
        subsector_name (str): Name of the subsector.
        tracker_df (pd.DataFrame): df holding info from stock tracker file, or an empty df with headers if tracker file not yet exists.
    """

    global master_stock #if master_stock is None (default at the beginning of each run), its value is assigned in this function

    already_done = [] #default value unless set below

    if not master_stock: #master_stock value was not defined by user (default use)
        with sqlite3.connect(MAIN_DB_FP) as conn:
            cur = conn.cursor()
            if os.path.exists(STOCK_TRACKER_PATH): #there is already a file for tracking for which stocks predictions were already made
                #look at stock prediction tracker
                tracker_df = pd.read_csv(STOCK_TRACKER_PATH)
                already_done = [tracker_df['Symbol'].to_list(), tracker_df['Subsector'].to_list()]
                last_subsector = already_done[1][-1] #last subsector for which a prediction has been made

                #find the next stock in the subsector for which predictions have not already been made:
                cur.execute("SELECT Symbol FROM Stocks WHERE GICS_Sub_Industry = ? ORDER BY Symbol", (last_subsector, ))
                remaining_in_subsector =  [tup[0] for tup in cur.fetchall() if tup[0] not in already_done[0]]
                if remaining_in_subsector:
                    master_stock = remaining_in_subsector[0]
                else: #there are no more stocks to be predicted in this subsector, move on to stocks outside this subsector
                    cur.execute("SELECT Symbol FROM Stocks ORDER BY Symbol")
                    remaining = [tup[0] for tup in cur.fetchall() if tup[0] not in already_done[0]]
                    if not remaining: #all stocks in the DB have already been modeled
                        print("Predictions have already been made for all stocks in database.\nProgram successfully completed.\n")
                        sys.exit()
                    else:
                        cur.execute("SELECT Symbol FROM Stocks WHERE Symbol = ?", (remaining[0], ))
                        master_stock = cur.fetchone()[0]

            else: #prediction tracker does not yet exist (first stock handled by this program)
                cur.execute("SELECT Symbol FROM Stocks ORDER BY Symbol")
                master_stock = cur.fetchone()[0]
                tracker_df = pd.DataFrame(columns=STOCK_TRACKER_COLUMNS)

    company = Company(master_stock) #instance of Company class for master_stock
    subsector = [master_stock] + [tup[0] for tup in company.sector_all['subsector']['competitors']] #list symbols of stocks within subsector
    subsector_name = company.sector_all['subsector']['name'] 
            
    return tracker_df, already_done, company, subsector, subsector_name


def make_main_df(subsector):
    """Create pandas df to store all data used for training and predictions. 

    Args:
        subsector (list): List of stock symbols (str) in the subsector.

    Globals:
        master_stock (str): Symbol of the stock for which the price will be predicted. 
        DIV_ADJ_FP (str): Path of sqlite file holding dividend-adjusted data.
        MAIN_DB_FP (str): Path of sqlite file holding data for stocks that have not paid dividends during the historical period.

    Returns:
        df (pd.DataFrame): A df containing daily prices for all stocks in subsector, and also trading volume for master_stock.
    """

    dfs = {} #dict to hold temporary dfs for each stock in subsector until they are joined 
    df = pd.DataFrame() #joined df containing data for all stocks in subsector

    for stock in subsector: 
        try:
            with sqlite3.connect(DIV_ADJ_FP) as conn: #first try and retrieve dividend-adjusted data if available
                sql_df = pd.read_sql(f"SELECT Date, Close, Volume FROM {stock} ORDER BY Date", conn)
        except pd.io.sql.DatabaseError: #no dividends for this stock, use the main "clean" DB
            with sqlite3.connect(MAIN_DB_FP) as conn:
                sql_df = pd.read_sql(f"SELECT Date, Close, Volume FROM {stock} ORDER BY Date", conn)
        #for master stock, keep all selected columns; for other stocks, keep only date and closing price
        if stock == master_stock:
            dfs[stock] = sql_df
        else: 
            dfs[stock] = sql_df[['Date', 'Close']]

    #df indices are the master stock trading dates
    df['Date'] = dfs[master_stock].Date 
    df = df.set_index('Date')

    #join data from dfs dict to df; columns are first renamed to avoid duplicates
    for stock, subsector_df in dfs.items():    
        subsector_df = subsector_df.rename(columns={'Close': f'{stock}_price', 'Volume': f'{stock}_volume'})  
        df = df.join(subsector_df.set_index('Date'), how='left')  

    df.dropna(subset=df.columns[0], inplace=True) #remove rows for which the master stock's price is NaN

    return df


def split_info(df):
    """Reports how the dataset is going to be split and returns split-related values.

    Args:
        df (pd.DataFrame): A df containing daily prices for all stocks in subsector, and also trading volume for master_stock.
        
    Globals:
        TRAIN_RATIO (float): Ratio of trials from the entire dataset to be used for training. 
        OUTPUT_WINDOW_SIZE (int): Length of prediction sequence. 
        master_stock (str): Symbol of stock for which the price will be predicted.

    Returns:
        num_train_trials (int): Number of trials included in the training split.
        num_val_trials (int): Number of trails included in the validation split. 
    """

    num_train_trials = int(len(df) * TRAIN_RATIO) #number of trials in the train split
    num_val_trials = len(df)-num_train_trials-OUTPUT_WINDOW_SIZE #number of trials in the validation split

    #dates
    train_dates = df.index[:num_train_trials]
    val_dates = df.index[num_train_trials:-OUTPUT_WINDOW_SIZE]
    test_dates = df.index[-(OUTPUT_WINDOW_SIZE+INPUT_WINDOW_SIZE):]

    print(f"""
.... Preparing to make predictions for stock with symbol '{master_stock}' ('{Company(master_stock).name}')....


Total number of trading days during historical period ({df.index.min()} - {df.index.max()}) : {len(df)}.

Training split:
Actual values from the following dates are used as input: {train_dates[0]} - {train_dates[-(OUTPUT_WINDOW_SIZE+1)]}, including ({num_train_trials-OUTPUT_WINDOW_SIZE} trading days).
Predictions are given for the following days: {train_dates[INPUT_WINDOW_SIZE]} - {train_dates[-1]}, including ({num_train_trials-INPUT_WINDOW_SIZE} trading days).

Validation split:
Actual values from the following dates are used as input: {val_dates[0]} - {val_dates[-(OUTPUT_WINDOW_SIZE+1)]}, including ({num_val_trials-OUTPUT_WINDOW_SIZE} trading days).
Predictions are given for the following days: {val_dates[INPUT_WINDOW_SIZE]} - {val_dates[-1]}, including ({num_val_trials-INPUT_WINDOW_SIZE} trading days).

Test split:
Actual values from the following dates are used as input: {test_dates[0]} - {test_dates[-(OUTPUT_WINDOW_SIZE+1)]} ({INPUT_WINDOW_SIZE} trading days).
Predictions are given for the following days: {test_dates[-OUTPUT_WINDOW_SIZE]} - {test_dates[-1]} ({OUTPUT_WINDOW_SIZE} trading days).

""")     
    
    return num_train_trials, num_val_trials


def scale_data(df, num_train_trials):
    """Normalizes (scales) the data in each column based on the training split. 

    Args:
        df (pd.DataFrame): A df containing daily prices for all stocks in subsector, and also trading volume for master_stock.
        num_train_trials (int): Number of trials included in the training split.

    Returns:
        scaler_master_price (sklearn.preprocessing.RobustScaler): Instance of RobustScaler fit on the master stock column; will be 
            used later for inverse scaling of predictions. 
        scaled_df (pd.DataFrame): The modified df containing scaled prices/volume. 
        max_value_scaled_target_training (float): Maximal scaled value of target dataset during the training split.
    """

    #scaler instances for master stock price (will be used later for unscaling) and for the other features
    scaler_master_price = RobustScaler()
    scaler_others = RobustScaler()
    
    scaled_df = pd.DataFrame(columns=df.columns, index=df.index) #initialize df to hold scaled values

    #fit scalers to training split data
    scaler_master_price.fit(df.iloc[:num_train_trials, [0]]) 
    scaler_others.fit(df.iloc[:num_train_trials, 1:])

    scaled_df[scaled_df.columns[0]] = scaler_master_price.transform(df.iloc[:, [0]]) 
    scaled_df[scaled_df.columns[1:]] = scaler_others.transform(df.iloc[:, 1:]) 


    #get the highest scaled value from the training split of the master stock price column
    #this will be used to calculate what is considered an acceptable validation RMSE later on
    max_value_scaled_target_training = scaled_df[scaled_df.columns[0]].iloc[:num_train_trials].max()

    return scaler_master_price, scaled_df, max_value_scaled_target_training


def plot_scaled_data(subsector, scaled_df, company, subsector_name):
    """Plots the scaled prices (and the volume for the master stock) throughout the historical period.

    Args:
        subsector (list): List of stock symbols (str) in the subsector.
        scaled_df (pd.DataFrame): The modified df containing scaled prices/volume. 
        subsector_name (str): Name of the subsector.
        company (ticker_to_company.Company): Instance of the custom class Company, representing the company with the stock symbol master_stock.

    Globals:
        FIG_DIR (str): Path to directory where figures are saved. 
        master_stock (str): Symbol of stock for which the price will be predicted.
        OVERWRITE_FIGURES (bool): If set to True, will cause figures that already exist for this master stock to be overwritten. 

    Returns:
        None
    """

    fig_path = os.path.join(FIG_DIR, master_stock, f'{master_stock}_subsector.png') #path to file showing scaled prices of subsector stocks (+scaled volume of master stock)
    if os.path.exists(fig_path) and not OVERWRITE_FIGURES: return #if figure exists and should not be overwritten, skip this function

    fig, ax = plt.subplots(figsize=(25, 10))
    colors = ['b', 'g', 'c', 'm', 'y', 'orange', 'purple', 'lime', 'navy', 'teal'] #save red for master stock

    #primary y-axis will be used for volume, use secondary y-axis for prices
    ax_prices = ax.twinx()

    #plot volume of master stock (bottom layer)
    ax.bar(scaled_df.index, scaled_df[f'{master_stock}_volume'], label=f'{master_stock} Volume', color='black', width=0.94)

    #plot prices of other stocks
    for i, stock in enumerate(subsector[1:]):
        ax_prices.plot(scaled_df[f'{stock}_price'], label=f'{stock} Price', color=colors[i % len(colors)], linewidth=0.5, alpha=0.5)
    
    #price of master stock is top layer
    ax_prices.plot(scaled_df[f'{master_stock}_price'], label=f'{master_stock} Price', color='tab:red', linewidth=3, alpha=0.7)

    #set labels and title
    ax.set_xlabel('Date')
    ax_prices.set_ylabel('Price')
    ax.set_ylabel('Volume')
    ax.set_title(f"Normalized Data for '{subsector_name}' subsector - Focus on '{company.name}' (Symbol: {master_stock})")

    #legend
    lines, labels = ax_prices.get_legend_handles_labels()
    lines_volume, labels_volume = ax.get_legend_handles_labels()
    ax.legend(lines + lines_volume, labels + labels_volume, loc='upper left')

    #adjust X axis
    plt.xticks(rotation=45, ha='right')
    ax.xaxis.set_major_locator(AutoLocator())

    #save figure
    fig.savefig(fig_path, bbox_inches='tight')
    print(f'Figures showing scaled prices of all subsector stocks + scaled volume of master stock saved to: {fig_path}')


def find_correlated_features(scaled_df, num_train_trials):
    """Find features that are correlated with master stock price during the train split.

    Args:
        scaled_df (pd.DataFrame): The modified df containing scaled prices/volume. 
        num_train_trials (int): Number of trials included in the training split.
        
    Globals:
        RESULTS_DIR (str): Path to directory where the CSV file containing correlation info will be stored. 
        master_stock (str): Symbol of stock for which the price will be predicted.
        FIG_DIR (str): Path to directory where figures are saved. 
        OVERWRITE_FIGURES (bool): If set to True, will cause figures that already exist for this master stock to be overwritten. 
        CORRELATION_THRESHOLD_LOW (float): Features with absolute correlations to master stock price higher than this threshold 
            will be considered as predictors by the LSTM model.
        CORRELATION_THRESHOLD_HIGH (float): Features with absolute correlations to master stock price higher than this threshold 
            will be excluded from acting as predictors in the LSTM model (to minimize multicollinearity).

    Returns:
        lstm_features (list): List of features that will be used as predictors by the LSTM model.
    """

    results_path = os.path.join(RESULTS_DIR, master_stock, f'{master_stock}_feature_correlations.csv')

    fig_path = os.path.join(FIG_DIR, master_stock, f'{master_stock}_feature_correlations.png') #file path to plot showing correlations between all features
    if not os.path.exists(fig_path) or OVERWRITE_FIGURES: #should figure file be created?
        print("\n.... Plotting between-features correlations ....") #this takes some time
        fig = sns.pairplot(scaled_df[:num_train_trials]) #create a pair plot for all features considered
        fig.savefig(fig_path, bbox_inches='tight')
        print(f"\nFigure showing correlations of scaled subsector features saved to: {fig_path}\n")

    #look for correlations in training split of scaled df
    training_df = scaled_df[:num_train_trials]

    #avoid including columns that have a different len than the master stock price (column 0)
    different_len_columns = [column for column in training_df.columns[1:] if len(training_df[column].dropna()) != len(training_df)]        
    training_df.drop(columns=different_len_columns, inplace=True)

    if different_len_columns: #if there are columns discarded for being too short
        print(f'The following features have a different number of samples compared to the main feature ({master_stock}_price), and will not be considered as co-predictors:\n{different_len_columns}\n')

    correlations = training_df.corr()[training_df.columns[0]]  #correlation values 

    print(f"\nFeatures correlation to {training_df.columns[0]} during the training period ('<--' indicates use as predictor by the LSTM model):")

    #initialize list of features that will be fed to the LSTM model (first feature is master stock price)
    lstm_features = [training_df.columns[0]] 
    csv_text = f"Feature, Correlation to {training_df.columns[0]},Used in model? (Y/N)\n" #headers for CSV file

    #fill lstm_featuers list with names of additional features that will be used as predictors    
    for index, value in correlations.iloc[1:].items():  
        if abs(value) >= CORRELATION_TRHESHOLD_LOW and abs(value) <= CORRELATION_TRHESHOLD_HIGH:
            lstm_features.append(index)
            print (f'{index}:\t{value}  <--') 
            csv_text += f'{index}, {value},Y\n'
        else:
            print (f'{index}:\t{value}')
            csv_text += f'{index}, {value},N\n'  

    print(f"\nThe following feature(s) will be used as predictors for the LSTM model:\t{', '.join([repr(feature) for feature in lstm_features])}\n")

    with open(results_path, 'w') as file: #write correlation results to CSV file
        file.write(csv_text)    
    print(f"A CSV file containing all feature correlations and marking predictors used by the LSTM model was saved to: {results_path}\n\n")

    return lstm_features


def df_to_X_y(scaled_df, lstm_features, num_train_trials, num_val_trials):
    """Reshapes and outputs the data to np arrays holding the values of features (X) and targets (y). 

    Args:
        scaled_df (pd.DataFrame): A df containing scaled prices for all stocks in subsector, and also trading volume for master_stock.
        lstm_features (list): List of features that will be used as predictors by the LSTM model. 
        num_train_trials (int): Number of trials included in the training split.
        num_val_trials (int): Number of trails included in the validation split. 

    Globals:
        INPUT_WINDOW_SIZE (int): Number of values inputted to the LSTM model for each date.
        OUTPUT_WINDOW_SIZE (int): Number of predicted values outputted by the LSTM model for each date.

    Returns:
        X_train (np.ndarray): np array containing predictor data for the training split.
        y_train (np.ndarray): np array containing target data for the training split. 
        X_val (np.ndarray): np array containing predictor data for the validation split.
        y_val (np.ndarray): np array containing target data for the validation split. 
        X_test (np.ndarray): np array containing predictor data for the test split.
    """

    X_df = scaled_df.loc[:, lstm_features] #only features with high correlation to master stock price are used as predictors
    y_df = scaled_df.loc[:, scaled_df.columns[0]] #only the first column is the target

    #create temporary dfs to hold only the relevant target for each split's X and y (predictors and targets)
    X_train_df = X_df[:num_train_trials-OUTPUT_WINDOW_SIZE]
    y_train_df = y_df[INPUT_WINDOW_SIZE:num_train_trials]
    X_val_df = X_df[num_train_trials:num_train_trials+num_val_trials-OUTPUT_WINDOW_SIZE]
    y_val_df = y_df[num_train_trials+INPUT_WINDOW_SIZE:num_train_trials+num_val_trials]
    X_test_df = X_df[-INPUT_WINDOW_SIZE-OUTPUT_WINDOW_SIZE:-OUTPUT_WINDOW_SIZE]
    y_test_df = y_df[-OUTPUT_WINDOW_SIZE:]

    #create np arrays holding the relevant feature & target data for training and validation splits
    X_train = np.array([X_train_df[i:i+INPUT_WINDOW_SIZE].to_numpy() for i in range(len(X_train_df) - INPUT_WINDOW_SIZE)])
    y_train = np.array([y_train_df[i:i+OUTPUT_WINDOW_SIZE].to_numpy() for i in range(len(y_train_df) - OUTPUT_WINDOW_SIZE)])  
    X_val = np.array([X_val_df[i:i+INPUT_WINDOW_SIZE].to_numpy() for i in range(len(X_val_df) - INPUT_WINDOW_SIZE)])
    y_val = np.array([y_val_df[i:i+OUTPUT_WINDOW_SIZE].to_numpy() for i in range(len(y_val_df) - OUTPUT_WINDOW_SIZE)])
    
    #create np arrays holding feature and target data for the test split (single sequence for both input and output)
    X_test = np.array([X_test_df.to_numpy()])
    y_test = np.array([y_test_df.to_numpy()]) #used only locally for data shape reporting

    #report data shapes (sanity check)
    print(f"""
Data shapes:
X_train: {X_train.shape}; y_train: {y_train.shape}
X_val: {X_val.shape}; y_val: {y_val.shape}
X_test: {X_test.shape}; y_test: {y_test.shape}\n""")

    return X_train, y_train, X_val, y_val, X_test


def choose_initial_hps(already_done):
    """Selects the initial hyperparameters to be used for the tuning of the LSTM model, or prompts the user to select them.

    Args:
        already_done (list): List containing 2 lists [stock symbols, subsectors] of stocks for which a model was already trained.

    Globals:
        TUNER_DIR (str): Path of dir holding tuning info for this project. 
        master_stock (str): Symbol of stock for which the price will be predicted.
        RUN_CONTINUOUSLY (bool): If set to True, program models more than one stock per run (serially). 

    Returns:
        str: Empty str or path from which to retrieve the hyperparameters with which to initialize tuning. 
    """

    #in case previous run was interrupted after tuning, don't bother getting initial best_hps (they will be used automatically)
    tuner_path = os.path.join(TUNER_DIR, master_stock, 'tuner0.json')
    
    if os.path.exists(tuner_path):
        print(f"""\nTuning for {master_stock} was already started in a previous run, and will resume automatically.
To start fresh, terminate the current run and delete the contents of the following directory:
{os.path.join(TUNER_DIR, master_stock)}\n""")
        return ''
    
    if RUN_CONTINUOUSLY: #if program is run continuously, user is not prompted - existing best_hps will be fetched from the previous stock
        if already_done:
            return os.path.join(TUNER_DIR, already_done[0][-1], 'best_hps.pkl')
        else: #no existing best_hps (first stock analyzed), return empty str
            return ''
    
    #RUN_CONTINUOUSLY is False
    if already_done: #was there already a model tuned for similar data?
        while True: #get user input
            use_last_tuning = input("Initiate hyperparameter values from last training values? [Y/N]\t").strip()
            if use_last_tuning.upper() == 'Y':
                best_hps_path = os.path.join(TUNER_DIR, already_done[0][-1], 'best_hps.pkl')
                break
            elif use_last_tuning.upper() == 'N':
                best_hps_path = input(
                    r"Insert a path for the previously obtained best hyperparameters to use as a starting point for the current tuning, or press ENTER to start from scratch (random initial hyperparameters):    ").strip().strip('"')
                break
        
    else: #no info on previously tuned models for this project, prompt user
        best_hps_path = input(
            r"Choose a path for the previously obtained best hyperparameters to use as a starting point for the current tuning, or press ENTER to start from scratch (random initial hyperparameters):    ").strip().strip('"')

    if best_hps_path: #path provided by user
        print(f'\n...Using best hyperparamters from:\n{best_hps_path}\n')
    else: #no path provided by user
        print('\n...Tuning will be done with random initial hyperparameters...\n')

    return best_hps_path


def build_model(hp, X_train):
    """Called by tune_model(). Builds a Sequential model with hyperparameters set by tuning.

    Args:
        hp (kt.HyperParameters): Hyperparameter object used for tuning.

    Returns:
        model (tf.keras.Model): Compiled Keras model.
    """

    #build a sequential model with hp values within the specified ranges
    model = Sequential()
    model.add(InputLayer((INPUT_WINDOW_SIZE, X_train.shape[2]))) #input layer
    model.add(LSTM(units=hp.Int('lstm-units', min_value=32, max_value=192, step=32), #LSTM layer
                dropout=hp.Float('lstm-dropout', min_value=0.0, max_value=0.5, step=0.05),
                return_sequences=False,
                kernel_regularizer=l2(hp.Float('lstm-l2', min_value=1e-5, max_value=0.01, sampling='LOG'))))  
    model.add(Dropout(hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.05))) #dropout layer
    model.add(Dense(units=hp.Int('dense_units', min_value=8, max_value=128, step=8), #dense (relu) layer
                    activation='relu',
                    kernel_regularizer=l2(hp.Float('relu-l2', min_value=1e-5, max_value=0.01, sampling='LOG'))))  
    model.add(Dense(OUTPUT_WINDOW_SIZE, activation='linear')) #output layer

    lr = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='LOG') #learning rate
    model.compile(optimizer=Adam(learning_rate=lr), #compile model
                loss=MeanSquaredError(),
                metrics=[RootMeanSquaredError()])
    
    return model
    

class CustomEarlyStopping(EarlyStopping):
    """    Custom callback that inherits from Keras' EarlyStopping. 
    Used to stop a trial if the minimal validation RMSE is too high after a specified number of epochs (AKA the checkpoint). 
    Also stops the trial completely if improvements are not large enough. 

    Attributes:
        stop_trial (bool): Class attribute to track trial stopping state.
        stop_trial_text (str): Message to show when stopping execution/trial.
        execution_id (int): Class attribute to track the current execution number.
        monitor (str): Name of the metric to monitor during training.
        threshold (float): Threshold value for the minimal validation RMSE. If the minimal RMSE exceeds this value,
                           the trial is stopped.
        checkpoint (int): Number of epochs after which to check if the trial should be terminated.
        min_history_value (float): Minimum validation RMSE observed during the trial.
        old_min (float): Reference minimum value to compare against for significant improvements.
        patience_counter (int): Counter to track number of epochs since a significant improvement.

    Globals:
        PATIENCE (int): Number of epochs with no improvement after which the execution/trial will be stopped.
        MIN_DELTA (float): Minimum reduction in validation RMSE to qualify as an improvement.
        NUM_EXECUTIONS (int): Total number of executions per trial.
        VAL_RMSE_CHECKPOINT (int): Epoch checkpoint to check validation RMSE against the threshold.
        MAX_VAL_RMSE_RATIO (float): Ratio used to determine the threshold for maximum validation RMSE.
    """

    #default values
    stop_trial = False  #trial stopping state
    stop_trial_text = '' #message to show when stopping execution/trial
    execution_id = 0 #for counting executions within a trial

    def __init__(self, monitor='val_root_mean_squared_error', threshold=None, checkpoint=VAL_RMSE_CHECKPOINT):
        """Initializes the CustomEarlyStopping callback with the specified parameters.
        """
        super().__init__(monitor=monitor, patience=PATIENCE, restore_best_weights=True, min_delta=MIN_DELTA) #call parent
        if threshold is None:
            raise ValueError("Error in creating custom callback - 'max_val_rmse' is incorrectly defined. Check that 'MAX_VAL_RMSE_RATIO' is defined correctly.") 
        self.threshold = threshold
        self.checkpoint = checkpoint
        self.min_history_value = self.old_min = float('inf')  # Initialize with a large value

    def on_train_begin(self, logs=None):
        """Called at the beginning of each execution. Increments the execution ID and checks if the trial should be stopped.
        """
        super().on_train_begin(logs) #call parent
        self.patience_counter = 0 #for each execution, count number of epochs since (improvement > MIN_DELTA)
        CustomEarlyStopping.execution_id += 1 #count execution
        print(f"\nExecution #{CustomEarlyStopping.execution_id}\n")
        if CustomEarlyStopping.stop_trial: #trial should be stopped completely (rush remaining executions)
            self.model.stop_training = True #stop this execution regardless of its id
            if CustomEarlyStopping.execution_id == NUM_EXECUTIONS: #this is the last execution for this trial
                #reset values for next trial 
                CustomEarlyStopping.stop_trial = False                  
                CustomEarlyStopping.execution_id = 0
                return            

        #if this is the last execution in this trial, set id of next execution (next trial) to 0
        elif CustomEarlyStopping.execution_id == NUM_EXECUTIONS:
            CustomEarlyStopping.execution_id = 0
            return

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch. Updates the minimum validation RMSE observed and checks if the trial should be stopped.
        """
        super().on_epoch_end(epoch, logs) #call parent
        #get validation RMSE for this epoch; if it's smaller than the current minimum, then this is the new minimum
        current_val_rmse = logs.get(self.monitor) 
        if current_val_rmse is None: return
        if current_val_rmse < self.min_history_value: #is the new value a new minimum?
            if self.old_min - current_val_rmse > MIN_DELTA: #is the new minimum sufficiently smaller than the reference minimum?
                self.patience_counter = 0
                self.min_history_value = self.old_min = current_val_rmse                
            else: #new value is smaller than reference minimum but not sufficiently
                self.patience_counter += 1
                self.min_history_value = current_val_rmse
        else: #new value is not the minimal value
            self.patience_counter += 1

        if self.patience_counter == PATIENCE:
            if self.min_history_value > self.threshold: #stop entire trial and not just this execution
                CustomEarlyStopping.stop_trial = True  #update shared stop_trial state
                if CustomEarlyStopping.execution_id < NUM_EXECUTIONS:
                    CustomEarlyStopping.stop_trial_text = f"\n...Losing patience... Validation RMSE too high (> {np.round(self.threshold, 4)}) - rushing remaining executions and ending tuning trial.....\n"
                else:
                    CustomEarlyStopping.stop_trial_text = f"\n...Losing patience... Validation RMSE too high (> {np.round(self.threshold, 4)}) - ending tuning trial.....\n"
            else:
                CustomEarlyStopping.stop_trial_text = "\n...Losing patience... ending current execution.....\n"
            self.model.stop_training = True #stop this execution 
            return

        #at checkpoint, stop the trial if the minimum validation RMSE is larger than the threshold
        if epoch == self.checkpoint and self.min_history_value > self.threshold:
            CustomEarlyStopping.stop_trial = True  #update shared stop_trial state
            if CustomEarlyStopping.execution_id < NUM_EXECUTIONS:
                CustomEarlyStopping.stop_trial_text = f"\n...Minimal validation RMSE too high (> {np.round(self.threshold, 4)}) after {self.checkpoint} epochs, rushing remaining executions and ending tuning trial.....\n"
            else: 
                CustomEarlyStopping.stop_trial_text = f"\n...Minimal validation RMSE too high (> {np.round(self.threshold, 4)}) after {self.checkpoint} epochs, ending tuning trial.....\n"
            self.model.stop_training = True #stop this execution 
            return

    def on_train_end(self, logs=None):
        """Called at the end of an execution. Prints stop message if exists."""
        super().on_train_end(logs) #call parent
        if CustomEarlyStopping.stop_trial_text: #there's a message to print (execution/trial will be stopped)
            print(CustomEarlyStopping.stop_trial_text) #print stopping message and reset it
            CustomEarlyStopping.stop_trial_text = ''
            return


def tune_model(best_hps_path, X_train, y_train, X_val, y_val, max_value_scaled_target_training):
    """Tunes the model (seeks the best hyperparameters). 

    Args:
        best_hps_path (str): Path from which to retrieve the hyperparameters with which to initialize tuning. 
        X_train (np.ndarray): Training data (features). 
        y_train (np.ndarray): Training data (targets).
        X_val (np.ndarray): Validation data (features).
        y_val (np.ndarray): Validation data (targets).
        max_value_scaled_target_training (float): Maximal scaled value of target dataset during the training split.

    Globals:
        TUNER_DIR (str): Path of dir holding tuning info for this project. 
        INITIAL_TUNER_TRIALS (int): Number of tuner trials to run before checking tuner results for the first time. 
        master_stock (str): Symbol of stock for which the price will be predicted.
        MAX_VAL_RMSE_RATIO (float): Ratio of max_value_scaled_target_training to be used as threshold for stopping tuning trials 
            in which the validation RMSE is too high.
        NUM_EXECUTIONS (int): Number of executions per tuning trial.
        ADDITIONAL_TUNER_TRIALS (int): Number of tuner trials to add after each tuning round if objectives not yet met. 
        MAX_TUNER_STD (float): A threshold for std of best validation RMSE values, below which the tuner will stop running (tuning optimized).
        MAX_TUNE_TRIALS (int): Maximum number of tuner trials after which the tuner will be stopped even if tuning objectives were not met.
        MAX_TUNER_EPOCHS (int): Maximum number of epochs for each tuning trial.

    Returns:
        tuner (kt.tuners.RandomSearch): Tuner object after tuning, to be passed to the model.
        best_hps (kt.engine.hyperparameters.HyperParameters): Best hyperparameters found during tuning.
    """
    
    #path to csv file saving the best hyperparameters for easy viewing
    csv_best_hps_path = os.path.join(TUNER_DIR, master_stock, f'{master_stock}_best_hps.csv') 

    # Load the best hyperparameters from the previous tuning, if exists
    if best_hps_path:
        with open(best_hps_path, 'rb') as file:
            best_hps = pickle.load(file)
    else:
        best_hps = None

    # Initialize variables for tracking total trials and results
    total_tuner_trials = INITIAL_TUNER_TRIALS
    tuner_df = pd.DataFrame()
    tuner_df_path = os.path.join(TUNER_DIR, master_stock, f'{master_stock}_tuner_df.csv')
    
    if os.path.exists(tuner_df_path): #if file exists, build on existing results
        tuner_df = pd.read_csv(tuner_df_path, index_col=0)

    max_val_rmse = max_value_scaled_target_training * MAX_VAL_RMSE_RATIO #calculate validation RMSE threshold

    #create instance of custom early stopping class to be used as a callback by the tuner's search function
    custom_early_stopping = CustomEarlyStopping(threshold=max_val_rmse) 
    
    #loop for running the tuner for N trials per iteration
    while True:
        #create a RandomSearch tuner with the objective of minimizing the validation RMSE
        tuner = kt.RandomSearch(
            lambda hp: build_model(hp, X_train),  # Pass build_model as a lambda function with X_train
            objective='val_root_mean_squared_error',
            max_trials=total_tuner_trials,
            executions_per_trial=NUM_EXECUTIONS,
            hyperparameters=best_hps,
            directory=TUNER_DIR,
            project_name=master_stock
        )

        #stop tuning if the validation RMSE values of best trials are similar, or if the tuner has run too many times
        if (
            len(tuner_df) > 0 
            and tuner_df.iloc[:ADDITIONAL_TUNER_TRIALS, 1].values.std() < MAX_TUNER_STD 
            ) or (
                len(tuner_df) >= MAX_TUNE_TRIALS
            ):            
            break 
        
        #perform the hyperparameter search
        tuner.search(
            X_train,
            y_train, 
            validation_data=(X_val, y_val), 
            epochs=MAX_TUNER_EPOCHS, 
            callbacks=[custom_early_stopping]
            )
        
        #get all previous trials and their validation rmses
        trials = tuner.oracle.get_best_trials(num_trials=len(tuner.oracle.trials))
        val_rmses = [
            (trial.trial_id, trial.metrics.get_best_value("val_root_mean_squared_error"))
            if trial.metrics.exists("val_root_mean_squared_error") 
            else (trial.trial_id, np.nan)
            for trial in trials
        ]
                  
        # Create tuner_df with all existing validation RMSE values and save to CSV file 
        tuner_df = pd.DataFrame(val_rmses, columns=['Trial', 'Validation_RMSE'])
        tuner_df.to_csv(tuner_df_path, mode='w') #overwritten each time, so that the full (sorted) df could be retrieved as necessary

        #before moving on to the next iteration, add additional tuner trials
        total_tuner_trials += ADDITIONAL_TUNER_TRIALS
        if total_tuner_trials > MAX_TUNE_TRIALS:
            total_tuner_trials = MAX_TUNE_TRIALS #do not excede the maximum number of tuning trials allowed

    #get the best hyperparameters and save them
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    with open(os.path.join(TUNER_DIR, master_stock, 'best_hps.pkl'), 'wb') as file:
        pickle.dump(best_hps, file)

    # Print the optimal hyperparameters found
    print(f"""
The optimal number of units in the LSTM layer is {best_hps.get('lstm-units')},
the optimal dropout rate for the LSTM layer is {best_hps.get('lstm-dropout')},
the optimal L2 regularization value for the LSTM layer is {best_hps.get('lstm-l2')},
the optimal dropout rate for the LSTM layer's output is {best_hps.get('dropout')},
the optimal number of units in the dense layer is {best_hps.get('dense_units')},
the optimal L2 regularization value for the dense layer is {best_hps.get('relu-l2')}
and the optimal learning rate is {best_hps.get('learning_rate')}.

The best validation RMSE achieved is {tuner_df.loc[0, 'Validation_RMSE']}.
""")
    
    #write tuner results to csv file
    with open(csv_best_hps_path, 'w') as file: 
        file.write(f"""
LSTM units, {best_hps.get('lstm-units')}
Dropout within LSTM, {best_hps.get('lstm-dropout')}
LSTM L2, {best_hps.get('lstm-l2')}
Dropout LSTM output, {best_hps.get('dropout')}
Dense units, {best_hps.get('dense_units')}
Dense L2, {best_hps.get('relu-l2')}
Learning rate, {best_hps.get('learning_rate')}
Best val RMSE, {tuner_df.loc[0, 'Validation_RMSE']}"""
        )

    return tuner, best_hps


def fit_model(tuner, best_hps, X_train, y_train, X_val, y_val):
    """Loads the model if exists, if not then trains it using the hyperparameters set by tuning. 

    Args:
        tuner (kt.tuners.RandomSearch): Tuner object after tuning.
        best_hps (kt.engine.hyperparameters.HyperParameters): Best hyperparameters found during tuning.
        X_train (np.ndarray): Training data (features). 
        y_train (np.ndarray): Training data (targets).
        X_val (np.ndarray): Validation data (features).
        y_val (np.ndarray): Validation data (targets).

    Globals:
        OVERWRITE_FIGURES (bool): If set to True, will cause figures that already exist for this master stock to be overwritten.
            However, if this function finds that the model needs to be fit anew, then OVERWRITE_FIGURES will be set to False and 
            all model-related figures will be overwritten. 
        master_stock (str): Symbol of stock for which the price will be predicted.
        MODEL_EPOCHS (int): Number of epochs for model fitting.

    Returns:
        model (tf.keras.Model): Model for making predictions based on feature data. 
        history (tf.keras.callbacks.History): History of model training outcomes. 
    """

    global OVERWRITE_FIGURES #if a new model is created (not loaded from file), then subsequent figures should be overwritten in any case.

    #paths for file saving / loading
    model_path = os.path.join(MODEL_DIR, master_stock, f"{master_stock}.keras")
    history_path = os.path.join(MODEL_DIR, master_stock, f'{master_stock}_training_history.pkl')

    if os.path.exists(history_path): #was the model already trained in a previous run?
        with open(history_path, 'rb') as file_pi:
            history = pickle.load(file_pi)
        model = load_model(model_path) #loads the best version of the existing model (lowest validation RMSE)
        print(model.summary(), '\n') 

    else: #model needs to be trained   
        try:
            #set callbacks (for saving best model)
            cp = ModelCheckpoint(model_path, monitor='val_root_mean_squared_error', save_best_only=True, mode='min') 
            model = tuner.hypermodel.build(best_hps) #build model with hyperparameters determined by tuner
            print(model.summary(), '\nTraining model....\n') 

            # fit model and save the training history (to plot changes in training and validation RMSEs)
            history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=MODEL_EPOCHS, callbacks=[cp])
            with open(history_path, 'wb') as file_pi: #save history
                pickle.dump(history, file_pi)

            OVERWRITE_FIGURES = True #override OVERWRITE_FIGURES = False for the rest of the figures, which are model-related
        except OSError:
            raise Exception("Problem saving model. Try running as administrator and/or pausing cloud synchronization.")

    return model, history


def plot_fitting_rmses(history):
    """Plots the training and validtion RMSEs during model fitting and saves to figure file. 
    
    Args:
        history (tf.keras.callbacks.History): History of model training outcomes. 

    Globals:
        FIG_DIR (str): Path to directory where figures are saved. 
        master_stock (str): Symbol of stock for which the price will be predicted.
        OVERWRITE_FIGURES (bool): If set to True, will cause figures that already exist for this master stock to be overwritten. 
        
    Returns:
        None 
    """

    fig_path = os.path.join(FIG_DIR, master_stock, f'{master_stock}_RMSE.png') #path for file showing prediction errors (RMSEs) of all data splits
    if os.path.exists(fig_path) and not OVERWRITE_FIGURES: return #if figure exists and should not be overwritten, skip this function

    fig = plt.figure()
    plt.plot(history.history['root_mean_squared_error'], label='Training RMSE')
    plt.plot(history.history['val_root_mean_squared_error'], label='Validation RMSE')
    plt.legend()
    fig.savefig(fig_path, bbox_inches='tight')
    print(f'\nFigure showing training and validation RMSEs during model training saved to: {fig_path}\n')


def make_predictions(model, X_train, X_val, X_test):
    """Make predictions using the trained model. 

    Args:
        model (tf.keras.Model): Trained model. 
        X_train (np.ndarray): Training data (features).         
        X_val (np.ndarray): Validation data (features).        
        X_test (np.ndarray): Test data (features).        

    Returns:
        predictions (dict): Contains the predictions for the different splits. 
            Keys: Names of splits (training, validation, test)
            Values: Predictions (each a np.ndarray)
    """

    predictions = {
        'Training': model.predict(X_train), 
        'Validation': model.predict(X_val),
        'Test': model.predict(X_test)
        }

    return predictions


def create_results_df(predictions, scaled_df, num_train_trials, num_val_trials, scaler):
    """Function called by make_predictions(). Arranges predictions and actual data in dfs for use in plotting and documentation. 

    Args:
        predictions (dict): Contains the predictions for the different splits. 
            Keys: Names of splits (training, validation, test)
            Values: Predictions (each a np.ndarray)
        scaled_df (pd.DataFrame): A df containing scaled data for all stocks in subsector.
        num_train_trials (int): Number of trials included in the training split.
        num_val_trials (int): Number of trails included in the validation split.
        scaler (sklearn.preprocessing.RobustScaler): Instance of RobustScaler; here used for unscaling predictions. 

    Globals:
        RESULTS_DIR (str): Path to directory where prediction-related results are stored. 
        master_stock (str): Symbol of stock for which the price will be predicted.
        OUTPUT_WINDOW_SIZE (int): Number of predicted values outputted by the LSTM model for each date.
        INPUT_WINDOW_SIZE (int): Number of values inputted to the LSTM model for each date.

    Returns:
        results_unscaled (pd.DataFrame): A df containing unscaled actual and predicted values of target data. 
    """

    #path of results files
    results_path = os.path.join(RESULTS_DIR, master_stock, f'{master_stock}_results.csv') #scaled results
    unscaled_results_path = os.path.join(RESULTS_DIR, master_stock, f'{master_stock}_unscaled_results.csv') #unscaled results
    
    #use the first column of the scaled df (actual scaled values) as a starting point for results_df
    results_df = scaled_df.copy().drop(columns=scaled_df.columns[1:]).rename(columns={scaled_df.columns[0]: 'Actual'})  

    #add OUTPUT_WINDOW_SIZE columns to results_df for holding predicted values per day in predicted sequence
    for i in range(OUTPUT_WINDOW_SIZE): #each predicted sequence is OUTPUT_WINDOW_SIZE long
        results_df[f'Prediction d{i+1}'] = [np.nan] * len(results_df) #temporarily fill with NaNs

    #populate results_df with prediction data per split
    results_df.iloc[INPUT_WINDOW_SIZE:num_train_trials-OUTPUT_WINDOW_SIZE, 1:] = predictions['Training']
    results_df.iloc[num_train_trials+INPUT_WINDOW_SIZE:num_train_trials+num_val_trials-OUTPUT_WINDOW_SIZE, 1:] = predictions['Validation']
    results_df.iloc[-OUTPUT_WINDOW_SIZE, 1:] = predictions['Test']

    #in results_df, add a new column (dtype=object) indicating to which split each prediction belongs
    results_df['Split (Prediction)'] = (
        [np.nan] * INPUT_WINDOW_SIZE 
        + ['Training'] * (num_train_trials-INPUT_WINDOW_SIZE-OUTPUT_WINDOW_SIZE) + [np.nan] * (INPUT_WINDOW_SIZE+OUTPUT_WINDOW_SIZE) 
        + ['Validation'] * (num_val_trials-INPUT_WINDOW_SIZE-OUTPUT_WINDOW_SIZE) + [np.nan] * OUTPUT_WINDOW_SIZE 
        + ['Test'] + [np.nan] * (OUTPUT_WINDOW_SIZE-1))
    
    #save results_df to CSV file
    results_df.to_csv(results_path)
    print(f'\nCSV file containing scaled actual and predicted values saved to: {results_path}\n')

    #make a new version of results_df with unscaled data
    results_unscaled = results_df.copy()
    for col in results_unscaled.columns[:-1]: #unscale all columns except the last one (split indicator)
        results_unscaled[[col]] = np.round(scaler.inverse_transform(results_df[[col]]), 2)

    #save unscaled results to CSV
    results_unscaled.to_csv(unscaled_results_path)
    print(f'\nCSV file containing unscaled actual and predicted stock prices saved to: {unscaled_results_path}\n')
    
    return results_unscaled


def plot_predictions(results_unscaled): 
    """Plots predictions and actuals per data split. For this, generates new dfs (one per split), 
    where predictions are shifted based on their location in the sequence. 

    Args:
        results_unscaled (pd.DataFrame): A df containing unscaled actual and predicted values of target data. 

    Globals:
        FIG_DIR (str): Path to figure folder. 
        master_stock (str): Symbol of stock for which the price will be predicted.
        OUTPUT_WINDOW_SIZE (int): Number of predicted values outputted by the LSTM model for each date.
        OVERWRITE_FIGURES (bool): If set to True, will cause figures that already exist for this master stock to be overwritten. 
        
    Returns:
        df_training_shifted (pd.DataFrame): A df holding, for each date, the actual master stock price,
            plus predictions made for this day, for the training split.
        df_val_shifted (pd.DataFrame): A df holding, for each date, the actual master stock price,
            plus predictions made for this day, for the validation split.
        df_test_shifted (pd.DataFrame): A df holding, for each date, the actual master stock price,
            plus predictions made for this day, for the test split.
    """
    
    #paths to save figures
    fig_path_train = os.path.join(FIG_DIR, master_stock, f'{master_stock}_predictions_training.png') #path for figure file showing actual vs predicted master stock prices, training split
    fig_path_val = os.path.join(FIG_DIR, master_stock, f'{master_stock}_predictions_validation.png') #path for figure file showing actual vs predicted master stock prices, validation split 
    fig_path_test = os.path.join(FIG_DIR, master_stock, f'{master_stock}_predictions_test.png')  #path for figure file showing actual vs predicted master stock prices, test split

    #define colors to ensure uniform plotting for all splits
    colors = plt.cm.tab10(np.linspace(0, 1, OUTPUT_WINDOW_SIZE+1))
    

    #Training split figure
        
    #pad df_training with subsequent dates for showing each prediction in a sequence wrt its relevant date
    df_training = results_unscaled.loc[results_unscaled['Split (Prediction)'] == 'Training']
    extra_dates = results_unscaled.loc[results_unscaled.index > df_training.index.max()].head(OUTPUT_WINDOW_SIZE-1)
    df_training_shifted = pd.concat([df_training, extra_dates]).drop(columns=['Split (Prediction)'])
    #shift predictions to match dates to which they refer
    for i in range(OUTPUT_WINDOW_SIZE):
        df_training_shifted[f'Prediction d{i+1}'] = df_training_shifted[f'Prediction d{i+1}'].shift(i)

    if not os.path.exists(fig_path_train) or OVERWRITE_FIGURES: #if figure does not exist, or should be overwritten, create new fig
        #create figure and plot the lines of actual and predicted values
        fig_train = plt.figure(figsize=(20, 10)) 
        for i, col in enumerate(df_training_shifted.columns):
            if i == 0: #actuals
                width, alpha = 3, 1
            else: #predictions
                width, alpha = 1.5, 0.7
            plt.plot(df_training_shifted.index, df_training_shifted[col], color=colors[i], label=col, linewidth=width, alpha=alpha)

        plt.title(f"Training Split: Actual vs Predicted Stock Prices - '{Company(master_stock).name}' (Symbol '{master_stock}')")
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.xticks(rotation=45, ha='right')
        ax = plt.gca()
        ax.xaxis.set_major_locator(AutoLocator())
        plt.legend()
        fig_train.savefig(fig_path_train, bbox_inches='tight')      
        print(f'\nFigure showing actual and predicted values for training split saved to: {fig_path_train}\n')

 
    #Validation split figure

    #pad df_val with subsequent dates for showing each prediction in a sequence wrt its relevant date
    df_val = results_unscaled.loc[results_unscaled['Split (Prediction)'] == 'Validation']
    extra_dates = results_unscaled.loc[results_unscaled.index > df_val.index.max()].head(OUTPUT_WINDOW_SIZE-1)
    df_val_shifted = pd.concat([df_val, extra_dates]).drop(columns=['Split (Prediction)'])
    #shift predictions to match dates to which they refer
    for i in range(OUTPUT_WINDOW_SIZE):
        df_val_shifted[f'Prediction d{i+1}'] = df_val_shifted[f'Prediction d{i+1}'].shift(i)

    if not os.path.exists(fig_path_val) or OVERWRITE_FIGURES: #if figure does not exist, or should be overwritten, create new fig
        #create figure and plot the lines of actual and predicted values
        fig_val = plt.figure(figsize=(20, 10))
        for i, col in enumerate(df_val_shifted.columns):
            if i == 0: #actuals
                width, alpha = 3, 1
            else: #predictions
                width, alpha = 1.5, 0.7
            plt.plot(df_val_shifted.index, df_val_shifted[col], color=colors[i], label=col, linewidth=width, alpha=alpha)

        plt.title(f"Validation Split: Actual vs Predicted Stock Prices - '{Company(master_stock).name}' (Symbol '{master_stock}')")
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.xticks(rotation=45, ha='right')
        ax = plt.gca()
        ax.xaxis.set_major_locator(AutoLocator())
        plt.legend()
        fig_val.savefig(fig_path_val, bbox_inches='tight')      
        print(f'\nFigure showing actual and predicted values for validation split saved to: {fig_path_val}\n')        

    
    #Test split figure - works differently (only one sequence); therefore, the styling is a bit different than the previous plots

    #pad df_test with subsequent dates (and 1 preceding date to act as bifurcation point)
    df_test = results_unscaled.loc[results_unscaled['Split (Prediction)'] == 'Test']
    extra_dates = results_unscaled.loc[results_unscaled.index > df_test.index.max()].head(OUTPUT_WINDOW_SIZE-1)
    earlier_date = results_unscaled.loc[results_unscaled.index < df_test.index.min()].tail(1)
    df_test_shifted = pd.concat([earlier_date, df_test, extra_dates]).drop(columns=['Split (Prediction)'])

    #shift predictions to their respective dates
    for i in range(OUTPUT_WINDOW_SIZE): 
        df_test_shifted[f'Prediction d{i+1}'] = df_test_shifted[f'Prediction d{i+1}'].shift(i)
    
    #get the last actual value before the prediction started + all predicted values
    predicted_sequence = [earlier_date.Actual.values[0]] + df_test.iloc[0, 1:-1].to_list() 
 
    if not os.path.exists(fig_path_test) or OVERWRITE_FIGURES: #if figure does not exist, or should be overwritten, create new fig

        #we want 2 y-axes for the test data - one showing price (USD), and one showing % change from last actual before predictions
        fig_test, ax1 = plt.subplots(figsize=(OUTPUT_WINDOW_SIZE*2, 7))

        #first, plot actual and predicted prices as lines (predictions in distinct color; use pre-prediction actual as starting point)    
        ax1.plot(df_test_shifted.index, df_test_shifted.iloc[:, 0], color=colors[0], label='Actual', linewidth=3) #actual
        ax1.plot(df_test_shifted.index, predicted_sequence, color='black', label='Prediction', linewidth=1.5, alpha=0.7) #predictions
        
        #then, plot diamonds for each prediction day, with colors matching those of prediction days in former plots
        for i in range(OUTPUT_WINDOW_SIZE):
            ax1.scatter(df_test_shifted.index[i+1], df_test_shifted.iloc[i+1, i+1], color=colors[i+1], marker='D', s=70)        
        
        ax1.set_title(f"Test Split: Actual vs Predicted Stock Prices - '{Company(master_stock).name}' (Symbol '{master_stock}')")
        ax1.set_ylabel('Price (USD)')
        ax1.set_xlabel('Date')
        ax1.tick_params(axis='x', rotation=45) 
        plt.setp(ax1.get_xticklabels(), ha='right')
        ax1.legend()
        
        #create a secondary y-axis for % change from last pre-prediction
        ax2 = ax1.twinx()
        ax2.set_ylabel("% Change from Pre-Prediction Actual")

        #adjust the secondary y-axis limits to match the price y-axis
        scaling_factor = earlier_date.Actual.values[0] / 100  #scale as % change
        ax2.set_ylim((ax1.get_ylim()[0] - earlier_date.Actual.values[0]) / scaling_factor, (ax1.get_ylim()[1] - earlier_date.Actual.values[0]) / scaling_factor)

        #adjust figure layout 
        plt.tight_layout()
        ax1.yaxis.labelpad = 15
        ax2.yaxis.labelpad = 15

        if OUTPUT_WINDOW_SIZE > 10:
            ax1.xaxis.set_major_locator(AutoLocator())
        
        fig_test.savefig(fig_path_test, bbox_inches='tight')
        print(f'\nFigure showing actual and predicted values for test split saved to: {fig_path_test}\n')

    return df_training_shifted, df_val_shifted, df_test_shifted


def report_prediction_errors(df_training_shifted, df_val_shifted, df_test_shifted, scaler):
    """Generates a CSV file containing RMSE values per prediction day for the training and validation splits, and a single
    RMSE value for the test split. 

    Args:
        df_training_shifted (pd.DataFrame): A df holding, for each date, the actual master stock price,
            plus predictions made for this day, for the training split.
        df_val_shifted (pd.DataFrame): A df holding, for each date, the actual master stock price,
            plus predictions made for this day, for the validation split.
        df_test_shifted (pd.DataFrame): A df holding, for each date, the actual master stock price,
            plus predictions made for this day, for the test split.

    Globals:
        RESULTS_DIR (str): Path to directory where prediction-related results are stored. 
        master_stock (str): Symbol of stock for which the price will be predicted.
        
    Returns:
        train_rmses (dict): Unscaled RMSE values for each day in the predicted sequence (training split).
            Keys: Day descriptor (str).
            Values: RMSE per prediction day in sequence, unscaled (float).
        scaled_train_rmses (dict): Scaled RMSE values for each day in the predicted sequence (training split).
            Keys: Day descriptor (str).
            Values: RMSE per prediction day in sequence, scaled (float).
        val_rmses (dict): Unscaled RMSE values for each day in the predicted sequence (validation split).
            Keys: Day descriptor (str).
            Values: RMSE per prediction day in sequence, unscaled (float).
        scaled_val_rmses (dict): Scaled RMSE values for each day in the predicted sequence (validation split).
            Keys: Day descriptor (str).
            Values: RMSE per prediction day in sequence, scaled (float).
        test_rmse (float): RMSE value for the test split predictions, unscaled.
        scaled_test_rmse (float): RMSE value for the test split predictions, scaled.
    """

    #path to CSV file for storing RMSE results for all splits
    rmse_path = os.path.join(RESULTS_DIR, master_stock, f'{master_stock}_RMSE_per_split.csv')

    #Training RMSEs
    #get RMSEs for unscaled and scaled predictions vs actuals
    train_rmses = {} #unscaled
    scaled_train_rmses = {} #scaled
    #calculate the RMSE for each prediction day and its appropriate actuals
    for col in df_training_shifted.columns[1:]: 
        temp_df = df_training_shifted.loc[df_training_shifted[col].notnull()]
        train_rmses[col] = np.round(root_mean_squared_error(temp_df[col], temp_df[temp_df.columns[0]]), 4)
        scaled_train_rmses[col] = np.round(
            root_mean_squared_error(
                scaler.transform(np.array(temp_df[[col]])), scaler.transform(np.array(temp_df[[temp_df.columns[0]]]))
                ), 4)
    
    #Training RMSEs
    #get RMSEs for unscaled and scaled predictions vs actuals    
    val_rmses = {} #unscaled
    scaled_val_rmses = {} #scaled
    #calculate the RMSE for each prediction day and its appropriate actuals
    for col in df_val_shifted.columns[1:]: 
        temp_df = df_val_shifted.loc[df_val_shifted[col].notnull()]
        val_rmses[col] = np.round(root_mean_squared_error(temp_df[col], temp_df[temp_df.columns[0]]), 4)
        scaled_val_rmses[col] = np.round(
            root_mean_squared_error(
                scaler.transform(np.array(temp_df[[col]])), scaler.transform(np.array(temp_df[[temp_df.columns[0]]]))
                ), 4)
        
    #Test RMSE
    #in this case we have one prediction per actual (no separation by prediction day within the predicted sequence); hences 1 RMSE value
    #go through df_test_shifted and collect the actual and predicted values to lists
    test_actuals = []
    test_predictions = []
    for i, row in df_test_shifted[1:].iterrows(): 
        test_actuals.append(row[0]) #first cell in row is the actual value for this date
        test_predictions.append(row[1:].loc[row.notnull()].values[0]) #find cell in row containing a predicted value for this date

    #calculate the RMSE for all predictions vs actual values (unscaled and scaled) in test split
    test_rmse = np.round(root_mean_squared_error(test_predictions, test_actuals), 4) #unscaled
    scaled_test_rmse = np.round( #scaled
        root_mean_squared_error(
            scaler.transform(np.array([test_predictions]).reshape(-1, 1)), scaler.transform(np.array([test_actuals]).reshape(-1, 1))
        ), 4)
    

    #store RMSE values in CSV format 
    #training split data
    csv_txt = 'Training Split\nday in predicted sequence,USD,Scaled Values\n'

    #loop through unscaled and scaled dicts to extract relevant values
    for (key, value), scaled_value in zip(train_rmses.items(), scaled_train_rmses.values()):
        digits = ''.join(re.findall(r'\d+', key)) #day number  
        csv_txt += f'{digits},{value},{scaled_value}\n' #add to CSV str

    #validation split data
    csv_txt += '\n\nValidation Split\nday in predicted sequence,USD,Scaled Values\n'

    #loop through unscaled and scaled dicts to extract relevant values
    for (key, value), scaled_value in zip(val_rmses.items(), scaled_val_rmses.values()):
        digits = ''.join(re.findall(r'\d+', key)) #day number    
        csv_txt += f'{digits},{value},{scaled_value}\n' #add to CSV str

    #test split data - here, all prediction days are bunched together (prediction day column skipped)
    csv_txt += '\n\nTest Split (single sequence)\n,USD,Scaled Values\n' 
    csv_txt += f',{test_rmse},{scaled_test_rmse}\n'

    #save RMSE values to CSV file
    with open(rmse_path, 'w') as file:
        file.write(csv_txt)    

    print(f"\nRMSE values for all dataset splits (scaled and unscaled) have been saved to: {rmse_path}\n")
    print(f"\nThe RMSE for the test split (scaled predicted vs actual values) is: {scaled_test_rmse}\n\n")

    return train_rmses, scaled_train_rmses, val_rmses, scaled_val_rmses, test_rmse, scaled_test_rmse


def plot_predictions_errors(train_rmses, val_rmses, test_rmse, scaled_test_rmse, predictions):
    """Plots RMSE values per prediction day for the training and validation splits, and a single RMSE value for the test split. 

    Args:
        train_rmses (dict): RMSE values for each day in the predicted sequence (training split).
            Keys: Day descriptor (str).
            Values: RMSE per prediction day in sequence (float).
        val_rmses (dict): RMSE values for each day in the predicted sequence (validation split).
            Keys: Day descriptor (str).
            Values: RMSE per prediction day in sequence (float).
        test_rmse (float): RMSE value for the test split predictions.
        scaled_test_rmse (float): Scaled RMSE value for the test split predictions.
        predictions (dict): Contains the predictions for the different splits. Here, used to get the number of predictions per split. 

    Globals:
        FIG_DIR (str): Path to figure folder. 
        master_stock (str): Symbol of stock for which the price will be predicted.
        OVERWRITE_FIGURES (bool): If set to True, will cause figures that already exist for this master stock to be overwritten. 
        
    Returns:
        None
    """

    fig_path = os.path.join(FIG_DIR, master_stock, f'{master_stock}_RMSE_per_split.png') #path of RMSE figure file
    if os.path.exists(fig_path) and not OVERWRITE_FIGURES: return #if figure exists and should not be overwritten, skip this function

    #X positions for the bar graphs
    x1 = np.arange(OUTPUT_WINDOW_SIZE)
    x2 = x1 + OUTPUT_WINDOW_SIZE + 1  # Shift positions for the second bar graph
    x3 = [x2[-1] + 2]  # Shift position for the third bar graph

    #define colors for each bar set (so that training and validation RMSE bars match, while test bar is unique)
    colors = plt.cm.tab10(np.linspace(0, 1, OUTPUT_WINDOW_SIZE+1))

    #create fig and plot the bar graphs showing RMSEs for unscaled data
    fig, ax1 = plt.subplots(figsize=(OUTPUT_WINDOW_SIZE*2, 5))
    for i, (key, value) in enumerate(train_rmses.items()):
        ax1.bar(x1[i], value, color=colors[i], label=key)
    for i, (key, value) in enumerate(val_rmses.items()):
        ax1.bar(x2[i], value, color=colors[i])
    ax1.bar(x3, test_rmse, color=colors[-1])
    ax1.legend()

    #add ticks and labels below each set of bars
    ax1.set_xticks([x1.mean(), x2.mean(), x3[0]])
    ax1.set_xticklabels(
        [f"Training (n={len(predictions['Training'])})", 
         f"Validation (n={len(predictions['Validation'])})", 
         f"Test (n={OUTPUT_WINDOW_SIZE})"]
         )

    ax1.set_title("RMSE of Stock Price Predictions vs Actual Data by Split")
    ax1.set_ylabel("Root Mean Square Error (USD)")

    #create a secondary y-axis for the scaled RMSE units
    ax2 = ax1.twinx()
    ax2.set_ylabel("Root Mean Square Error (Scaled Values)")

    #adjust the secondary y-axis limits to match the non-scaled data
    scaling_factor = np.max(test_rmse) / np.max(scaled_test_rmse)
    ax2.set_ylim(ax1.get_ylim()[0] / scaling_factor, ax1.get_ylim()[1] / scaling_factor)

    #adjust figure layout 
    plt.tight_layout()
    ax1.yaxis.labelpad = 15
    ax2.yaxis.labelpad = 15

    fig.savefig(fig_path, bbox_inches='tight') #save figure


def update_tracker(tracker_df, company, subsector_name, scaled_train_rmses, scaled_val_rmses, scaled_test_rmse):
    """Updates the tracker CSV file to document that predictions have been made for the master stock.  

    Args:
        tracker_df (pd.DataFrame): df holding info from stock tracker file.
        company (ticker_to_company.Company): Instance of the custom class Company, representing the company with the stock symbol master_stock.
        subsector_name (str): Name of the subsector.
        scaled_train_rmses (dict): Scaled RMSE values for each day in the predicted sequence (training split).
            Keys: Day descriptor (str).
            Values: RMSE per prediction day in sequence, scaled (float).
        scaled_val_rmses (dict): Scaled RMSE values for each day in the predicted sequence (validation split).
            Keys: Day descriptor (str).
            Values: RMSE per prediction day in sequence, scaled (float).
        scaled_test_rmse (float): RMSE value for the test split predictions, scaled.

    Globals:
        master_stock (str): Symbol of stock for which the predictions were made in the current run. 
        STOCK_TRACKER_PATH (str): File path to the stock tracker CSV file.
        STOCK_TRACKER_COLUMNS (list): Headers (str) of stock tracker CSV file. 

    Returns:
        None
    """

    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S') #timestamp for completing work on this stock (now)

    #add info re current master stock to stock tracker df
    tracker_df.loc[len(tracker_df)] = [master_stock, company.name, subsector_name, now, np.mean(list(scaled_train_rmses.values())),
                                       np.mean(list(scaled_val_rmses.values())), scaled_test_rmse]
    
    #update tracker CSV file
    tracker_df.to_csv(STOCK_TRACKER_PATH, index=False)

    print(f"""\n
***************************************************************************************************************
Program completed successfully for stock with symbol '{master_stock}' - stock tracker CSV file updated:
{STOCK_TRACKER_PATH}
***************************************************************************************************************\n\n\n""")


def main():
    """Program for predicting stock prices based on past performance of this stock and the subsector to which it belongs."""

    try:
        #get the next stock to make predictions for ("master stock")
        tracker_df, already_done, company, subsector, subsector_name = get_stocks_subsector()

        #make subdirs containing files for this master stock:
        for main_dir in (TUNER_DIR, MODEL_DIR, RESULTS_DIR, FIG_DIR):
            os.makedirs(os.path.join(main_dir, master_stock), exist_ok=True)

        #make df to hold info for subsector (including master stock)
        df = make_main_df(subsector)

        #get number of samples and relevant dates for training, validation, and test splits
        num_train_trials, num_val_trials = split_info(df)

        #scale data
        scaler, scaled_df, max_value_scaled_target_training = scale_data(df, num_train_trials)

        #plot scaled prices for all stocks + trading volume for master stock
        plot_scaled_data(subsector, scaled_df, company, subsector_name)

        #find features highly-correlated to master stock price, to be included as predictors in the LSTM
        lstm_features = find_correlated_features(scaled_df, num_train_trials)

        #split the data and transform it to np arrays that can be fed to the model
        X_train, y_train, X_val, y_val, X_test = df_to_X_y(scaled_df, lstm_features, num_train_trials, num_val_trials)

        #prompt user to provide path of best hyperparameters from previous training, or start tuning from scratch. 
        best_hps_path = choose_initial_hps(already_done)

        #tune model - find best hyperparameters for master stock, based on validation RMSE
        tuner, best_hps = tune_model(best_hps_path, X_train, y_train, X_val, y_val, max_value_scaled_target_training)        

        #train model using the best hyperparameters found during model tuning
        model, history = fit_model(tuner, best_hps, X_train, y_train, X_val, y_val)

        #plot the progression of training and validation RMSES during model training
        plot_fitting_rmses(history)

        #make predictions for all data splits, store them as dfs, and plot actuals and predictions in figure files (one per split) 
        predictions = make_predictions(model, X_train, X_val, X_test)

        #create tables containig actual prices vs predictions (both scaled and unscaled), and save to CSV files
        results_unscaled = create_results_df(predictions, scaled_df, num_train_trials, num_val_trials, scaler)

        #plot predicted vs actual master stock prices for all splits
        df_training_shifted, df_val_shifted, df_test_shifted = plot_predictions(results_unscaled)

        #calculate prediction errors (RMSE) for each split and save to single CSV file
        train_rmses, scaled_train_rmses, val_rmses, scaled_val_rmses, test_rmse, scaled_test_rmse = report_prediction_errors(df_training_shifted, df_val_shifted, df_test_shifted, scaler)

        #plot prediction errors (RMSE) for each split and save to single figure file
        plot_predictions_errors(train_rmses, val_rmses, test_rmse, scaled_test_rmse, predictions)

        #document that this program was successfully completed for the current master stock (next run will run on next stock)
        update_tracker(tracker_df, company, subsector_name, scaled_train_rmses, scaled_val_rmses, scaled_test_rmse)

    except KeyboardInterrupt:
        raise KeyboardInterrupt("\n****Program terminated by user****\n")

    except Exception as e:
        raise Exception(f"{type(e).__name__} encountered:\n{e}")
    

"""Run program:"""
if __name__ == "__main__":
    """If RUN_CONTINUOUSLY is True, runs main() CONTINUOUS_LIMIT times, or until all stocks have been processed if CONTINUOUS_LIMIT
    is set to None. For each iteration, master_stock is reset to None so that the next master stock is set by get_stocks_subsector().
    """

    counter = 0
    master_list = []
   
    while True:
        main()
        if not RUN_CONTINUOUSLY:             
            break
        #add symbol to list of stocks already processed, and reset master_stock's value to None
        master_list.append(master_stock)
        master_stock = None
        counter += 1
        if counter == CONTINUOUS_LIMIT:
            print(f"Program completed successfully for stocks with the following symbols:")
            [print(f"'{stock}'") for stock in master_list]
            break    