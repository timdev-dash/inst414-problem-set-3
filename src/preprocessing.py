'''
PART 1: PRE-PROCESSING
- Tailor the code scaffolding below to load and process the data
- Write the functions below
    - Further info and hints are provided in the docstrings
    - These should return values when called by the main.py
'''

# Standard library imports
from pathlib import Path

# Third party imports
import pandas as pd

# Setting constants
MAIN_FOLDER: Path = Path(__file__).absolute().parent
DATA_FOLDER: str = '../data/'

def load_data():
    '''
    Load data from CSV files
    
    Returns:
        model_pred_df (pd.DataFrame): DataFrame containing model predictions
        genres_df (pd.DataFrame): DataFrame containing genre information
    '''

    # Setting file names for the data sources    
    model_file_name: str = 'prediction_model_03.csv'
    genres_file_name: str = 'genres.csv'
    model_path_and_file: str = DATA_FOLDER + model_file_name
    genres_path_and_file: str = DATA_FOLDER + genres_file_name

    # Saving the files as dataframes to return
    model_pred_df: pd = pd.read_csv(MAIN_FOLDER / model_path_and_file, sep = ',', encoding = 'UTF-8')
    genres_df: pd = pd.read_csv(MAIN_FOLDER / genres_path_and_file, sep = ',', encoding = 'UTF-8')

    return model_pred_df, genres_df


def process_data(model_pred_df, genres_df):
    '''
    Process data to get genre lists and count dictionaries
    
    Returns:
        genre_list (list): List of unique genres
        genre_true_counts (dict): Dictionary of true genre counts
        genre_tp_counts (dict): Dictionary of true positive genre counts
        genre_fp_counts (dict): Dictionary of false positive genre counts
    '''

    # Your code here
