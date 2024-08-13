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

    # Create list of genres
    genre_list: list = []

    # Begin loop to fill list
    row_num: int = 0

    while row_num < genres_df.shape[0]:

        # Adding the genre to the list
        genre_list.append(genres_df.iloc[row_num, 0])

        # Iterating row_num
        row_num += 1

    # Building the lists used to create dictionaries to return
    dict_keys: list = []
    dict_values: list = []

    # Popluating dictionaries with the list of genres
    for genre in genre_list:
        dict_keys.append(genre)
        dict_values.append(0)

    # Creating dictionaries of true genre counts, true positive genre counts, and 
    # fales positive genre counts.
    genre_true_counts: dict = dict(zip(dict_keys, dict_values))
    genre_tp_counts: dict = dict(zip(dict_keys, dict_values))
    genre_fp_counts: dict = dict(zip(dict_keys, dict_values))

    # Begin loop to record true, tp, and fp counts
    movie_index: int = 0

    while movie_index < model_pred_df.shape[0]:

        # Setting predicted and actual genres
        predicted_genre: str = model_pred_df.iloc[movie_index, 1]
        actual_genres: list = ((model_pred_df.iloc[movie_index, 2].strip("[]")).strip("'")).split("', '")

        # Counting true and false positives
        if model_pred_df.iloc[movie_index, 3] == 0:
            genre_fp_counts[predicted_genre] += 1
        elif model_pred_df.iloc[movie_index, 3] == 1:
            genre_tp_counts[predicted_genre] += 1
        
        # Counting true genres
        for movie_genre in actual_genres:
            if movie_genre == '':
                print('Blank genre found in movie ' + model_pred_df.iloc[movie_index, 0])
                continue
            genre_true_counts[movie_genre] += 1

        movie_index += 1

    return genre_list, genre_true_counts, genre_tp_counts, genre_fp_counts


if __name__ == '__main__':

    
    # Load data from CSV files
    model_pred_df, genres_df = load_data()
    
    # Process data to get genre counts and predictions
    genre_list, genre_true_counts, genre_tp_counts, genre_fp_counts = process_data(model_pred_df, genres_df)
