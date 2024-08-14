'''
PART 2: METRICS CALCULATION
- Tailor the code scaffolding below to calculate various metrics
- Write the functions below
    - Further info and hints are provided in the docstrings
    - These should return values when called by the main.py
'''

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd

def calculate_metrics(model_pred_df, genre_list, genre_true_counts, genre_tp_counts, genre_fp_counts):
    '''
    Calculate micro and macro metrics
    
    Args:
        model_pred_df (pd.DataFrame): DataFrame containing model predictions
        genre_list (list): List of unique genres
        genre_true_counts (dict): Dictionary of true genre counts
        genre_tp_counts (dict): Dictionary of true positive genre counts
        genre_fp_counts (dict): Dictionary of false positive genre counts
    
    Returns:
        tuple: Micro precision, recall, F1 score
        lists of macro precision, recall, and F1 scores
    
    Hint #1: 
    tp -> true positives
    fp -> false positives
    tn -> true negatives
    fn -> false negatives

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    Hint #2: Micro metrics are tuples, macro metrics are lists

    '''

    # Starting the construction of micro metrics
    tp_micro: int = sum(genre_tp_counts.values())
    fp_micro: int = sum(genre_fp_counts.values())
    fn_micro: int = 0
    
    ### Setting the micro true negative to 0. All the movies were identified
    ### as dramas. To be a true negative, it must not have been identified as
    ### a drama
    tn_micro: int = 0
    
    # Calculating fn_micro
    movie_index: int = 0

    while movie_index < model_pred_df.shape[0]:

        # Sorting through the false negatives
        if model_pred_df.iloc[movie_index, 3] == 1:
            break
        elif model_pred_df.iloc[movie_index, 1] not in model_pred_df.iloc[movie_index, 2]:
            break
        else:
            fn_micro += 1
    
    # Preparing to calculate micro metrics
    micro_precision: float = 0.00
    micro_recall: float = 0.00
    micro_f1: float = 0.00

    # Finalizing micro metrics for return
    if not tp_micro + fp_micro == 0:
        micro_precision += tp_micro / (tp_micro + fp_micro)
    if not tp_micro + fn_micro == 0:
        micro_recall += tp_micro / (tp_micro + fn_micro)
    if not micro_precision + micro_recall == 0:
        micro_f1 += (2 * micro_precision * micro_recall) / (micro_precision + micro_recall)

    # Starting the construction of macro metric lists
    macro_precision_list: list = []
    macro_recall_list: list = []
    macro_f1_list: list = []

    # Calculating macro metrics by genre
    for genre in genre_list:

        # Extracting true positive, false positive, and false negative amounts
        # per genre
        tp_macro: int = genre_tp_counts[genre]
        fp_macro: int = genre_fp_counts[genre]
        fn_macro: int = genre_true_counts[genre] - tp_macro

        # Preparint to calculate macro metrics for genre
        macro_precision: float = 0.00
        macro_recall: float = 0.00
        macro_f1: float = 0.00

        # Finalizing macro metrics for genre
        if not tp_macro + fp_macro == 0:
            macro_precision += tp_macro / (tp_macro + fp_macro)
        if not tp_macro + fn_macro == 0:
            macro_recall += tp_macro / (tp_macro + fn_macro)
        if not macro_precision + macro_recall == 0:
            macro_f1 += (2 * macro_precision * macro_recall) / (macro_precision + macro_recall)
        
        # Adding genre results to macro metric lists
        macro_precision_list.append(macro_precision)
        macro_recall_list.append(macro_recall)
        macro_f1_list. append(macro_f1)

        return micro_precision, micro_recall, micro_f1, macro_precision_list, macro_recall_list, macro_f1_list
    
def calculate_sklearn_metrics(model_pred_df, genre_list):
    '''
    Calculate metrics using sklearn's precision_recall_fscore_support.
    
    Args:
        model_pred_df (pd.DataFrame): DataFrame containing model predictions.
        genre_list (list): List of unique genres.
    
    Returns:
        tuple: Macro precision, recall, F1 score, and micro precision, recall, F1 score.
    
    Hint #1: You'll need these two lists
    pred_rows = []
    true_rows = []
    
    Hint #2: And a little later you'll need these two matrixes for sk-learn
    pred_matrix = pd.DataFrame(pred_rows)
    true_matrix = pd.DataFrame(true_rows)
    '''

    # Prepare data for sklearn processing
    pred_rows: list = []
    true_rows: list = []

    # Iterate through rows to update valuess
    movie_index: int = 0
    
    while movie_index < model_pred_df.shape[0]:

        # Setting predicted and actual genres
        predicted_genre: str = model_pred_df.iloc[movie_index, 1]
        actual_genres: list = ((model_pred_df.iloc[movie_index, 2].strip("[]")).strip("'")).split("', '")

        # Resetting status marker list to build details for movie_index
        marker_true: list = []
        marker_pred: list = []

        # Marking true and predictive genres
        for movie_genre in genre_list:
            if movie_genre == '':
                continue

            if movie_genre in actual_genres:
                marker_true.append(1)
            else:
                marker_true.append(0)

            if movie_genre in predicted_genre:
                marker_pred.append(1)
            else:
                marker_pred.append(0)
        
        # Adding markers for movie_index
        pred_rows.append(marker_pred)
        true_rows.append(marker_true)

        # Iterate the movie_index
        movie_index += 1

    # Assembling data for analysis
    pred_matrix = pd.DataFrame(pred_rows, columns = genre_list)
    true_matrix = pd.DataFrame(true_rows, columns = genre_list)

    # Running analysis for return
    micro_sk_precision, micro_sk_recall, micro_sk_f1, micro_sk_support = precision_recall_fscore_support(true_matrix, pred_matrix, average = 'micro')

    macro_sk_precision, macro_sk_recall, macro_sk_f1, macro_sk_support = precision_recall_fscore_support(true_matrix, pred_matrix, average = 'macro')

    return micro_sk_precision, micro_sk_recall, micro_sk_f1, macro_sk_precision, macro_sk_recall, macro_sk_f1