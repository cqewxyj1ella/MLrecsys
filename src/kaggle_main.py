import numpy as np
import scipy
import pandas as pd
import math
import time
import sklearn
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from sklearn.preprocessing import MinMaxScaler
from evaluate import ModelEvaluator
from kaggle_models import PopularityRecommender, CFRecommender
from tqdm import tqdm

def preprocess():
    train_df = pd.read_csv('../data/train.csv')
    test_df = pd.read_csv('../data/test.csv')
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    # Data munging
    # there are four types of event, we give then weights/strengths
    event_type_strength = {
        'view': 1.0,
        'cart': 2.0,
        'purchase': 4.0,
        'remove_from_cart': -1
    }
    full_df['event_strength'] = full_df['event_type'].apply(lambda x: event_type_strength[x])
    # due to "user cold-start" problem, we only consider thoses users who have at least 5 interactions
    users_interactions_count_df = full_df.groupby(['user_id', 'product_id']).size().groupby('user_id').size()
    users_with_enough_interactions_df = users_interactions_count_df[users_interactions_count_df >= 1]\
                                        .reset_index()[['user_id']]
    interactions_from_selected_users_df = full_df.merge(users_with_enough_interactions_df, 
                how = 'right',
                left_on = 'user_id',
                right_on = 'user_id')
    # a user can interact with an item for many times. we aggregate them together by applying weight
    def smooth_user_preference(x):
        return math.log(max(1, 1+x), 2)
        
    interactions_full_df = interactions_from_selected_users_df \
                        .groupby(['user_id', 'product_id'])['event_strength'].sum() \
                        .apply(smooth_user_preference).reset_index()
    return train_df, test_df, full_df, interactions_full_df


def popularity(train_df, test_df, full_df, interactions_full_df):
    # one common baseline: popularity model
    # score: 0.00537634400 
    interactions_train_df = interactions_full_df[interactions_full_df['user_id'].isin(train_df['user_id'].unique())]
    interactions_test_df = interactions_full_df[interactions_full_df['user_id'].isin(test_df['user_id'].unique())]
    # use top-N metric
    # indexing by user_id to speed up the searches during evaluation
    interactions_full_indexed_df = interactions_full_df.set_index('user_id')
    interactions_train_indexed_df = interactions_train_df.set_index('user_id')
    interactions_test_indexed_df = interactions_test_df.set_index('user_id')
    popularity_model = PopularityRecommender(interactions_full_indexed_df, \
                    interactions_test_indexed_df, interactions_train_indexed_df)
    user_ids = test_df['user_id'].unique()
    preds = []
    for uid in tqdm(user_ids):
        preds.append(popularity_model.predict(user_id=uid))
    submit_df = pd.read_csv('../data/submit_example.csv')
    submit_df['product_id'] = preds
    submit_df.to_csv('popularity_sampled.csv', index=False)
    
def svd(test_df, full_df, interactions_full_df):
    # creating a sparse pivot table with users in rows and items in columns
    # score: 0.01792114800
    users_items_pivot_matrix_df = interactions_full_df.pivot(index='user_id', 
                                                            columns='product_id', 
                                                            values='event_strength').fillna(0)
    users_items_pivot_matrix = users_items_pivot_matrix_df.values
    users_ids = list(users_items_pivot_matrix_df.index)
    users_items_pivot_sparse_matrix = csr_matrix(users_items_pivot_matrix)
    # the number of factors to factor the user-item matrix.
    NUMBER_OF_FACTORS_MF = 15
    # performs matrix factorization of the original user item matrix
    # U, sigma, Vt = svds(users_items_pivot_matrix, k = NUMBER_OF_FACTORS_MF)
    U, sigma, Vt = svds(users_items_pivot_sparse_matrix, k = NUMBER_OF_FACTORS_MF)
    sigma = np.diag(sigma)
    dot1 = np.dot(U, sigma)
    print(np.shape(dot1))
    all_user_predicted_ratings = np.dot(dot1, Vt) 
    # normalization
    all_user_predicted_ratings_norm = (all_user_predicted_ratings - all_user_predicted_ratings.min()) \
        / (all_user_predicted_ratings.max() - all_user_predicted_ratings.min())
    # converting the reconstructed matrix back to a Pandas dataframe
    cf_preds_df = pd.DataFrame(all_user_predicted_ratings_norm, columns = users_items_pivot_matrix_df.columns, index=users_ids).transpose()
    cf_recommender_model = CFRecommender(cf_preds_df, full_df)
    user_ids = test_df['user_id'].unique()
    preds = []
    for uid in tqdm(user_ids):
        preds.append(cf_recommender_model.predict(user_id=uid))
    submit_df = pd.read_csv('../data/submit_example.csv')
    submit_df['product_id'] = preds
    submit_df.to_csv('SVD.csv', index=False)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run different methods.')
    parser.add_argument('--method', default='SVD', required=True,
                        help='enter the name of the method you want to run.\n\
                            For example, --method SVD, or --method popularity')

    args = parser.parse_args()
    
    train_df, test_df, full_df, interactions_full_df = preprocess()
    if args.method == 'SVD':
        svd(test_df, full_df, interactions_full_df)
    elif args.method == 'popularity':
        popularity(train_df, test_df, full_df, interactions_full_df)