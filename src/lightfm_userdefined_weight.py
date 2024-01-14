import pandas as pd
from scipy.sparse import csr_matrix
from lightfm import LightFM
import numpy as np

train_df = pd.read_csv("../data/train.csv")
test_df = pd.read_csv("../data/test.csv")
submit_df = pd.read_csv('../data/submit_example.csv')
full_df = pd.concat([train_df, test_df], ignore_index=True)

# create train set: 
# combined from train and test, because test data can also be used to predict. 
# add a weight column

def create_train_data(dataset):
    data = dataset[['user_id', 'product_id']]
    # Add a weight column that scales each interaction by how often the user buys it
    data = data.groupby(["user_id", "product_id"], as_index=False).size()
    
    data["weight"] = np.where(data["size"]>=5, 5, data["size"]) # cap it at 5
    data = data[["user_id", "product_id", "weight"]]
    return data
train = create_train_data(full_df)

# create test set, remove users not in training data

def create_test_data(test, train):
    data = test[["user_id", "product_id"]].drop_duplicates()
    data = data.merge(train["user_id"].drop_duplicates()) # remove users not in training data
    data = data.merge(train["product_id"].drop_duplicates()) # remove items not training data
    return data
test = create_test_data(test_df, train) 

# unique list of user IDs
train_users = train["user_id"].unique()

# unique list of prod IDs
train_items = train["product_id"].unique()

# ## Use `Dataset` method to help us build interaction matrix

from lightfm.data import Dataset
# Create user, item and feature mappings: (user id map, user feature map, item id map, item feature map)
dataset = Dataset() # helper function
dataset.fit(train_users, # creates mappings between userIDs and row indices for LightFM
                 train_items) 

# We want the user and item mappings (we'll use feature mappings later on)
user_mappings = dataset.mapping()[0]
item_mappings = dataset.mapping()[2]

# Create inverse mappings 
inv_user_mappings = {v:k for k, v in user_mappings.items()}
inv_item_mappings = {v:k for k, v in item_mappings.items()}

# Create an interactions matrix for each user, item and the weight
train_interactions, train_weights = dataset.build_interactions(train[['user_id', 'product_id', 'weight']].values)

# Have a look at the matrices
train_interactions.todense(), train_weights.todense() # weights and interactions are the same if we just use 1s

# User defined rating matrix
event_type_strength = {
    'view': 1.0,
    'cart': 2.0,
    'purchase': 4.0,
    'remove_from_cart': -1
}

import math
full_df['event_strength'] = full_df['event_type'].apply(lambda x: event_type_strength[x])
# due to "user cold-start" problem, we only consider thoses users who have at least 5 interactions
users_interactions_count_df = full_df.groupby(['user_id', 'product_id']).size().groupby('user_id').size()
users_with_enough_interactions_df = users_interactions_count_df[users_interactions_count_df >= 5]\
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

# Evaluation
interactions_train_df = interactions_full_df[interactions_full_df['user_id'].isin(train_df['user_id'].unique())]
interactions_test_df = interactions_full_df[interactions_full_df['user_id'].isin(test_df['user_id'].unique())]
# use top-N metric
# indexing by user_id to speed up the searches during evaluation
interactions_full_indexed_df = interactions_full_df.set_index('user_id')
interactions_train_indexed_df = interactions_train_df.set_index('user_id')
interactions_test_indexed_df = interactions_test_df.set_index('user_id')

row_users = interactions_full_df['user_id'].to_numpy()
row_mapped = np.array([user_mappings[user] for user in row_users])
col_items = interactions_full_df['product_id'].to_numpy()
col_mapped = np.array([item_mappings[item] for item in col_items])
values = interactions_full_df['event_strength'].to_numpy()
# Create all user and item matrix to get predictions for it
n_users, n_items = train_interactions.shape
user_defined_ratings = csr_matrix((values, (row_mapped, col_mapped)), shape=(n_users, n_items))

model_user_defined = LightFM(no_components=10,  # the dimensionality of the feature latent embeddings
                			learning_schedule='adagrad', # type of optimiser to use
                			loss='warp', # loss type
                			learning_rate=0.05,) # set the initial learning rate
             
model_user_defined.fit(user_defined_ratings, # our training data
               epochs = 20,
               verbose=True)

# Create all user and item matrix to get predictions for it
n_users, n_items = user_defined_ratings.shape

# Force lightFM to create predictions for all users and all items
new_scoring_user_ids = np.concatenate([np.full((n_items, ), i) for i in range(n_users)]) # repeat user ID for number of prods
new_scoring_item_ids = np.concatenate([np.arange(n_items) for i in range(n_users)]) # repeat entire range of item IDs x number of user
new_scores = model_user_defined.predict(user_ids = new_scoring_user_ids, 
                                     item_ids = new_scoring_item_ids)
new_scores = new_scores.reshape(-1, n_items) # get 1 row per user
new_recommendations = pd.DataFrame(new_scores)
print(new_recommendations.shape)

# Have a look at the predicted scores for the first 5 users and first 5 items
print(new_recommendations.iloc[:5,:5] )

# Find the maximum value in each row
new_max_scores = new_recommendations.max(axis=1)
new_max_column = new_recommendations.idxmax(axis=1)
new_user_ids = test['user_id'].unique()
new_preds = []
for uid in new_user_ids:
    user_mapped_id = user_mappings[uid]
    top_mapped_id = new_max_column[user_mapped_id]
    top_item_id = inv_item_mappings[top_mapped_id]
    new_preds.append(top_item_id)

submit_df['product_id'] = new_preds
submit_df.to_csv('../data/submit_lightfm_userdefined_weight.csv', index=False)