# Use lightfm according to tutorial
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

# Use `Dataset` method to help us build interaction matrix
from lightfm.data import Dataset
# Create user, item and feature mappings: (user id map, user feature map, item id map, item feature map)
dataset = Dataset() # helper function
dataset.fit(train_users, # creates mappings between userIDs and row indices for LightFM
                 train_items) 
len(dataset.mapping()) # we always get 4x mappings out

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

# create and fit our lightfm model

model = LightFM(no_components=10,  # the dimensionality of the feature latent embeddings
                			learning_schedule='adagrad', # type of optimiser to use
                			loss='warp', # loss type
                			learning_rate=0.05,) # set the initial learning rate
             
model.fit(train_interactions, # our training data
               epochs = 20,
               verbose=True)

# now we get predicted result (score matrix for every user and every item) using `predict()` method

# Create all user and item matrix to get predictions for it
n_users, n_items = train_interactions.shape

# Force lightFM to create predictions for all users and all items
scoring_user_ids = np.concatenate([np.full((n_items, ), i) for i in range(n_users)]) # repeat user ID for number of prods
scoring_item_ids = np.concatenate([np.arange(n_items) for i in range(n_users)]) # repeat entire range of item IDs x number of user
scores = model.predict(user_ids = scoring_user_ids, 
                                     item_ids = scoring_item_ids)
scores = scores.reshape(-1, n_items) # get 1 row per user
recommendations = pd.DataFrame(scores)
print(recommendations.shape)

# Have a look at the predicted scores for the first 5 users and first 5 items
print(recommendations.iloc[:5,:5] )

# now it's time to predict for test
# Find the maximum value in each row
max_scores = recommendations.max(axis=1)
max_column = recommendations.idxmax(axis=1)

user_ids = test['user_id'].unique()
preds = []
for uid in user_ids:
    user_mapped_id = user_mappings[uid]
    top_mapped_id = max_column[user_mapped_id]
    top_item_id = inv_item_mappings[top_mapped_id]
    preds.append(top_item_id)

submit_df['product_id'] = preds
submit_df.to_csv('lightfm_default_weight.csv', index=False)