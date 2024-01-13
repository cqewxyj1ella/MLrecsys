import random
import pandas as pd

class PopularityRecommender:
    
    MODEL_NAME = 'Popularity'
    EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS = 100
    
    def __init__(self, interactions_full_indexed_df, interactions_test_indexed_df, interactions_train_indexed_df):
        self.interactions_full_indexed_df = interactions_full_indexed_df
        self.interactions_test_indexed_df = interactions_test_indexed_df
        self.interactions_train_indexed_df = interactions_train_indexed_df
        self.popularity_df = interactions_full_indexed_df.groupby('product_id')['event_strength'].sum().sort_values(ascending=False).reset_index()
        
    def get_model_name(self):
        return self.MODEL_NAME
    
    def get_items_interacted(self, user_id, interactions_df):
        # get the user's data and merge in the item information.
        if user_id not in interactions_df.index.tolist():
            return set()
        interacted_items = interactions_df.loc[user_id]['product_id']
        return set(interacted_items if type(interacted_items) == pd.Series else [interacted_items])

    def get_not_interacted_items_sample(self, user_id, sample_size, seed=42):
        interacted_items = self.get_items_interacted(user_id, self.interactions_full_indexed_df)
        all_items = set(self.interactions_full_indexed_df['product_id'].unique())
        non_interacted_items = all_items - interacted_items
        random.seed(seed)
        non_interacted_items_sample = random.sample(non_interacted_items, sample_size)
        return set(non_interacted_items_sample)
        
    def recommend_items(self, user_id, items_to_ignore=[], topn=10):
        # recommend the more popular items that the user hasn't seen yet.
        recommendations_df = self.popularity_df[~self.popularity_df['product_id'].isin(items_to_ignore)] \
                               .sort_values('event_strength', ascending = False) \
                               .head(topn)
        return recommendations_df
    
    def predict(self, user_id):
        # only predict one most likely item
        # getting the items in test set
        if user_id not in self.interactions_test_indexed_df.index.tolist():
            person_interacted_items_testset = set()
        else:
            interacted_values_testset = self.interactions_test_indexed_df.loc[user_id]
            if type(interacted_values_testset['product_id']) == pd.Series:
                person_interacted_items_testset = set(interacted_values_testset['product_id'])
            else:
                person_interacted_items_testset = set([int(interacted_values_testset['product_id'])])
        # getting a ranked recommendation list from a model for a given user
        # recommendations_df = self.recommend_items(user_id, 
        #                                        items_to_ignore=self.get_items_interacted(user_id, 
        #                                                                                  self.interactions_train_indexed_df), 
        #                                        topn=10000000000)
        recommendations_df = self.recommend_items(user_id, topn=10000000000)
        non_interacted_items_sample = self.get_not_interacted_items_sample(user_id, 
                                                                          sample_size=self.EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS, 
                                                                          seed=user_id%(2**32))
        # combining the interacted items with the 100 random items
        items_to_filter_recs = non_interacted_items_sample.union(person_interacted_items_testset)
        # filtering only recommendations that are either the interacted item or from a random sample of 100 non-interacted items
        valid_recs_df = recommendations_df[recommendations_df['product_id'].isin(items_to_filter_recs)]                    
        valid_recs = valid_recs_df['product_id'].values
        return valid_recs[0]
        # return recommendations_df['product_id'].values[0]
        
        
class CFRecommender:
    
    MODEL_NAME = 'Collaborative Filtering(SVD)'
    
    def __init__(self, cf_predictions_df, items_df=None):
        self.cf_predictions_df = cf_predictions_df
        self.items_df = items_df
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        # Get and sort the user's predictions
        sorted_user_predictions = self.cf_predictions_df[user_id].sort_values(ascending=False) \
                                    .reset_index().rename(columns={user_id: 'recStrength'})

        # Recommend the highest predicted rating movies that the user hasn't seen yet.
        recommendations_df = sorted_user_predictions[~sorted_user_predictions['product_id'].isin(items_to_ignore)] \
                               .sort_values('recStrength', ascending = False) \
                               .head(topn)

        return recommendations_df
    
    def predict(self, user_id):
        recommendations_df = self.recommend_items(user_id=user_id, topn=100)
        rec_values = recommendations_df['product_id'].values
        return rec_values[0]
    
