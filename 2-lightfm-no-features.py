# %%
import os
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp

import lightfm
from lightfm import LightFM
# Import LightFM's evaluation metrics
from lightfm.evaluation import precision_at_k as lightfm_prec_at_k
from lightfm.evaluation import recall_at_k as lightfm_recall_at_k

# Import repo's evaluation metrics
from reco_utils.evaluation.python_evaluation import (
    precision_at_k, recall_at_k)
from lightfm.cross_validation import random_train_test_split
from lightfm.data import Dataset
from lightfm import cross_validation
from tqdm import tqdm

from reco_utils.common.timer import Timer
from reco_utils.recommender.lightfm.lightfm_utils import (
    track_model_metrics, prepare_test_df, prepare_all_predictions,
    compare_metric, similar_users, similar_items)

# %%
df_tracks = pd.read_csv('dfs/data_tracks.csv') #track metadata
df_playlists = pd.read_csv('dfs/data_playlists_tracks.csv') #playlist-track matrix
df_playlists_info = pd.read_csv('dfs/data_playlists.csv') #playlist metadata
df_playlists_user_item = df_playlists

#rename pid column to userID
df_playlists_user_item.rename(columns={'pid': 'userID'}, inplace=True)
df_playlists_user_item.rename(columns={'tid': 'itemID'}, inplace=True)

# df_playlists_test = pd.read_csv('dfs/data_playlists_tracks_test.csv')
# df_playlists_test_info = pd.read_csv('dfs/data_playlists_test.csv')

# %%
dataset = Dataset()

dataset.fit(users = df_playlists_info['pid'],
            items = df_tracks['tid'])

num_users, num_items = dataset.interactions_shape()
print('Num users: {}, num_items {}.'.format(num_users, num_items))

# %%
(interactions, weights) = dataset.build_interactions(df_playlists[['pid', 'tid']].values)

# %%
train_interactions, test_interactions = cross_validation.random_train_test_split(
    interactions, test_percentage=0.2,
    random_state=np.random.RandomState(1337))

print(f"Shape of train interactions: {train_interactions.shape}")
print(f"Shape of test interactions: {test_interactions.shape}")

# %%
model1 = LightFM(loss='warp', no_components=50, 
                 learning_rate=0.02,                 
                 random_state=np.random.RandomState(1337))

# %%
model1.fit(interactions=train_interactions, epochs=20)

# %%
uids, iids, interaction_data = cross_validation._shuffle(
    interactions.row, interactions.col, interactions.data, 
    random_state=np.random.RandomState(1337))

cutoff = int((1.0 - 0.2) * len(uids))
test_idx = slice(cutoff, None)

# %%
uid_map, ufeature_map, iid_map, ifeature_map = dataset.mapping()

# %%
with Timer() as test_time:
    test_df = prepare_test_df(test_idx, uids, iids, uid_map, iid_map, weights)
print(f"Took {test_time.interval:.1f} seconds for prepare and predict test data.")  
time_reco1 = test_time.interval

# %%
test_df.sample(5)


# %%
with Timer() as test_time:
    all_predictions = prepare_all_predictions(df_playlists_user_item, uid_map, iid_map, 
                                              interactions=train_interactions,
                                              model=model1, 
                                              num_threads=1)
print(f"Took {test_time.interval:.1f} seconds for prepare and predict all data.")
time_reco2 = test_time.interval

# %%
all_predictions.sample(5)

# %%
with Timer() as test_time:
    eval_precision = precision_at_k(rating_true=test_df, 
                                rating_pred=all_predictions, k=10)
    eval_recall = recall_at_k(test_df, all_predictions, k=10)
time_reco3 = test_time.interval

with Timer() as test_time:
    eval_precision_lfm = lightfm_prec_at_k(model1, test_interactions, 
                                           train_interactions, k=K).mean()
    eval_recall_lfm = lightfm_recall_at_k(model1, test_interactions, 
                                          train_interactions, k=K).mean()
time_lfm = test_time.interval
    
print(
    "------ Using Repo's evaluation methods ------",
    f"Precision@K:\t{eval_precision:.6f}",
    f"Recall@K:\t{eval_recall:.6f}",
    "\n------ Using LightFM evaluation methods ------",
    f"Precision@K:\t{eval_precision_lfm:.6f}",
    f"Recall@K:\t{eval_recall_lfm:.6f}", 
    sep='\n')


