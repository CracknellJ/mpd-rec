# %%
import os
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
import lightfm
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k
from lightfm.cross_validation import random_train_test_split
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer

# %%
df_tracks = pd.read_csv('dfs/data_tracks.csv') #item features
df_playlists = pd.read_csv('dfs/data_playlists_tracks.csv') #user-items
df_playlists_info = pd.read_csv('dfs/data_playlists.csv') #user features
df_playlists_test = pd.read_csv('dfs/data_playlists_tracks_test.csv')
df_playlists_test_info = pd.read_csv('dfs/data_playlists_test.csv')

# %%
df_playlists_info.head()

#generator to get the names of the playlists in df_playlists_info
def get_playlist_names(df_playlists_info):
    for i, row in df_playlists_info.iterrows():
        yield row['name']
        

# %%
dataset = Dataset()
dataset.fit(df_playlists.pid, df_tracks.tid)


num_users, num_items = dataset.interactions_shape()
print('Num users: {}, num_items {}.'.format(num_users, num_items))

dataset.fit_partial(users=(row['pid'] for _, row in df_playlists_info.iterrows()),
                    user_features=(row['name'] for _, row in df_playlists_info.iterrows()))

# %%
(interactions, weights) = dataset.build_interactions(df_playlists[['pid', 'tid']].values)
print(repr(interactions))

# %%
df_playlists_info.head()


# %%
user_features = dataset.build_user_features([(x[0], [x[1]]) for x in df_playlists_info[['pid', 'name']].values])
print(repr(user_features))

#split the data into train and test
train, test = random_train_test_split(interactions, test_percentage=0.2, random_state=np.random.RandomState(1337))

# %%
model = LightFM(loss='warp', no_components=100, learning_rate=0.05, random_state=1337)

model.fit(interactions, epochs=10, num_threads=8, user_features=user_features)

print("Train precision: %.2f" % precision_at_k(model, train, k=50, user_features=user_features).mean())
print("Test precision: %.2f" % precision_at_k(model, test, k=50, user_features=user_features).mean())


# %%



