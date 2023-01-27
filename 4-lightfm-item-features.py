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
#read feature jsons from features folder
with open('features/features-0-999.json') as f:
    features0 = json.load(f)

# %%
track_features = []
track_feature_columns = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'uri', 'duration_ms', 'time_signature']

path = 'features/'
filenames = ['features-0-999.json']
for filename in filenames:
    slice = json.loads(open(path + filename).read())
    for feature_list in slice:
        for feature in feature_list:
            track_features.append([feature[col] for col in track_feature_columns])

# %%
df_track_features = pd.DataFrame(track_features, columns=track_feature_columns)
df_track_features.head()

#merge above with df_tracks on track_uri
df_tracks = df_tracks.merge(df_track_features, left_on='track_uri', right_on='uri', how='left')
#drop uri column
df_tracks.drop('uri', axis=1, inplace=True)

# %%
# # item_features = []
# # col = ['danceability', 'energy', 'key', 'loudness']
# # unique_f1 = []
# # for c in col:
# #     unique_f1.extend(df_tracks[c].unique())

# # for x,y in zip(col, unique_f1):
# #     res = str(x)+ ":" +str(y)
# #     item_features.append(res)
# #     print(res)

# col = ['danceability', 'energy', 'key', 'loudness']
# unique_f1 = []
# for c in col:
#     print(c)
#     unique_f1.extend(df_tracks[c].unique())

# fit_item_features = {}
# for tid in df_tracks['tid'].unique():
#     item_df = df_tracks[df_tracks['tid'] == tid]
#     fit_item_features[tid] = {}
#     for x, y in zip(col, unique_f1):
#         fit_item_features[tid][x] = item_df[x].mean()


# %%
item_feature_cols = ['danceability', 'energy', 'key', 'loudness']#, 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'danceability', 'duration_ms', 'time_signature']

dataset = Dataset()
dataset.fit(df_playlists.pid, df_tracks.tid)

num_users, num_items = dataset.interactions_shape()
print('Num users: {}, num_items {}.'.format(num_users, num_items))

dataset.fit_partial(items=(row['tid'] for _, row in df_tracks.iterrows()),
                    item_features=(row['danceability'] for _, row in df_tracks.iterrows()))


# %%
# # change item_features from {tid: {feature: value}} to 
# # [
# # (tid1: ['feature1:value1', 'feature2:value2', ...])
# # (tid2: ['feature1:value1', 'feature2:value2', ...])
# # ]
# build_item_features = [(tid, [f'{k}:{v}' for k, v in features.items()]) for tid, features in fit_item_features.items()]

# build_item_features

# %%
(interactions, weights) = dataset.build_interactions(df_playlists[['pid', 'tid']].values)
print(repr(interactions))

# %%
item_features = dataset.build_item_features([(x[0], [x[1]]) for x in df_tracks[['tid', 'danceability']].values])
#item_features = dataset.build_item_features(build_item_features)
print(repr(item_features))
#split the data into train and test
train, test = random_train_test_split(interactions, test_percentage=0.2, random_state=np.random.RandomState(1337))

# %%
model = LightFM(loss='warp', no_components=50, learning_rate=0.02, random_state=1337)

model.fit(interactions, epochs=10, num_threads=8, item_features=item_features)

print("Train precision: %.2f" % precision_at_k(model, train, k=10, item_features=item_features).mean())
print("Test precision: %.2f" % precision_at_k(model, test, k=10, item_features=item_features).mean())



