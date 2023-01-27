# %%
import os
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
import lightfm
from lightfm import LightFM
from lightfm.evaluation import precision_at_k
from lightfm.cross_validation import random_train_test_split
from tqdm import tqdm

# %%
df_tracks = pd.read_csv('dfs/data_tracks.csv')
df_playlists = pd.read_csv('dfs/data_playlists_tracks.csv')
df_playlists_info = pd.read_csv('dfs/data_playlists.csv')
df_playlists_test = pd.read_csv('dfs/data_playlists_tracks_test.csv')
df_playlists_test_info = pd.read_csv('dfs/data_playlists_test.csv')

# %%
config = {
    'num_playlists': df_playlists_test_info.pid.max() + 1,
    'num_tracks': df_tracks.tid.max() + 1,
}

# %%
mat = sp.coo_matrix(
    (np.ones(df_playlists.shape[0]), (df_playlists.pid, df_playlists.tid)),
    shape=(config['num_playlists'], config['num_tracks'])
)

X_train, X_test = random_train_test_split(mat, test_percentage=0.2, random_state=42)

config['model_path'] = 'models/lightfm_model.pkl'

# %%
model = LightFM(loss='warp', no_components=100, learning_rate=0.05, random_state=1337)

# %%
model.fit(X_train, epochs=10, num_threads=12)

print("Train precision: %.2f" % precision_at_k(model, X_train, k=50).mean())
print("Test precision: %.2f" % precision_at_k(model, X_test, k=50).mean())



