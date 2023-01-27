# %%
import os
import json
import numpy as np
import pandas as pd
import scipy as sp
import scipy.sparse as sps
import lightfm
from lightfm import LightFM

# %%
path = 'data/'
filenames = os.listdir(path)

playlist_columns = ['pid', 'num_tracks', 'num_followers', 'num_edits', 'num_artists', 'num_albums', 'name', 'modified_at', 'duration_ms', 'collaborative']
playlist_test_columns = ['name', 'num_holdouts', 'num_samples', 'num_tracks', 'pid']
tracks_columns = ['album_name', 'album_uri', 'artist_name', 'artist_uri', 'duration_ms', 'track_name', 'track_uri']

data_playlists = [] #playlist metadata
playlists = [] #playlist track data
data_tracks = [] #track metadata
tracks = set() #unique track(ids)


for filename in filenames:
    slice = json.loads(open(path + filename).read())

    for playlist in slice['playlists']:
        data_playlists.append([playlist[col] for col in playlist_columns])
        for track in playlist['tracks']:
            playlists.append([playlist['pid'], track['track_uri'], track['pos']])
            if track['track_uri'] not in tracks:
                data_tracks.append([track[col] for col in tracks_columns])
                tracks.add(track['track_uri'])


# %%
data_playlists_test = []
playlist_test = []

challenge = json.loads(open('challenge_set.json').read())

for playlist in challenge['playlists']:
    data_playlists_test.append([playlist.get(col, '') for col in playlist_test_columns])
    for track in playlist['tracks']:
        playlist_test.append([playlist['pid'], track['track_uri'], track['pos']])
        if track['track_uri'] not in tracks:
            data_tracks.append([track[col] for col in tracks_columns])
            tracks.add(track['track_uri'])

# %%
df_data_playlists = pd.DataFrame(data_playlists, columns=playlist_columns)

df_tracks = pd.DataFrame(data_tracks, columns=tracks_columns)
df_tracks['tid'] = df_tracks.index
trackuri2tid = df_tracks.set_index('track_uri').tid

df_playlists = pd.DataFrame(playlists, columns=['pid', 'tid', 'pos'])
df_playlists.tid = df_playlists.tid.map(trackuri2tid)

df_data_playlists_test = pd.DataFrame(data_playlists_test, columns=playlist_test_columns)

df_playlists_test = pd.DataFrame(playlist_test, columns=['pid', 'tid', 'pos'])
df_playlists_test.tid = df_playlists_test.tid.map(trackuri2tid)

# %%
#save dataframes to csv
df_data_playlists.to_csv('dfs/data_playlists.csv', index=False)
df_data_playlists_test.to_csv('dfs/data_playlists_test.csv', index=False)
df_tracks.to_csv('dfs/data_tracks.csv', index=False)
df_playlists.to_csv('dfs/data_playlists_tracks.csv', index=False)
df_playlists_test.to_csv('dfs/data_playlists_tracks_test.csv', index=False)


