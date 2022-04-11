"""
BERT implementation on movie recommendation.

sources:
1) https://www.pragnakalp.com/nlp-tutorial-movie-recommendation-system-using-bert/
"""

import json
import pandas as pd

# grab movie metadata 
df_meta = pd.read_csv("data/movies_metadata.csv")
df_meta = df_meta[[
    'id',
    'genres',
    'original_language',
    'production_countries',
    'tagline',
    'original_title',
    'adult',
    'release_date',
    'status'
]]

# filter out any unreleased films
for i in range(len(df_meta)):
    stat = df_meta['status'][i]
    if not stat == 'Released':
        df_meta['status'][i] = ''

# clean up genres to just one piece of raw text
for i in range(len(df_meta['genres'])):
    genre_str = ''
    genre = df_meta['genres'][i]
    genre = genre.replace("\'", "\"")
    json_genre = json.loads(genre)
    for j in range(len(json_genre)):
        genre_str += json_genre[j]['name'] + ' '
    df_meta['genres'][i] = genre_str

# clean up production countries