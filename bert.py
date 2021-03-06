"""
BERT implementation on movie recommendation.

sources:
1) https://www.pragnakalp.com/nlp-tutorial-movie-recommendation-system-using-bert/
"""

from cmath import nan
import json
import ast
import pickle
from zlib import DEF_MEM_LEVEL
import pandas as pd
from yaml import KeyToken
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class BertRecommender:

    def __init__(self):
        try:
            self.df = pd.read_csv('cleaned_data.csv')
            print('\nloading existing pre-processed data...\n')
        except:
            print('\npre-processing data...\n')
            self.process_data()

    def process_data(self):
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
        print('\nfiltering genres...\n')
        for i in range(len(df_meta['genres'])):
            genre_str = ''
            genre = df_meta['genres'][i]
            genre = genre.replace("\'", "\"")
            json_genre = json.loads(genre)
            for j in range(len(json_genre)):
                genre_str += json_genre[j]['name'] + ' '
            df_meta['genres'][i] = genre_str

        # clean up production countries
        print('\nfiltering production countries...\n')
        for i in range(len(df_meta['production_countries'])):
            countries_str = ''
            country = df_meta['production_countries'][i]
            if (country != '' and country != nan):
                try:
                    country = json.dumps(ast.literal_eval(country))
                except:
                    df_meta['production_countries'][i] = ''
                    continue
                json_country = json.loads(country)
                # print(json_country)
                try:
                    for j in range(len(json_country)):
                        countries_str += (json_country[j]['name'])
                    # print(countries_str)
                    df_meta['production_countries'][i] = countries_str
                except:
                    print('Error....')
            else:
                print("Blank...")
                    
        # grab keywords data
        df_keywords = pd.read_csv('data/keywords.csv')

        # clean keywords
        print('\nfiltering keywords...\n')
        for i in range(len(df_keywords['keywords'])):
            keywords_str = ''
            keyword = df_keywords['keywords'][i]
            keyword = json.dumps(ast.literal_eval(keyword))
            json_keyword = json.loads(keyword)
            for j in range(len(json_keyword)):
                keywords_str += json_keyword[j]['name'] + " "
            df_keywords['keywords'][i] = keywords_str

        # grab cast from credits.csv
        df_credits = pd.read_csv('data/credits.csv')

        # filter cast 
        print('\nfiltering cast...\n')
        for i in range(len(df_credits['cast'])):
            cast_str = ''
            credits = df_credits['cast'][i]
            credits = json.dumps(ast.literal_eval(credits))
            json_credits = json.loads(credits)
            for j in range(len(json_credits)):
                cast_str += json_credits[j]['name'] + " "
            df_credits['cast'][i] = cast_str

        # filter crew to only the director (saved as 'crew' key in df_credits)
        print('\nfiltering directors...\n')
        for i in range(len(df_credits['crew'])):
            director_str = ''
            director = df_credits['crew'][i]
            director = json.dumps(ast.literal_eval(director))
            json_director = json.loads(director)
            for j in range(len(json_director)):
                if json_director[j]['job'] == 'Director':
                    director_str += json_director[j]['name'] + " "
            df_credits['crew'][i] = director_str


        # merge all labels
        df_meta['id'] = df_meta['id'].astype(str)
        df_keywords['id'] = df_keywords['id'].astype(str)

        df_merge = pd.merge(df_keywords, df_meta, on='id', how='inner')[[
            'id',
            'genres',
            'original_language',
            'production_countries',
            'tagline',
            'original_title',
            'keywords',
            'adult',
            'release_date',
            'status'
        ]]

        df_credits['id'] = df_credits['id'].astype(str)

        df_merge_whole = pd.merge(df_merge, df_credits, on='id', how='inner')[[
            'id',
            'genres',
            'original_language',
            'production_countries',
            'tagline',
            'original_title',
            'keywords',
            'cast',
            'crew', # director only ...
            'adult',
            'release_date',
            'status'
        ]]

        # replacing blanks with NAN and dropping
        df_merge_whole['keywords'].replace('', np.nan, inplace=True)
        df_merge_whole['genres'].replace('', np.nan, inplace=True)
        df_merge_whole['original_title'].replace('', np.nan, inplace=True)
        df_merge_whole['cast'].replace('', np.nan, inplace=True)
        df_merge_whole['crew'].replace('', np.nan, inplace=True)
        df_merge_whole['release_date'].replace('', np.nan, inplace=True)
        df = df_merge_whole.dropna()
        

        # helper function for combining columns
        def combine_features(row):
            return row['original_title'] + ' ' + row['genres'] + ' ' + row['original_language'] \
                + ' ' + row['crew'] + ' ' + row['keywords'] + ' ' + row['cast'] + ' ' \
                + row['tagline'] + ' ' + row['production_countries']

        # combine all columns to use for embedding
        df['combined_value'] = df.apply(combine_features, axis=1)
        df['index'] = [i for i in range(0, len(df))]

        print('\nsaving data...')
        df.to_csv('cleaned_data.csv')
        self.df = df

    def create_model(self):
        df = self.df
        print('\ninit bert\n')
        bert = SentenceTransformer('nq-distilbert-base-v1')

        try: # try loading pre-compouted embeddings
            with open('movie_rec_bert_embeddings.pkl', 'rb') as f:
                print('\nFound existing embeddings, loading...\n')
                data = pickle.load(f)
                self.embeddings = data['embeddings']
        except: # create new embeddings and save
            # create embeddings
            print('\ncreating embeddings\n')
            self.embeddings = bert.encode(
                df['combined_value'].tolist(),
                show_progress_bar=True,
                convert_to_numpy=True
            )

            # save embeddings
            print('\nsaving BERT embeddings on corpus\n')
            with open('movie_rec_bert_embeddings.pkl', 'wb') as f:
                pickle.dump({'embeddings': self.embeddings}, f)

        # compute similarity kernel
        print('\ncalculating cosine similarity\n')
        self.similarity = cosine_similarity(self.embeddings)
        self.df = df

    def prompt_model(self, prompt):
        df = self.df
        
        # get title of movie 
        def title(index):
            return df[df.index == index]['original_title'].values[0]

        # get index of movie title to grab its similarity matrix
        def index(original_title):
            print('index out: ')
            print(df[df.original_title == original_title]['index'])
            return df[df.original_title == original_title]['index'].values[0]

        self.movie_rec = sorted(list(enumerate(self.similarity[index(prompt)])), key=lambda x:x[1], reverse=True)
        print('Top 5 movie recommendations for %s:' % prompt)
        print(title(self.movie_rec[1][0]))
        print(title(self.movie_rec[2][0]))
        print(title(self.movie_rec[3][0]))
        print(title(self.movie_rec[4][0]))
        print(title(self.movie_rec[5][0]))



if __name__ == "__main__":
    Bert = BertRecommender()
    Bert.create_model()
    Bert.prompt_model('Inception')
    