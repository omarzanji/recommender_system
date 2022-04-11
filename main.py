'''
Movie Recommender System based on ROUNAK BANIK's notebook below:
https://www.kaggle.com/rounakbanik/movie-recommender-systems/notebook

author: Omar Barazanji
date: 2/13/22

Python Version: 3.7.6
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate # instead of depricated 'evaluate'
import warnings; warnings.simplefilter('ignore')


class Recommender:

    def __init__(self):
        # Contains metadata for IMDB movies
        self.metadata = pd.read_csv('data/movies_metadata.csv')
        self.md = self.metadata
        self.md['genres'] = self.md['genres'].fillna('[]').\
            apply(literal_eval).\
                apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
        self.md['year'] = pd.to_datetime(self.md['release_date'], errors='coerce').\
            apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
        self.md = self.md.drop([19730, 29503, 35587])
        self.md['id'] = self.md['id'].astype('int')

        # Prepare data for IMDB's weighted rating formula
        vote_counts = self.md[self.md['vote_count'].notnull()]['vote_count'].astype('int')
        vote_averages = self.md[self.md['vote_average'].notnull()]['vote_average'].astype('int')
        self.C = vote_averages.mean()
        self.m = vote_counts.quantile(0.95)

        # Contains TMDB and IMDB IDs of a small subset of 9,000 movies.
        links_small = pd.read_csv('data/links_small.csv')
        self.links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')

        self.credits = pd.read_csv('data/credits.csv')
        self.keywords = pd.read_csv('data/keywords.csv')

        self.ratings = pd.read_csv('data/ratings_small.csv')


    def simple_recommender(self):
        md = self.md
        C = self.C
        m = self.m
        """
        The Simple Recommender offers generalized recommnendations to every
        user based on movie popularity and (sometimes) genre. The basic idea
        behind this recommender is that movies that are more popular and more
        critically acclaimed will have a higher probability of being liked
        by the average audience. This model does not give personalized
        recommendations based on the user.

        The implementation of this model is extremely trivial.
        All we have to do is sort our movies based on ratings and
        popularity and display the top movies of our list.
        As an added step, we can pass in a genre argument to get
        the top movies of a particular genre. -ROUNAK BANIK
        """

        qualified = md[(md['vote_count'] >= m) & (md['vote_count'].\
            notnull()) & (md['vote_average'].\
                notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity', 'genres']]

        qualified['vote_count'] = qualified['vote_count'].astype('int')
        qualified['vote_average'] = qualified['vote_average'].astype('int')

        # IMDB's weighted rating
        def weighted_rating(x):
            v = x['vote_count']
            R = x['vote_average']
            return (v/(v+m) * R) + (m/(m+v) * C)

        qualified['wr'] = qualified.apply(weighted_rating, axis=1)
        qualified = qualified.sort_values('wr', ascending=False).head(250)

        # simple top 15 rated in dataset:
        print('top 15 rated movies:')
        print(qualified.head(15))

        # Creating top by genre:
        s = md.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)
        s.name = 'genre'
        gen_md = md.drop('genres', axis=1).join(s)

        def build_chart(genre, percentile=0.85):
            df = gen_md[gen_md['genre'] == genre]
            vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
            vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')
            C = vote_averages.mean()
            m = vote_counts.quantile(percentile)
            
            qualified = df[(df['vote_count'] >= m) & (df['vote_count'].notnull()) & (df['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity']]
            qualified['vote_count'] = qualified['vote_count'].astype('int')
            qualified['vote_average'] = qualified['vote_average'].astype('int')
            
            qualified['wr'] = qualified.apply(lambda x: (x['vote_count']/(x['vote_count']+m) * x['vote_average']) + (m/(m+x['vote_count']) * C), axis=1)
            qualified = qualified.sort_values('wr', ascending=False).head(250)
    
            return qualified

        # Top 15 Romance movies
        print('top 15 rated Romance movies:')
        print(build_chart('Romance').head(15))


    def content_based_recommender(self):
        md = self.md
        links_small = self.links_small
        
        # drop some data
        smd = md[md['id'].isin(links_small)]
        print("small movie dataset shape: ")
        print(smd.shape)

        # Movie Description Based Recommender
        smd['tagline'] = smd['tagline'].fillna('')
        smd['description'] = smd['overview'] + smd['tagline']
        smd['description'] = smd['description'].fillna('')

        # Tokenize and vectorize movie taglines, overviews, and descriptions
        tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
        tfidf_matrix = tf.fit_transform(smd['description'])

        print('vectorized descriptions shape: ')
        print(tfidf_matrix.shape)

        # Cosine Similarity
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

        smd = smd.reset_index()
        titles = smd['title']
        indices = pd.Series(smd.index, index=smd['title'])

        # function that returns the 30 most similar movies based on the cosine similarity score
        def get_recommendations(title):
            idx = indices[title]
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:31]
            movie_indices = [i[0] for i in sim_scores]
            return titles.iloc[movie_indices]        

        print(get_recommendations('The Dark Knight').head(10))

        self.smd = smd
        self.cosine_sim = cosine_sim
    

    def metadata_based_recommender(self):
        keywords = self.keywords
        credits = self.credits
        md = self.md
        smd = self.smd
        links_small = self.links_small

        keywords['id'] = keywords['id'].astype('int')
        credits['id'] = credits['id'].astype('int')
        md['id'] = md['id'].astype('int')

        print('metadata shape:')
        print(md.shape)

        md = md.merge(credits, on='id')
        md = md.merge(keywords, on='id')

        smd = md[md['id'].isin(links_small)]
        print('small movie dataset shape: ')
        print(smd.shape)

        smd['cast'] = smd['cast'].apply(literal_eval)
        smd['crew'] = smd['crew'].apply(literal_eval)
        smd['keywords'] = smd['keywords'].apply(literal_eval)
        smd['cast_size'] = smd['cast'].apply(lambda x: len(x))
        smd['crew_size'] = smd['crew'].apply(lambda x: len(x))

        def get_director(x):
            for i in x:
                if i['job'] == 'Director':
                    return i['name']
            return np.nan

        smd['director'] = smd['crew'].apply(get_director)
        smd['cast'] = smd['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
        smd['cast'] = smd['cast'].apply(lambda x: x[:3] if len(x) >=3 else x)

        smd['keywords'] = smd['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

        smd['cast'] = smd['cast'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])

        smd['director'] = smd['director'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))
        smd['director'] = smd['director'].apply(lambda x: [x, x, x])

        # Keywords
        s = smd.apply(lambda x: pd.Series(x['keywords']),axis=1).stack().reset_index(level=1, drop=True)
        s.name = 'keyword'

        s = s.value_counts()
        # Keywords occur in frequencies ranging from 1 to 610. We do not have 
        # any use for keywords that occur only once. Therefore, these can be safely removed.
        s = s[s > 1]

        stemmer = SnowballStemmer('english')

        def filter_keywords(x):
            words = []
            for i in x:
                if i in s:
                    words.append(i)
            return words

        smd['keywords'] = smd['keywords'].apply(filter_keywords)
        smd['keywords'] = smd['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
        smd['keywords'] = smd['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])

        smd['soup'] = smd['keywords'] + smd['cast'] + smd['director'] + smd['genres']
        smd['soup'] = smd['soup'].apply(lambda x: ' '.join(x))

        count = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
        count_matrix = count.fit_transform(smd['soup'])

        cosine_sim = cosine_similarity(count_matrix, count_matrix)

        smd = smd.reset_index()
        titles = smd['title']
        indices = pd.Series(smd.index, index=smd['title'])

        # def get_recommendations(title):
        #     idx = indices[title]
        #     sim_scores = list(enumerate(cosine_sim[idx]))
        #     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        #     sim_scores = sim_scores[1:31]
        #     movie_indices = [i[0] for i in sim_scores]
        #     return titles.iloc[movie_indices]

        # print(get_recommendations('The Dark Knight').head(10))

        # take the top 25 movies based on similarity scores and calculate the vote 
        # of the 60th percentile movie. Then, using this as the value of  m , 
        # we will calculate the weighted rating of each movie using IMDB's formula
        # like we did in the Simple Recommender section.      
        
        # IMDB's weighted rating
        def weighted_rating(x):
            C = self.C
            m = self.m
            v = x['vote_count']
            R = x['vote_average']
            return (v/(v+m) * R) + (m/(m+v) * C)
  
        def improved_recommendations(title):
            idx = indices[title]
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:26]
            movie_indices = [i[0] for i in sim_scores]
            
            movies = smd.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year']]
            vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')
            vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('int')
            C = vote_averages.mean()
            m = vote_counts.quantile(0.60)
            qualified = movies[(movies['vote_count'] >= m) & (movies['vote_count'].notnull()) & (movies['vote_average'].notnull())]
            qualified['vote_count'] = qualified['vote_count'].astype('int')
            qualified['vote_average'] = qualified['vote_average'].astype('int')
            qualified['wr'] = qualified.apply(weighted_rating, axis=1)
            qualified = qualified.sort_values('wr', ascending=False).head(10)
            return qualified

        print(improved_recommendations('The Dark Knight'))
        self.smd = smd
        self.indices = indices

    def collaborative_filtering(self):
        smd = self.smd
        indices = self.indices
        cosine_sim = self.cosine_sim

        reader = Reader()
        data = Dataset.load_from_df(self.ratings[['userId', 'movieId', 'rating']], reader)
        # data.split(n_folds=5) - depricated 
        svd = SVD()
        print('\ncollaborative filtering: ')
        print('cross validation results (RMSE and MAE): \n')
        print(cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5))
        trainset = data.build_full_trainset()
        svd.fit(trainset) # instead of depricated "train" method ...
        print(svd.predict(1, 302, 3))

        # Hybrid Recommender implementation
        # input: user id + movie title
        # output: top movies based on user's ratings 

        def convert_int(x):
            try:
                return int(x)
            except:
                return np.nan
        id_map = pd.read_csv('data/links_small.csv')[['movieId', 'tmdbId']]
        id_map['tmdbId'] = id_map['tmdbId'].apply(convert_int)
        id_map.columns = ['movieId', 'id']
        id_map = id_map.merge(smd[['title', 'id']], on='id').set_index('title')
        #id_map = id_map.set_index('tmdbId')

        indices_map = id_map.set_index('id')

        def hybrid(userId, title):
            idx = indices[title]
            tmdbId = id_map.loc[title]['id']
            #print(idx)
            movie_id = id_map.loc[title]['movieId']
            
            sim_scores = list(enumerate(cosine_sim[int(idx)]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:26]
            movie_indices = [i[0] for i in sim_scores]
            
            movies = smd.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year', 'id']]
            movies['est'] = movies['id'].apply(lambda x: svd.predict(userId, indices_map.loc[x]['movieId']).est)
            movies = movies.sort_values('est', ascending=False)
            return movies.head(10)

        print('\nHybrid Content Based + Collaborative Filtering')

        user = 1
        movie = 'The Dark Knight'

        print('Results for \nuser: %d\n movie: %s\n' % (user, movie))
        res = hybrid(user, movie)
        print(res)
        with open('user1.csv', 'w') as f:
            res.to_csv(f)

if __name__ == "__main__":

    # load in data
    rec = Recommender()

    # simple recommender
    # rec.simple_recommender()

    # content based recommender
    rec.content_based_recommender()

    # metadata based recommender
    rec.metadata_based_recommender()

    # Hybrid collaborative + content & metadata based filtering
    rec.collaborative_filtering()

    