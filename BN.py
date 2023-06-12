import numpy as np
import pandas as pd
import math
from tqdm import tqdm
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the movie ratings data
ratings_data = pd.read_csv('ratings.csv')
movies_data = pd.read_csv('movies.csv')

# Merge ratings and movies data on movieId
merged_data = pd.merge(ratings_data, movies_data, on='movieId')[['userId', 'movieId', 'rating', 'genres']]
total_size = len(merged_data)

# Get all unique genres in the dataset
all_genres = set(genre for sublist in merged_data['genres'].str.split('|') for genre in sublist)
unique_user_ids = merged_data['userId'].unique().tolist()
result_se = []

for i, user_id in tqdm(enumerate(unique_user_ids), total=len(unique_user_ids)):
    print("{} of {} processed".format(i, len(unique_user_ids)))
    user_data = merged_data[merged_data['userId'] == user_id][['userId', 'movieId', 'genres', 'rating']]

    # Add separate columns for each genre and indicate genre presence with 0 or 1
    for genre in all_genres:
        user_data[genre] = user_data['genres'].str.contains(genre).astype(int)
    user_data.drop('genres', axis=1, inplace=True)

    # Split the user data into train and test sets
    train_data, test_data = train_test_split(user_data, test_size=0.3, random_state=42)
    train_data.drop('movieId', axis=1, inplace=True)

    # ----------------------------------- BayesianNetwork -----------------------------------
    model = BayesianNetwork()                                                     
    # Add nodes for 'userId', 'rating', and genres
    model.add_nodes_from(['userId', 'rating'])
    model.add_nodes_from(all_genres)

    # Add edges between nodes
    model.add_edges_from([('userId', 'rating')])
    model.add_edges_from([(genre, 'rating') for genre in all_genres])
    # ---------------------------------------------------------------------------------------
    model.fit(train_data, estimator=BayesianEstimator)

    # Predict ratings for all movie IDs for a particular user
    test_movieId = pd.DataFrame({'movieId': test_data['movieId'].tolist()})
    actual_ratings = test_data['rating'].tolist()

    test_data.drop('rating', axis=1, inplace=True)
    test_data.drop('movieId', axis=1, inplace=True)
    test_data.to_csv("test_data2.csv", index = False)

    try:
        pred_ratings = model.predict(test_data)
    except Exception as e:
        continue
    
    # square error (without mean) for aggregation purpose later to calculate rmse after we iterate all users
    sample_size = len(actual_ratings)
    se = mean_squared_error(actual_ratings, pred_ratings) * sample_size
    result_se.append(se)

rmse = math.sqrt(sum(result_se)/total_size)
print(rmse)


