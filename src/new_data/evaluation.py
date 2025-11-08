import json
from tqdm import tqdm
from time import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

def load_data():
    # Load the dataset
    df = pd.read_csv("dataset.csv", index_col=0)
    df.drop_duplicates(subset=['explicit', 'danceability', 'energy', 'key', 'loudness', 'mode',
        'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'popularity',
        'artists', 'track_name', 'time_signature'], inplace=True) # There are duplicates that have different track_id, genre and album. There are duplicates in other dimensions (eg. popularity and duration) but these are taken

    # One-hot encode genres
    genre_dummies = pd.get_dummies(df['track_genre'], prefix='genre')
    time_signature_dummies = pd.get_dummies(df['time_signature'], prefix='time_signature')

    # Combine with original dataframe
    df_with_genres = pd.concat([df, genre_dummies, time_signature_dummies], axis=1)

    # Define columns to aggregate
    group_cols = [
        'track_id', 'explicit', 'danceability', 'energy', 'key', 'loudness', 'mode',
        'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'popularity',
        'artists', 'album_name', 'track_name'
    ] + list(time_signature_dummies.columns)

    # Aggregation dictionary
    agg_dict = {col: 'max' for col in genre_dummies.columns}
    for col in group_cols:
        agg_dict[col] = 'first'

    # Merge duplicates
    df_merged = df_with_genres.groupby('track_id', as_index=False).agg(agg_dict)

    # Final feature columns
    feature_columns = [
        'danceability', 'energy', 'key', 'loudness', 'mode',
        'speechiness', 'acousticness', 'instrumentalness',
        'liveness', 'valence', 'tempo', 'duration_ms', 'popularity',
        'explicit'
    ] + list(genre_dummies.columns) + list(time_signature_dummies.columns)

    # Drop missing values
    df_clean = df_merged.dropna(subset=feature_columns).copy()

    # Create feature matrix
    X = df_clean[feature_columns].values

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)


    with open('../testset.json', 'r') as f:
        testset = json.load(f)

    return df_clean, X_scaled, testset

class BaselineRecommender:
    def __init__(self, df, features_scaled):
        """
        Initialize the recommender system
        
        Args:
            df: DataFrame with song information including track_id
            features_scaled: Normalized feature matrix (after PCA)
        """
        self.df = df.reset_index(drop=True)
        self.features = features_scaled
        self.track_id_to_idx = {track_id: idx for idx, track_id in enumerate(self.df['track_id'])}
        
    def get_recommendations(self, input_track_ids, n_recommendations, target_artist):
        """
        Get recommendations based on multiple input songs
        
        Args:
            input_track_ids: List of track IDs to base recommendations on
            n_recommendations: Number of songs to recommend
            target_artist: If provided, only recommend songs from this artist
            
        Returns:
            List of recommended track IDs
        """
        valid_indices = [self.track_id_to_idx[item] for item in input_track_ids if item in self.track_id_to_idx]
        
        if len(valid_indices) == 0:
            # If no valid input tracks, return random recommendations
            artist_songs = self.df[self.df['artists'].isin(target_artist)]
            if len(artist_songs) >= n_recommendations:
                return artist_songs.sample(n_recommendations)['track_id'].tolist()
            return self.df.sample(n_recommendations)['track_id'].tolist()
        
        # Get features of input tracks and compute average profile
        input_features = self.features[valid_indices]
        avg_profile = np.mean(input_features, axis=0).reshape(1, -1)
        
        # Calculate similarity with all songs
        similarities = cosine_similarity(avg_profile, self.features)[0]
        
        # If target_artist is specified, filter to only songs from those artists
        artist_mask = self.df['artists'].isin(target_artist)
        # Set similarity to -inf for songs not by the target artists
        similarities = np.where(artist_mask, similarities, -np.inf)
        
        # Get indices of most similar songs
        similar_indices = np.argsort(similarities)[::-1]
        
        # Filter out input songs if requested
        similar_indices = [idx for idx in similar_indices if idx not in valid_indices]
        
        # Get top n recommendations
        recommended_indices = similar_indices[:n_recommendations]
        recommended_track_ids = self.df.iloc[recommended_indices]['track_id'].tolist()
        
        return recommended_track_ids

def recommender_metrics(recommender, testset, n_recommendations=5):
    """
    Evaluate the recommender system using the testset
    
    Args:
        recommender: ContentBasedRecommender instance
        testset: Dict of playlist_name -> [input_tracks, target_tracks] pairs
                 where each track is a [track_id, artist] pair
        n_recommendations: Number of recommendations from the system
        
    Returns:
        Dictionary with evaluation metrics
    """
    total_ndcg = 0
    
    for playlist_name, (input_tracks, target_tracks) in tqdm(testset.items()):
        # Extract track IDs from tracks
        # _tracks is a list of [track_id, artist] pairs
        input_track_ids = [track[0] for track in input_tracks]
        target_track_ids = [track[0] for track in target_tracks]
        target_set = set(target_track_ids)
        
        # Build a pool of candidate songs: only songs by artists in target_tracks
        target_artists = set([track[1] for track in target_tracks])
        
        # Get recommendations filtered by artists
        predictions = recommender.get_recommendations(
            input_track_ids, 
            n_recommendations=n_recommendations,
            target_artist=target_artists
        )
                
        # NDCG@K: Normalized Discounted Cumulative Gain
        # Binary relevance: 1 if the song is in target_tracks, 0 otherwise
        dcg = 0.0
        idcg = 0.0
        
        # Calculate DCG for predictions
        for rank, track_id in enumerate(predictions, start=1):
            relevance = 1 if track_id in target_set else 0
            dcg += relevance / np.log2(rank + 1)
        
        # Calculate IDCG (ideal DCG) - assumes all relevant items at top positions
        n_relevant = min(len(target_track_ids), n_recommendations)
        for rank in range(1, n_relevant + 1):
            idcg += 1.0 / np.log2(rank + 1)
        
        # Normalize DCG
        ndcg = dcg / idcg if idcg > 0 else 0.0
        total_ndcg += ndcg
        
    n_playlists = len(testset)
    
    metrics = {
        'NDCG@5': total_ndcg.item() / n_playlists
    }
    
    return metrics

def evaluate(recommender, n_recommendations=5):
    df_clean, X_scaled, testset = load_data()
    baseline = BaselineRecommender(df_clean, X_scaled)
    t0 = time()
    print('Testing recommender quality...')
    metrics = recommender_metrics(recommender, testset, n_recommendations)
    t1 = time()
    print('Testing recommender performance...')
    recommender_metrics(baseline, testset, n_recommendations)
    t2 = time()
    
    metrics['Performance'] = (t1-t0)/(t2-t1)

    return metrics
