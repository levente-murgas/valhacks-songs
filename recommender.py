from evaluation import evaluate

class Recommender:

    def get_recommendations(self, input_track_ids: list[str], n_recommendations: int, target_artist: set[str]) -> list[str]:
        """
        Get recommendations based on multiple input songs
        
        Args:
            input_track_ids: List of track IDs to base recommendations on
            n_recommendations: Integer specifying how many songs to recommend
            target_artist: A set of artist names. This is a hint of which artists were removed from the playlist. You may use this set to recommend songs.
            
        Returns:
            List of recommended track IDs of length n_recommendations
            The list should be ordered by relevance (most relevant first)
        """
        
        return recommended_track_ids

recommender = Recommender()

results = evaluate(recommender)
print(results)