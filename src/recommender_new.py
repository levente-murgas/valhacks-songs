
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

        recommended_track_ids = []

        return recommended_track_ids



r = Recommender()
print(
    r.get_recommendations(
        [
            "7o2CTH4ctstm8TNelqjb51",
            "2zYzyRzz6pRmhPzyfMEC8s",
            "08mG3Y1vljYA6bvDt4Wqkj",
            "0bVtevEgtDIeRjCJbK3Lmv",
            "3YBZIN3rekqsKxbJc9FZko",
            "57bgtoPSgt236HzfBOd8kj",
            "7LRMbd3LEoV5wZJvXT1Lwb",
            "2SiXAy7TuUkycRVbbWDEpo",
            "0C80GCp0mMuBzLf3EAXqxv"
        ],
        2,
        {"AC/DC", "Europe", "Guns N' Roses"}
    )
)