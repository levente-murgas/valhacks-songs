These JSON slices are copied from the open mirror of the Spotify Million Playlist Dataset (MPD)
hosted on HuggingFace at https://huggingface.co/datasets/jaxliu/Spotify_Million_Playlist_Dataset_Challenge.

Files currently included:
- mpd.slice.0-999.json
- mpd.slice.1000-1999.json
- mpd.slice.10000-10999.json
- mpd.slice.11000-11999.json
- mpd.slice.12000-12999.json

Each file contains 1,000 human-curated playlists with Spotify track URIs. They are used by
`src/recommender_new.py` to train the playlist co-occurrence model.
