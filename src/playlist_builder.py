from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from data.recommender import Recommender

DATASET_PATH = PROJECT_ROOT / "data" / "dataset.csv"


def load_catalog(path: Path = DATASET_PATH) -> pd.DataFrame:
    if not path.exists():
        msg = f"Dataset not found at {path}"
        raise FileNotFoundError(msg)
    df = pd.read_csv(path)
    return df


def pick_random_track_id(catalog: pd.DataFrame) -> str:
    if "track_id" not in catalog.columns:
        raise KeyError("Catalog is missing the 'track_id' column.")
    track_id = random.choice(catalog["track_id"].astype(str).tolist())
    return track_id


def build_playlist(
    seed_track_ids: Sequence[str],
    target_artist: Iterable[str],
    total_tracks: int,
    step_size: int,
    recommender: Recommender,
) -> tuple[list[str], list[dict]]:
    playlist: list[str] = list(seed_track_ids)
    explanations: list[dict] = []

    target_artist_set = set(target_artist)

    while len(playlist) < total_tracks:
        n_to_request = min(step_size, total_tracks - len(playlist))
        recommendations, step_explanations = recommender.get_recommendations_with_details(
            input_track_ids=playlist,
            n_recommendations=n_to_request,
            target_artist=target_artist_set,
        )

        playlist.extend(recommendations)
        explanations.extend(step_explanations)

        if not recommendations:
            break

    return playlist[:total_tracks], explanations


def describe_playlist(track_ids: Sequence[str], catalog: pd.DataFrame) -> list[dict]:
    catalog_indexed = catalog.drop_duplicates(subset="track_id", keep="first").set_index("track_id")
    descriptions: list[dict] = []
    for position, track_id in enumerate(track_ids, start=1):
        if track_id in catalog_indexed.index:
            row = catalog_indexed.loc[track_id]
            popularity_value = row.get("popularity")
            popularity: int | None
            if pd.notna(popularity_value):
                try:
                    popularity = int(popularity_value)
                except (TypeError, ValueError):
                    popularity = None
            else:
                popularity = None

            descriptions.append(
                {
                    "position": int(position),
                    "track_id": track_id,
                    "track_name": row.get("track_name"),
                    "artists": row.get("artists"),
                    "album_name": row.get("album_name"),
                    "track_genre": row.get("track_genre"),
                    "popularity": popularity,
                }
            )
        else:
            descriptions.append(
                {
                    "position": int(position),
                    "track_id": track_id,
                    "track_name": None,
                    "artists": None,
                    "album_name": None,
                    "track_genre": None,
                    "popularity": None,
                }
            )
    return descriptions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a playlist using the recommender system.")
    parser.add_argument(
        "--seed-track-id",
        help="Spotify track ID to seed the playlist. If omitted, a random track from the dataset is used.",
    )
    parser.add_argument(
        "--target-artist",
        action="append",
        default=[],
        help="Artist name to prioritize in recommendations (may be supplied multiple times).",
    )
    parser.add_argument(
        "--total-tracks",
        type=int,
        default=10,
        help="Total number of tracks desired in the final playlist (including the seed).",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=5,
        help="How many new tracks to request per iteration while building the playlist.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output the playlist as JSON for downstream consumption.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to save the generated playlist as JSON.",
    )
    parser.add_argument(
        "--text-output",
        type=Path,
        help="Optional path to write a plain-text list of 'Artist - Title' lines.",
    )
    return parser.parse_args()


def sanitize(obj: object) -> object:
    if isinstance(obj, dict):
        return {key: sanitize(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [sanitize(item) for item in obj]
    if isinstance(obj, tuple):
        return [sanitize(item) for item in obj]
    if isinstance(obj, set):
        return [sanitize(item) for item in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def main() -> None:
    args = parse_args()

    catalog = load_catalog()
    seed_track_id = args.seed_track_id or pick_random_track_id(catalog)
    print(f"Seed track ID: {seed_track_id}")

    recommender = Recommender()

    playlist_ids, explanations = build_playlist(
        seed_track_ids=[seed_track_id],
        target_artist=args.target_artist,
        total_tracks=args.total_tracks,
        step_size=max(1, args.step_size),
        recommender=recommender,
    )

    playlist_details = describe_playlist(playlist_ids, catalog)

    payload: dict[str, object] = {
        "seed_track_id": seed_track_id,
        "target_artists": args.target_artist,
        "total_tracks": len(playlist_ids),
        "playlist": playlist_details,
        "explanations": explanations,
    }
    if args.json:
        print(json.dumps(sanitize(payload), indent=2))
    elif args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(sanitize(payload), indent=2), encoding="utf-8")
        print(f"Playlist saved to {args.output}")
    else:
        print("\nGenerated Playlist:")
        for detail in playlist_details:
            track_name = detail["track_name"] or "(unknown)"
            artists = detail["artists"] or "(unknown artist)"
            print(f"{detail['position']:02d}. {track_name} — {artists} [{detail['track_id']}]")

        if explanations:
            print("\nRecommendation Details:")
            for idx, info in enumerate(explanations, start=1):
                print(f"#{idx}: {sanitize(info)}")

    if args.text_output:
        lines = []
        for detail in playlist_details:
            track_name = detail["track_name"] or "(unknown)"
            artists = detail["artists"] or "(unknown artist)"
            lines.append(f"{artists} - {track_name}")
        args.text_output.parent.mkdir(parents=True, exist_ok=True)
        args.text_output.write_text("\n".join(lines), encoding="utf-8")
        print(f"Text playlist saved to {args.text_output}")


if __name__ == "__main__":
    main()

