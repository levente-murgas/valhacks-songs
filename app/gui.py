from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"
DATA_PATH = PROJECT_ROOT / "data"

if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from playlist_builder import (  # type: ignore[import]
    build_playlist,
    describe_playlist,
    load_catalog,
)

from data.recommender import Recommender  # type: ignore[import]
from spotify_export import (  # type: ignore[import]
    SpotifyExportError,
    push_playlist_to_spotify,
)


logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)


FEATURE_COLUMNS: Sequence[str] = (
    "popularity",
    "danceability",
    "energy",
    "valence",
    "acousticness",
    "instrumentalness",
    "speechiness",
    "liveness",
    "tempo",
    "loudness",
    "duration_ms",
)


@dataclass
class PlaylistView:
    title: str
    details: list[dict[str, Any]]
    explanations: list[dict[str, Any]]
    seed_track_ids: list[str]
    target_artists: list[str]
    track_ids: list[str]
    seed_resolutions: list[dict[str, Any]]


def playlist_view_to_dict(view: PlaylistView) -> dict[str, Any]:
    return {
        "title": view.title,
        "details": list(view.details),
        "explanations": list(view.explanations),
        "seed_track_ids": list(view.seed_track_ids),
        "target_artists": list(view.target_artists),
        "track_ids": list(view.track_ids),
        "seed_resolutions": list(view.seed_resolutions),
    }


def playlist_view_from_dict(data: Mapping[str, Any]) -> PlaylistView:
    return PlaylistView(
        title=str(data.get("title") or ""),
        details=[dict(item) for item in data.get("details", []) if isinstance(item, Mapping)],
        explanations=[
            dict(item) for item in data.get("explanations", []) if isinstance(item, Mapping)
        ],
        seed_track_ids=[str(item) for item in data.get("seed_track_ids", []) if item],
        target_artists=[str(item) for item in data.get("target_artists", []) if item],
        track_ids=[str(item) for item in data.get("track_ids", []) if item],
        seed_resolutions=[
            dict(item) for item in data.get("seed_resolutions", []) if isinstance(item, Mapping)
        ],
    )


@st.cache_resource(show_spinner=False)
def get_recommender() -> Recommender:
    return Recommender()


@st.cache_data(show_spinner=False)
def get_catalog() -> pd.DataFrame:
    return load_catalog()


@st.cache_data(show_spinner=False)
def load_sample_playlists() -> list[PlaylistView]:
    playlists: list[PlaylistView] = []
    for path in sorted(DATA_PATH.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        if isinstance(payload, dict) and "playlist" in payload:
            title = payload.get("title") or path.stem.replace("_", " ").title()
            raw_details = payload.get("playlist") or []
            if not isinstance(raw_details, list):
                continue
            details: list[dict[str, Any]] = []
            track_ids: list[str] = []
            for entry in raw_details:
                if not isinstance(entry, dict):
                    continue
                detail = dict(entry)
                details.append(detail)
                track_id = detail.get("track_id")
                if track_id:
                    track_ids.append(str(track_id))
            raw_explanations = payload.get("explanations") or []
            explanations = [
                dict(explanation)
                for explanation in raw_explanations
                if isinstance(explanation, dict)
            ]
            seed_track_ids = [
                str(track_id)
                for track_id in (payload.get("seed_track_ids") or [])
                if track_id
            ]
            target_artists = [
                str(artist)
                for artist in (payload.get("target_artists") or [])
                if artist
            ]
            playlists.append(
                PlaylistView(
                    title=title,
                    details=details,
                    explanations=explanations,
                    seed_track_ids=seed_track_ids,
                    target_artists=target_artists,
                    track_ids=track_ids,
                    seed_resolutions=[],
                )
            )
        elif isinstance(payload, list):
            for item in payload:
                if not isinstance(item, dict):
                    continue
                tracks = item.get("tracks")
                if not isinstance(tracks, list):
                    continue
                title = item.get("title") or path.stem.replace("_", " ").title()
                details: list[dict[str, Any]] = []
                track_ids: list[str] = []
                for idx, track in enumerate(tracks):
                    if not isinstance(track, dict):
                        continue
                    track_id = track.get("track_id")
                    if track_id:
                        track_ids.append(str(track_id))
                    details.append(
                        {
                            "position": idx + 1,
                            "track_id": track_id,
                            "track_name": track.get("track_name"),
                            "artists": track.get("artists"),
                            "album_name": track.get("album_name"),
                            "track_genre": track.get("track_genre"),
                            "popularity": track.get("popularity"),
                        }
                    )
                playlists.append(
                    PlaylistView(
                        title=str(title),
                        details=details,
                        explanations=[],
                        seed_track_ids=[],
                        target_artists=[],
                        track_ids=track_ids,
                        seed_resolutions=[],
                    )
                )
    return playlists


def split_artists(value: Any) -> list[str]:
    if value is None:
        return []
    return [artist.strip() for artist in str(value).split(";") if artist.strip()]


def resolve_seed_inputs(
    raw_inputs: Sequence[str],
    catalog: pd.DataFrame,
) -> tuple[list[str], list[dict[str, Any]], list[str]]:
    if catalog.empty:
        return [], [], ["Catalog is empty; cannot resolve seed inputs."]

    deduped = catalog.drop_duplicates(subset="track_id", keep="first").copy()
    for column in ("track_id", "track_name", "artists", "track_genre"):
        if column in deduped.columns:
            deduped[column] = deduped[column].fillna("").astype(str)
        else:
            deduped[column] = ""
    if "popularity" not in deduped.columns:
        deduped["popularity"] = 0

    deduped = deduped.set_index("track_id", drop=False)

    resolved_track_ids: list[str] = []
    resolutions: list[dict[str, Any]] = []
    warnings: list[str] = []

    for raw_value in raw_inputs:
        query = raw_value.strip()
        if not query:
            continue
        query_lower = query.lower()

        if query in deduped.index:
            row = deduped.loc[query]
            track_id = str(row["track_id"])
            if track_id not in resolved_track_ids:
                resolved_track_ids.append(track_id)
            resolutions.append(
                {
                    "input": query,
                    "resolved_track_id": track_id,
                    "resolved_track_name": row.get("track_name"),
                    "resolved_artists": row.get("artists"),
                    "resolution_type": "track_id",
                }
            )
            continue

        name_matches = deduped[
            deduped["track_name"].str.strip().str.lower() == query_lower
        ]
        if not name_matches.empty:
            match = name_matches.sort_values("popularity", ascending=False).iloc[0]
            track_id = str(match["track_id"])
            if track_id not in resolved_track_ids:
                resolved_track_ids.append(track_id)
            resolutions.append(
                {
                    "input": query,
                    "resolved_track_id": track_id,
                    "resolved_track_name": match.get("track_name"),
                    "resolved_artists": match.get("artists"),
                    "resolution_type": "track_name",
                }
            )
            continue

        artist_matches = deduped[
            deduped["artists"].str.contains(re.escape(query), case=False, na=False)
        ]
        if not artist_matches.empty:
            match = artist_matches.sample(n=1).iloc[0]
            track_id = str(match["track_id"])
            if track_id not in resolved_track_ids:
                resolved_track_ids.append(track_id)
            resolutions.append(
                {
                    "input": query,
                    "resolved_track_id": track_id,
                    "resolved_track_name": match.get("track_name"),
                    "resolved_artists": match.get("artists"),
                    "resolution_type": "artist",
                }
            )
            continue

        genre_matches = deduped[
            deduped["track_genre"].str.strip().str.lower() == query_lower
        ]
        if not genre_matches.empty:
            match = genre_matches.sample(n=1).iloc[0]
            track_id = str(match["track_id"])
            if track_id not in resolved_track_ids:
                resolved_track_ids.append(track_id)
            resolutions.append(
                {
                    "input": query,
                    "resolved_track_id": track_id,
                    "resolved_track_name": match.get("track_name"),
                    "resolved_artists": match.get("artists"),
                    "resolution_type": "genre",
                }
            )
            continue

        warnings.append(f"Could not resolve seed '{query}' to a track, artist, or genre.")

    return resolved_track_ids, resolutions, warnings


def enrich_playlist_details(
    details: list[dict[str, Any]],
    dataset: pd.DataFrame,
    explanations: Iterable[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    if not details:
        return pd.DataFrame()

    deduped = dataset.drop_duplicates(subset="track_id", keep="first").set_index("track_id")

    records: list[dict[str, Any]] = []
    for item in details:
        track_id = str(item.get("track_id") or "")
        record = dict(item)
        if track_id and track_id in deduped.index:
            row = deduped.loc[track_id]
            for column in FEATURE_COLUMNS:
                record[column] = row.get(column)
        records.append(record)

    df = pd.DataFrame(records)
    if explanations:
        explanation_df = pd.DataFrame(list(explanations))
        if not explanation_df.empty and "track_id" in explanation_df.columns:
            df = df.merge(explanation_df, on="track_id", how="left", suffixes=("", "_detail"))
    return df


def compute_artist_counts(df: pd.DataFrame) -> pd.Series:
    if "artists" not in df.columns:
        return pd.Series(dtype=int)
    exploded = (
        df["artists"]
        .dropna()
        .apply(split_artists)
        .explode()
        .dropna()
        .astype(str)
        .str.strip()
    )
    if exploded.empty:
        return pd.Series(dtype=int)
    return exploded.value_counts().sort_values(ascending=False)


def render_summary(df: pd.DataFrame) -> None:
    track_count = int(df.shape[0])
    artist_counts = compute_artist_counts(df)
    unique_artists = int(artist_counts.index.nunique()) if not artist_counts.empty else 0
    avg_popularity = float(df["popularity"].dropna().mean()) if "popularity" in df else float("nan")

    feature_metrics = {}
    for column in ("danceability", "energy", "valence", "tempo"):
        if column in df:
            feature_metrics[column] = float(df[column].dropna().mean())

    metric_cols = st.columns(4)
    metric_cols[0].metric("Tracks", track_count)
    metric_cols[1].metric("Unique Artists", unique_artists)
    if not pd.isna(avg_popularity):
        metric_cols[2].metric("Avg Popularity", f"{avg_popularity:.1f}")
    if "tempo" in feature_metrics:
        metric_cols[3].metric("Avg Tempo (BPM)", f"{feature_metrics['tempo']:.1f}")

    secondary_cols = st.columns(3)
    if "danceability" in feature_metrics:
        secondary_cols[0].metric("Avg Danceability", f"{feature_metrics['danceability']:.2f}")
    if "energy" in feature_metrics:
        secondary_cols[1].metric("Avg Energy", f"{feature_metrics['energy']:.2f}")
    if "valence" in feature_metrics:
        secondary_cols[2].metric("Avg Valence", f"{feature_metrics['valence']:.2f}")

    if not artist_counts.empty:
        st.subheader("Artist Distribution")
        st.bar_chart(artist_counts.to_frame("count"))

    characteristic_columns = [
        column
        for column in ("danceability", "energy", "valence", "acousticness", "instrumentalness")
        if column in df
    ]
    if characteristic_columns:
        averages = df[characteristic_columns].mean().dropna()
        if not averages.empty:
            st.subheader("Average Audio Characteristics")
            st.bar_chart(averages.to_frame(name="average"))


def render_track_table(df: pd.DataFrame) -> None:
    display_columns = [
        "position",
        "track_name",
        "artists",
        "popularity",
        "danceability",
        "energy",
        "valence",
        "tempo",
        "artist_diversity_penalty",
        "adjusted_score",
    ]
    available_columns = [column for column in display_columns if column in df.columns]
    if not available_columns:
        return
    table_df = df[available_columns].copy()
    table_df = table_df.sort_values("position") if "position" in table_df.columns else table_df
    st.subheader("Track Details")
    st.dataframe(table_df.set_index("position") if "position" in table_df.columns else table_df)


def render_explanations(df: pd.DataFrame) -> None:
    score_columns = [
        column
        for column in (
            "base_similarity",
            "short_term_similarity",
            "genre_weight",
            "artist_overlap",
            "target_artist_boost",
            "explicit_penalty",
            "novelty_bonus",
            "variance_penalty",
            "final_score",
            "artist_diversity_penalty",
            "adjusted_score",
        )
        if column in df.columns
    ]
    if not score_columns:
        return
    score_df = (
        df[["track_name", "artists"] + score_columns]
        .dropna(subset=["adjusted_score"], how="all")
        .sort_values("adjusted_score", ascending=False)
    )
    if score_df.empty:
        return
    with st.expander("Scoring Breakdown"):
        st.dataframe(score_df.reset_index(drop=True))


def render_spotify_export(view: PlaylistView) -> None:
    if not view.track_ids:
        return

    state_seed = "||".join(view.track_ids)
    state_key = hashlib.md5(state_seed.encode("utf-8")).hexdigest()

    with st.expander("Add to Spotify", expanded=False):
        default_name = view.title or "ValHacks Playlist"
        playlist_name = st.text_input(
            "Playlist name",
            value=default_name,
            key=f"playlist_name_{state_key}",
        )
        description = st.text_area(
            "Description (optional)",
            placeholder="Describe the vibe, collaborators, or context for this playlist.",
            key=f"playlist_description_{state_key}",
            height=80,
        )
        public = st.checkbox(
            "Make playlist public",
            value=False,
            key=f"playlist_public_{state_key}",
        )
        show_advanced = st.checkbox(
            "Advanced authentication options",
            value=False,
            key=f"playlist_advanced_{state_key}",
        )

        client_kwargs: dict[str, Any] = {}
        if show_advanced:
            client_id = st.text_input(
                "Spotify Client ID",
                key=f"spotify_client_id_{state_key}",
            )
            if client_id.strip():
                client_kwargs["client_id"] = client_id.strip()

            client_secret = st.text_input(
                "Spotify Client Secret",
                type="password",
                key=f"spotify_client_secret_{state_key}",
            )
            if client_secret.strip():
                client_kwargs["client_secret"] = client_secret.strip()

            redirect_uri = st.text_input(
                "Spotify Redirect URI",
                key=f"spotify_redirect_uri_{state_key}",
            )
            if redirect_uri.strip():
                client_kwargs["redirect_uri"] = redirect_uri.strip()

            cache_path = st.text_input(
                "Token cache path (optional)",
                key=f"spotify_cache_path_{state_key}",
            )
            if cache_path.strip():
                client_kwargs["cache_path"] = cache_path.strip()

        export_clicked = st.button(
            "Add to Spotify",
            type="primary",
            key=f"spotify_export_button_{state_key}",
        )

        if export_clicked:
            trimmed_name = playlist_name.strip()
            if not trimmed_name:
                st.warning("Please provide a playlist name.")
                return

            description_value = description.strip() if description else ""
            attempt_msg = (
                f"Attempting export: {len(view.track_ids)} tracks -> '{trimmed_name}' "
                f"(public={public}, advanced_auth={bool(client_kwargs)})"
            )
            logger.info(attempt_msg)

            with st.spinner("Creating playlist on Spotify..."):
                try:
                    result = push_playlist_to_spotify(
                        view.track_ids,
                        trimmed_name,
                        description=description_value or None,
                        public=public,
                        **client_kwargs,
                    )
                except SpotifyExportError as exc:
                    logger.error("Spotify export failed: %s", exc)
                    st.session_state["last_spotify_export"] = {
                        "status": "error",
                        "message": f"Spotify export failed: {exc}",
                    }
                    st.session_state["pending_export_banner"] = True
                    st.error(f"Spotify export failed: {exc}")
                except Exception as exc:  # pragma: no cover - safety net for unexpected errors
                    logger.exception("Unexpected error during Spotify export")
                    st.session_state["last_spotify_export"] = {
                        "status": "error",
                        "message": f"Unexpected error: {exc}",
                    }
                    st.session_state["pending_export_banner"] = True
                    st.error(f"Unexpected error: {exc}")
                else:
                    playlist_url = result.get("playlist_url")
                    if playlist_url:
                        success_msg = f"Playlist created: {playlist_url}"
                        logger.info("Spotify export succeeded: %s", playlist_url)
                        st.session_state["last_spotify_export"] = {
                            "status": "success",
                            "message": success_msg,
                            "url": playlist_url,
                        }
                        st.session_state["pending_export_banner"] = True
                        st.success(f"Playlist created! [Open in Spotify]({playlist_url})")
                    else:
                        success_msg = "Playlist created on Spotify."
                        logger.info("Spotify export succeeded without URL response.")
                        st.session_state["last_spotify_export"] = {
                            "status": "success",
                            "message": success_msg,
                        }
                        st.session_state["pending_export_banner"] = True
                        st.success("Playlist created on Spotify.")


def display_export_banner() -> None:
    if not st.session_state.get("pending_export_banner"):
        return
    event = st.session_state.get("last_spotify_export")
    if not isinstance(event, dict):
        st.session_state["pending_export_banner"] = False
        return
    message = str(event.get("message") or "")
    status = event.get("status")
    if not message:
        st.session_state["pending_export_banner"] = False
        return
    if status == "success":
        st.success(message)
    elif status == "error":
        st.error(message)
    else:
        st.info(message)
    st.session_state["pending_export_banner"] = False


def generate_playlist_view(
    seed_track_ids: Sequence[str],
    target_artists: Iterable[str],
    total_tracks: int,
    step_size: int,
    seed_resolutions: list[dict[str, Any]] | None = None,
) -> PlaylistView:
    recommender = get_recommender()
    playlist_ids, explanations = build_playlist(
        seed_track_ids=seed_track_ids,
        target_artist=target_artists,
        total_tracks=total_tracks,
        step_size=max(1, step_size),
        recommender=recommender,
    )
    catalog = get_catalog()
    details = describe_playlist(playlist_ids, catalog)
    title = "Generated Playlist"
    if target_artists:
        title = f"Generated Playlist · Focus on {', '.join(target_artists)}"
    return PlaylistView(
        title=title,
        details=details,
        explanations=explanations,
        seed_track_ids=list(seed_track_ids),
        target_artists=list(target_artists),
        track_ids=[str(track_id) for track_id in playlist_ids if track_id],
        seed_resolutions=list(seed_resolutions or []),
    )


def render_playlist(view: PlaylistView, dataset: pd.DataFrame) -> None:
    st.header(view.title)
    if view.seed_resolutions:
        bullet_lines: list[str] = []
        for resolution in view.seed_resolutions:
            input_label = str(resolution.get("input") or "").strip()
            resolution_type = str(resolution.get("resolution_type") or "").strip()
            track_name = str(resolution.get("resolved_track_name") or "").strip()
            artists = str(resolution.get("resolved_artists") or "").strip()
            track_id = str(resolution.get("resolved_track_id") or "").strip()

            if resolution_type in {"artist", "genre"} and input_label:
                prefix = f"{input_label} ({resolution_type})"
            else:
                prefix = input_label or resolution_type or "Seed"

            resolved_display = track_name
            if artists:
                resolved_display = f"{resolved_display} — {artists}" if resolved_display else artists

            summary = f"{prefix}"
            if resolved_display:
                summary += f" → {resolved_display}"
            if track_id:
                summary += f" ({track_id})"
            bullet_lines.append(f"- {summary}")

        if bullet_lines:
            st.markdown("**Seed inputs**\n" + "\n".join(bullet_lines))
    elif view.seed_track_ids:
        st.caption(f"Seeds: {', '.join(view.seed_track_ids)}")
    if view.target_artists:
        st.caption(f"Target artists: {', '.join(view.target_artists)}")

    df = enrich_playlist_details(view.details, dataset, view.explanations)
    if df.empty:
        st.info("No tracks to display yet. Try generating a playlist or loading a sample.")
        return

    render_summary(df)
    render_track_table(df)
    render_explanations(df)

    score_chart_columns = {"track_name", "adjusted_score"}.issubset(df.columns)
    if score_chart_columns:
        chart_data = df.dropna(subset=["adjusted_score"])[["track_name", "adjusted_score"]]
        if not chart_data.empty:
            st.subheader("Adjusted Score by Track")
            st.bar_chart(chart_data.set_index("track_name"))

    render_spotify_export(view)
    display_export_banner()


def main() -> None:
    st.set_page_config(page_title="ValHacks Playlist Explorer", layout="wide")
    st.title("ValHacks Playlist Explorer")
    st.markdown(
        "Interactively explore generated playlists, inspect their characteristics, and review scoring heuristics."
    )

    catalog = get_catalog()
    sample_playlists = load_sample_playlists()

    with st.sidebar:
        st.header("Playlist Source")
        source = st.radio(
            "Select how to populate the view",
            options=("Generate with seeds", "Load sample playlist"),
            key="playlist_source",
        )

        current_view: PlaylistView | None = None

        if source == "Load sample playlist":
            if not sample_playlists:
                st.info("No sample playlists found in the data directory.")
            else:
                titles = [playlist.title for playlist in sample_playlists]
                default_index = titles.index("My Playlist") if "My Playlist" in titles else 0
                selected_title = st.selectbox(
                    "Sample playlists",
                    titles,
                    index=default_index,
                    key="sample_playlist_selection",
                )
                current_view = next(
                    (playlist for playlist in sample_playlists if playlist.title == selected_title),
                    None,
                )
        else:
            seed_input = st.text_area(
                "Seed track IDs",
                placeholder="Enter one Spotify track ID per line",
            )
            target_input = st.text_input(
                "Target artists (optional)",
                placeholder="Comma-separated artist names",
            )
            total_tracks = st.slider("Total tracks", min_value=5, max_value=40, value=15)
            step_size = st.slider("Recommendation batch size", min_value=1, max_value=10, value=5)
            generate_clicked = st.button("Generate playlist", type="primary")

            if generate_clicked:
                seeds = [line.strip() for line in seed_input.splitlines() if line.strip()]
                if not seeds:
                    st.warning("Provide at least one seed track ID to generate a playlist.")
                else:
                    resolved_seed_ids, seed_resolutions, resolution_warnings = resolve_seed_inputs(
                        seeds, catalog
                    )
                    for message in resolution_warnings:
                        st.warning(message)
                    if not resolved_seed_ids:
                        st.warning(
                            "None of the provided seeds could be matched to tracks, artists, or genres."
                        )
                        current_view = None
                        st.session_state.pop("generated_playlist", None)
                        st.session_state.pop("active_playlist_view", None)
                        st.session_state.pop("last_spotify_export", None)
                        st.session_state.pop("pending_export_banner", None)
                        st.stop()
                    target_artists = [artist.strip() for artist in target_input.split(",") if artist.strip()]
                    with st.spinner("Running recommender..."):
                        try:
                            current_view = generate_playlist_view(
                                seed_track_ids=resolved_seed_ids,
                                target_artists=target_artists,
                                total_tracks=total_tracks,
                                step_size=step_size,
                                seed_resolutions=seed_resolutions,
                            )
                            st.session_state["generated_playlist"] = playlist_view_to_dict(current_view)
                        except Exception as exc:  # pragma: no cover - surfaced in UI
                            st.error(f"Unable to generate a playlist: {exc}")
            else:
                cached_view = st.session_state.get("generated_playlist")
                if isinstance(cached_view, Mapping):
                    try:
                        current_view = playlist_view_from_dict(cached_view)
                    except Exception:  # pragma: no cover - defensive
                        current_view = None

    if current_view is not None:
        st.session_state["active_playlist_view"] = playlist_view_to_dict(current_view)
    else:
        stored_view = st.session_state.get("active_playlist_view")
        if isinstance(stored_view, Mapping):
            try:
                current_view = playlist_view_from_dict(stored_view)
            except Exception:  # pragma: no cover - defensive
                current_view = None

    if current_view is None:
        st.info("Select a sample playlist or generate one using seed tracks to get started.")
        return

    render_playlist(current_view, catalog)


if __name__ == "__main__":
    main()

