from __future__ import annotations

import logging
from typing import Iterable, Sequence
from urllib.parse import urlparse

import spotipy
from spotipy import Spotify
from spotipy.oauth2 import SpotifyOAuth

logger = logging.getLogger(__name__)

DEFAULT_SCOPES = ("playlist-modify-private", "playlist-modify-public")
TRACK_ID_PREFIX = "spotify:track:"


class SpotifyExportError(RuntimeError):
    """Raised when exporting a playlist to Spotify fails."""


def _normalize_track_id(track_id: str) -> str:
    """
    Normalize various Spotify track ID formats to the URI form required by the Web API.

    Args:
        track_id: Spotify track identifier (URI, URL, or raw ID).

    Returns:
        Normalized Spotify track URI.

    Raises:
        ValueError: If the track_id is empty or cannot be parsed.
    """
    cleaned = (track_id or "").strip()
    if not cleaned:
        raise ValueError("Empty track ID.")

    if cleaned.startswith(TRACK_ID_PREFIX):
        return cleaned

    if cleaned.startswith("http://") or cleaned.startswith("https://"):
        parsed = urlparse(cleaned)
        segments = [segment for segment in parsed.path.split("/") if segment]
        candidate = ""
        for segment in reversed(segments):
            if segment.lower() != "track":
                candidate = segment.split("?")[0]
                break
        if not candidate:
            raise ValueError(f"Cannot parse track ID from URL: {cleaned}")
        cleaned = candidate

    if cleaned.startswith("spotify:"):
        prefix, _, candidate_id = cleaned.partition("spotify:track:")
        if candidate_id:
            return f"{TRACK_ID_PREFIX}{candidate_id}"
        cleaned = cleaned.rsplit(":", 1)[-1]

    if ":" in cleaned:
        cleaned = cleaned.split(":")[-1]

    if not cleaned:
        raise ValueError("Track ID was empty after parsing.")

    return f"{TRACK_ID_PREFIX}{cleaned}"


def _chunked(items: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def push_playlist_to_spotify(
    track_ids: Sequence[str],
    playlist_name: str,
    *,
    description: str | None = None,
    public: bool = False,
    scopes: Sequence[str] | None = None,
    client_id: str | None = None,
    client_secret: str | None = None,
    redirect_uri: str | None = None,
    cache_path: str | None = None,
) -> dict[str, str]:
    """
    Create a playlist in the authenticated Spotify account and populate it with tracks.

    Args:
        track_ids: Ordered collection of Spotify track identifiers (raw IDs, URLs, or URIs).
        playlist_name: Name for the Spotify playlist.
        description: Optional playlist description.
        public: Whether the playlist should be public.
        scopes: Optional iterable of scopes to request. Defaults to playlist modification scopes.
        client_id: Optional Spotify client ID (falls back to environment variables).
        client_secret: Optional Spotify client secret.
        redirect_uri: Optional redirect URI registered with Spotify.
        cache_path: Optional path for storing OAuth tokens.

    Returns:
        Dictionary containing playlist metadata, including ID, URL, name, and snapshot ID.

    Raises:
        SpotifyExportError: If authentication fails or playlist creation/addition fails.
    """
    normalized_ids: list[str] = []
    for track_id in track_ids:
        try:
            normalized_ids.append(_normalize_track_id(track_id))
        except ValueError as exc:
            logger.warning("Skipping invalid track ID '%s': %s", track_id, exc)

    if not normalized_ids:
        raise SpotifyExportError("No valid Spotify track IDs were provided.")

    scope_str = " ".join(scopes or DEFAULT_SCOPES)

    try:
        auth_manager = SpotifyOAuth(
            scope=scope_str,
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            cache_path=cache_path,
        )
        client: Spotify = spotipy.Spotify(auth_manager=auth_manager)
        current_user = client.current_user()
        user_id = current_user["id"]
    except Exception as exc:  # noqa: BLE001 - surface as SpotifyExportError
        raise SpotifyExportError(f"Failed to authenticate with Spotify: {exc}") from exc

    try:
        created_playlist = client.user_playlist_create(
            user=user_id,
            name=playlist_name,
            public=public,
            description=description or "",
        )
        playlist_id = created_playlist["id"]
        snapshot_id = created_playlist.get("snapshot_id", "")

        for batch in _chunked(normalized_ids, 100):
            response = client.playlist_add_items(playlist_id, list(batch))
            snapshot_id = response.get("snapshot_id", snapshot_id)

        playlist_url = created_playlist.get("external_urls", {}).get(
            "spotify", f"https://open.spotify.com/playlist/{playlist_id}"
        )
    except Exception as exc:  # noqa: BLE001
        raise SpotifyExportError(f"Failed to create or populate playlist: {exc}") from exc

    return {
        "playlist_id": playlist_id,
        "playlist_url": playlist_url,
        "snapshot_id": snapshot_id,
        "name": playlist_name,
        "public": public,
        "track_count": len(normalized_ids),
    }

