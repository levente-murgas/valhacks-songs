from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def _safe_path(base: Path, maybe_relative: str) -> Path:
    candidate = Path(maybe_relative)
    if candidate.is_absolute():
        return candidate
    return base / maybe_relative.lstrip("/")


def _parse_track_id(track_obj: object) -> Optional[str]:
    """
    Handles the common Spotify playlist formats (Million Playlist Dataset, custom JSON, etc.).
    """
    if isinstance(track_obj, str):
        tid = track_obj.strip()
        if tid.startswith("spotify:track:"):
            return tid.rsplit(":", 1)[-1]
        return tid or None
    if isinstance(track_obj, dict):
        for key in ("track_uri", "uri", "id", "track_id"):
            if key in track_obj:
                val = str(track_obj[key]).strip()
                if not val:
                    continue
                if val.startswith("spotify:track:"):
                    return val.rsplit(":", 1)[-1]
                return val
    return None


class Recommender:
    """
    Playlist-aware recommender that learns track co-occurrence statistics from
    human curated Spotify playlists (or genre/album surrogates if curated data is missing)
    and blends them with content-based similarity + popularity fallback.
    """

    BASE_NUMERIC_FEATURES: Sequence[str] = (
        "danceability",
        "energy",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "duration_ms",
        "popularity",
        "key",
        "mode",
        "time_signature",
    )

    PLAYLIST_MIN_LENGTH = 6
    PLAYLIST_MAX_LENGTH = 400
    PLAYLIST_WINDOW = 25
    MAX_NEIGHBORS = 800
    RECENCY_WINDOW = 80
    RECENCY_DECAY = 0.78
    PLAYLIST_WEIGHT = 0.63
    CONTENT_WEIGHT = 0.27
    POP_WEIGHT = 0.10
    ARTIST_HINT_BONUS = 0.18
    CANDIDATE_POOL_MULTIPLIER = 6
    SURROGATE_PLAYLIST_LEN = 80

    def __init__(
        self,
        dataset_file: str = "dataset.csv",
        curated_source: Optional[str] = None,
        feature_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        data_root = Path("src/data").resolve()
        dataset_path = _safe_path(data_root, dataset_file)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        self.df = pd.read_csv(dataset_path)
        if "Unnamed: 0" in self.df.columns:
            self.df = self.df.drop(columns=["Unnamed: 0"])
        self.df = self.df.dropna(subset=["track_id"]).reset_index(drop=True)
        self.df["artists_set"] = (
            self.df["artists"]
            .fillna("")
            .apply(lambda s: {a.strip() for a in str(s).split(";") if a.strip()})
        )
        self.artist_sets: List[Set[str]] = self.df["artists_set"].tolist()
        self.artist_sets_lower: List[Set[str]] = [
            {a.lower() for a in aset} for aset in self.artist_sets
        ]

        numeric_features = [col for col in self.BASE_NUMERIC_FEATURES if col in self.df.columns]
        if not numeric_features:
            raise ValueError("No usable numeric features found in dataset.")

        self.numeric_features = numeric_features
        self._build_feature_space(feature_weights)

        self.track_ids = self.df["track_id"].to_numpy(dtype=object)
        self.id_to_idx: Dict[str, int] = {tid: i for i, tid in enumerate(self.track_ids)}
        self.popularity_norm = self._build_popularity_vector()
        self.popularity_sorted_idx = np.argsort(-self.popularity_norm, kind="mergesort")

        self.curated_source = curated_source or self._guess_curated_source()
        playlists = self._load_curated_playlists(self.curated_source)
        if not playlists:
            playlists = self._build_surrogate_playlists()
        (
            self.playlist_neighbors,
            self.playlist_freq,
            self.playlist_norm_factor,
            self.playlist_training_size,
        ) = self._build_playlist_graph(playlists)

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------
    def _build_feature_space(self, feature_weights: Optional[Dict[str, float]]) -> None:
        matrix = self.df[self.numeric_features].copy()
        for col in self.numeric_features:
            matrix[col] = pd.to_numeric(matrix[col], errors="coerce")
        med = matrix.median(axis=0, skipna=True)
        matrix = matrix.fillna(med)

        scaler = StandardScaler(with_mean=True, with_std=True)
        z = scaler.fit_transform(matrix.to_numpy(dtype=np.float32)).astype(np.float32)

        weights = np.ones(len(self.numeric_features), dtype=np.float32)
        if feature_weights:
            for i, name in enumerate(self.numeric_features):
                if name in feature_weights:
                    weights[i] = float(feature_weights[name])

        weighted = z * weights
        norms = np.linalg.norm(weighted, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.feature_matrix = weighted / norms
        self.feature_weights = weights
        self.feature_scaler = scaler

    def _build_popularity_vector(self) -> np.ndarray:
        if "popularity" not in self.df.columns:
            return np.ones(len(self.df), dtype=np.float32) * 0.5
        pop = pd.to_numeric(self.df["popularity"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        pop = np.clip(pop, 0.0, 100.0)
        return (pop + 1.0) / 101.0

    def _guess_curated_source(self) -> Optional[str]:
        candidates = [
            "src/data/human_curated_playlists.json",
            "src/data/human_curated_playlists.jsonl",
            "src/data/human_curated_playlists",
            "src/data/mpd",
        ]
        for candidate in candidates:
            path = Path(candidate)
            if path.exists():
                return str(path)
        return None

    # ------------------------------------------------------------------
    # Playlist ingestion & graph building
    # ------------------------------------------------------------------
    def _load_curated_playlists(self, source: Optional[str]) -> List[List[str]]:
        if not source:
            return []
        path = Path(source)
        if not path.exists():
            return []
        sequences: List[List[str]] = []

        if path.is_dir():
            for suffix in ("*.jsonl", "*.json"):
                for file in sorted(path.glob(suffix)):
                    sequences.extend(self._parse_playlist_file(file))
            return sequences
        return self._parse_playlist_file(path)

    def _parse_playlist_file(self, file_path: Path) -> List[List[str]]:
        sequences: List[List[str]] = []
        try:
            if file_path.suffix.lower() == ".jsonl":
                with file_path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        sequences.extend(self._extract_sequences(json.loads(line)))
            else:
                with file_path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                sequences.extend(self._extract_sequences(payload))
        except (OSError, json.JSONDecodeError):
            return []
        return sequences

    def _extract_sequences(self, payload: object) -> List[List[str]]:
        sequences: List[List[str]] = []
        if isinstance(payload, list):
            for item in payload:
                sequences.extend(self._extract_sequences(item))
            return sequences

        if isinstance(payload, dict):
            if "playlists" in payload and isinstance(payload["playlists"], list):
                for playlist in payload["playlists"]:
                    seq = self._extract_single_playlist(playlist)
                    if seq:
                        sequences.append(seq)
            else:
                seq = self._extract_single_playlist(payload)
                if seq:
                    sequences.append(seq)
        return sequences

    def _extract_single_playlist(self, playlist_obj: object) -> Optional[List[str]]:
        if isinstance(playlist_obj, dict):
            track_candidates: Iterable = ()
            if "track_ids" in playlist_obj:
                track_candidates = playlist_obj["track_ids"]
            elif "tracks" in playlist_obj:
                track_candidates = playlist_obj["tracks"]
            else:
                track_candidates = playlist_obj.values()
            seq = [tid for tid in map(_parse_track_id, track_candidates) if tid]
        elif isinstance(playlist_obj, list):
            seq = [tid for tid in map(_parse_track_id, playlist_obj) if tid]
        else:
            return None
        if len(seq) < self.PLAYLIST_MIN_LENGTH:
            return None
        return seq[: self.PLAYLIST_MAX_LENGTH]

    def _build_surrogate_playlists(self) -> List[List[str]]:
        """
        When no curated data is provided we synthesize playlists by grouping tracks
        by genre/album popularity to keep the pipeline functional.
        """
        playlists: List[List[str]] = []

        if "track_genre" in self.df.columns:
            grouped = self.df.groupby("track_genre")
            for _, group in grouped:
                ids = (
                    group.sort_values("popularity", ascending=False, na_position="last")
                    ["track_id"]
                    .tolist()
                )
                if len(ids) >= self.PLAYLIST_MIN_LENGTH:
                    playlists.append(ids[: self.SURROGATE_PLAYLIST_LEN])

        if "album_name" in self.df.columns:
            album_grouped = self.df.groupby("album_name")
            for _, group in album_grouped:
                if len(group) < self.PLAYLIST_MIN_LENGTH:
                    continue
                ids = (
                    group.sort_values("popularity", ascending=False, na_position="last")["track_id"].tolist()
                )
                playlists.append(ids[: self.PLAYLIST_MAX_LENGTH])

        return playlists

    def _build_playlist_graph(
        self, playlists: Iterable[List[str]]
    ) -> Tuple[Dict[int, Tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray, int]:
        num_tracks = len(self.df)
        freq = np.zeros(num_tracks, dtype=np.float32)
        co_counts: Dict[int, Dict[int, float]] = defaultdict(dict)
        playlist_counter = 0

        for seq in playlists:
            deduped: List[int] = []
            seen_local: Set[int] = set()
            for tid in seq:
                idx = self.id_to_idx.get(tid)
                if idx is None or idx in seen_local:
                    continue
                deduped.append(idx)
                seen_local.add(idx)
            if len(deduped) < self.PLAYLIST_MIN_LENGTH:
                continue

            playlist_counter += 1
            for idx in deduped:
                freq[idx] += 1.0

            length = len(deduped)
            for pos, idx_i in enumerate(deduped):
                left_start = max(0, pos - self.PLAYLIST_WINDOW)
                right_end = min(length, pos + self.PLAYLIST_WINDOW + 1)

                for rel in range(left_start, pos):
                    idx_j = deduped[rel]
                    distance = pos - rel
                    weight = 1.0 / (1.0 + distance)
                    prev = co_counts[idx_i].get(idx_j, 0.0)
                    co_counts[idx_i][idx_j] = prev + weight

                for rel in range(pos + 1, right_end):
                    idx_j = deduped[rel]
                    distance = rel - pos
                    weight = 1.0 / (1.0 + distance)
                    prev = co_counts[idx_i].get(idx_j, 0.0)
                    co_counts[idx_i][idx_j] = prev + weight

        freq_safe = np.where(freq > 0, freq, 1.0)
        freq_norm = 1.0 / np.sqrt(freq_safe)
        neighbor_map: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        for idx, neigh_dict in co_counts.items():
            if not neigh_dict:
                continue
            items = sorted(neigh_dict.items(), key=lambda kv: kv[1], reverse=True)[: self.MAX_NEIGHBORS]
            neighbor_indices = np.array([nid for nid, _ in items], dtype=np.int32)
            neighbor_scores = np.array([val for _, val in items], dtype=np.float32)
            neighbor_map[idx] = (neighbor_indices, neighbor_scores)

        return neighbor_map, freq, freq_norm.astype(np.float32), playlist_counter

    # ------------------------------------------------------------------
    # Recommendation helpers
    # ------------------------------------------------------------------
    def _recency_weights(self, length: int) -> np.ndarray:
        if length <= 0:
            return np.empty(0, dtype=np.float32)
        positions = np.arange(length, dtype=np.float32)
        exponents = length - 1 - positions
        weights = np.power(self.RECENCY_DECAY, exponents)
        total = float(weights.sum())
        if total == 0.0:
            return np.full(length, 1.0 / length, dtype=np.float32)
        return weights / total

    def _playlist_signal(
        self, seed_indices: np.ndarray, recency_weights: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if not self.playlist_neighbors:
            return None
        scores = np.zeros(len(self.df), dtype=np.float32)
        coverage = False
        for weight, seed_idx in zip(recency_weights, seed_indices):
            neighbors = self.playlist_neighbors.get(int(seed_idx))
            if neighbors is None:
                continue
            neigh_idx, neigh_scores = neighbors
            if neigh_idx.size == 0:
                continue
            coverage = True
            normed = neigh_scores * self.playlist_norm_factor[seed_idx]
            normed = normed * self.playlist_norm_factor[neigh_idx]
            np.add.at(scores, neigh_idx, weight * normed)
        if not coverage:
            return None
        scores[seed_indices] = 0.0
        nz = np.flatnonzero(scores > 0.0)
        return scores, nz

    def _content_scores(self, seed_indices: np.ndarray, recency_weights: np.ndarray) -> np.ndarray:
        centroid = np.average(
            self.feature_matrix[seed_indices], axis=0, weights=recency_weights
        ).astype(np.float32)
        norm = np.linalg.norm(centroid)
        if norm == 0.0:
            return np.zeros(len(self.df), dtype=np.float32)
        centroid /= norm
        scores = (self.feature_matrix @ centroid).astype(np.float32)
        scores[seed_indices] = -np.inf
        return scores

    def _top_indices(self, scores: np.ndarray, limit: int) -> np.ndarray:
        limit = int(max(0, min(limit, scores.size)))
        if limit == 0:
            return np.empty(0, dtype=np.int32)
        finite_mask = np.isfinite(scores)
        if not finite_mask.any():
            return np.empty(0, dtype=np.int32)
        finite_indices = np.flatnonzero(finite_mask)
        limit = min(limit, finite_indices.size)
        part = np.argpartition(-scores[finite_indices], limit - 1)[:limit]
        idx = finite_indices[part]
        idx = idx[np.argsort(-scores[idx], kind="mergesort")]
        return idx.astype(np.int32)

    def _popularity_fallback(self, seen: Set[str], n: int, target_artist: Set[str]) -> List[str]:
        target_lower = {a.lower() for a in target_artist}
        prioritized: List[str] = []
        general: List[str] = []
        for idx in self.popularity_sorted_idx:
            tid = self.track_ids[idx]
            if tid in seen:
                continue
            if target_lower and self.artist_sets_lower[idx] & target_lower:
                prioritized.append(tid)
            else:
                general.append(tid)
            if len(prioritized) >= n:
                break
            if len(prioritized) + len(general) >= n * 3:
                break
        ordered = prioritized + general
        return ordered[:n]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_recommendations(
        self,
        input_track_ids: List[str],
        n_recommendations: int,
        target_artist: Set[str],
    ) -> List[str]:
        n_recommendations = max(0, int(n_recommendations))
        if n_recommendations == 0:
            return []

        seen = set(input_track_ids)
        valid_indices = [self.id_to_idx[tid] for tid in input_track_ids if tid in self.id_to_idx]
        if not valid_indices:
            return self._popularity_fallback(seen, n_recommendations, target_artist)

        seed_indices = np.array(valid_indices[-self.RECENCY_WINDOW :], dtype=np.int32)
        recency_weights = self._recency_weights(seed_indices.size)

        playlist_signal = self._playlist_signal(seed_indices, recency_weights)
        if playlist_signal:
            playlist_scores, playlist_candidates = playlist_signal
        else:
            playlist_scores = None
            playlist_candidates = np.empty(0, dtype=np.int32)

        content_scores = self._content_scores(seed_indices, recency_weights)
        candidate_set: Set[int] = set(int(idx) for idx in playlist_candidates.tolist())
        candidate_set -= set(seed_indices.tolist())

        target_pool = max(n_recommendations * self.CANDIDATE_POOL_MULTIPLIER, n_recommendations * 2)
        if len(candidate_set) < target_pool:
            extra = target_pool - len(candidate_set)
            candidate_set.update(int(idx) for idx in self._top_indices(content_scores, extra * 2))
        if len(candidate_set) < target_pool:
            needed = target_pool - len(candidate_set)
            candidate_set.update(int(idx) for idx in self.popularity_sorted_idx[: needed * 2])

        target_lower = {a.lower() for a in target_artist}
        finals: List[Tuple[int, float]] = []
        for idx in candidate_set:
            if idx in seed_indices:
                continue
            score = 0.0
            if playlist_scores is not None:
                ps = playlist_scores[idx]
                if ps > 0:
                    score += self.PLAYLIST_WEIGHT * ps
            cs = content_scores[idx]
            if np.isfinite(cs) and cs > -np.inf:
                score += self.CONTENT_WEIGHT * max(cs, 0.0)
            score += self.POP_WEIGHT * self.popularity_norm[idx]
            if target_lower and self.artist_sets_lower[idx] & target_lower:
                score += self.ARTIST_HINT_BONUS
            if score > 0:
                finals.append((idx, score))

        finals.sort(key=lambda kv: kv[1], reverse=True)
        rec_ids: List[str] = []
        for idx, _ in finals:
            tid = self.track_ids[idx]
            if tid in seen:
                continue
            rec_ids.append(tid)
            seen.add(tid)
            if len(rec_ids) >= n_recommendations:
                break

        if len(rec_ids) < n_recommendations:
            rec_ids.extend(
                self._popularity_fallback(seen, n_recommendations - len(rec_ids), target_artist)
            )
        return rec_ids[:n_recommendations]


if __name__ == "__main__":
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
