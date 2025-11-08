#!/usr/bin/env python3
import argparse, json, random, os, re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set

from recommender import Recommender

RNG_SEED = 42

def _parse_track_id(track_obj: Dict[str, Any]) -> Optional[str]:
    for key in ("track_uri", "uri", "id", "track_id"):
        if key in track_obj and track_obj[key]:
            v = str(track_obj[key])
            if "spotify:track:" in v:
                return v.rsplit(":", 1)[-1].strip()
            return v.strip()
    return None

def _iter_mpd_playlists(mpd_dir: Path):
    """Yields (playlist_name, all_track_ids) for every playlist in MPD."""
    for p in sorted(mpd_dir.glob("mpd.slice.*.json")):
        with p.open("r", encoding="utf-8") as f:
            blob = json.load(f)
        for pl in blob.get("playlists", []):
            name = pl.get("name", "")
            tids: List[str] = []
            for tr in pl.get("tracks", []):
                tid = _parse_track_id(tr)
                if tid:
                    tids.append(tid)
            if tids:
                yield name, tids

def _safe_slug(name: str, fallback: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")
    return slug or fallback

def _find_full_overlap_playlists(
    rec: Recommender, mpd_dir: Path, min_size: int, n: int
) -> List[Tuple[str, List[str]]]:
    """
    Returns up to n playlists [(name, all_ids)] where ALL tracks exist in dataset
    and playlist length >= min_size.
    """
    dataset_ids: Set[str] = set(rec.id_to_idx.keys())
    found: List[Tuple[str, List[str]]] = []
    for name, all_ids in _iter_mpd_playlists(mpd_dir):
        if len(all_ids) >= min_size and all(t in dataset_ids for t in all_ids):
            found.append((name, all_ids))
            if len(found) >= n:
                break
    return found

def _artists_of(rec: Recommender, track_ids: List[str]) -> Set[str]:
    id_to_artists: Dict[str, Set[str]] = dict(
        zip(rec.df["track_id"].tolist(), rec.df["artists_set"].tolist())
    )
    out: Set[str] = set()
    for t in track_ids:
        out |= id_to_artists.get(t, set())
    return out

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def main():
    ap = argparse.ArgumentParser(description="Hide K tracks from FULL-overlap MPD playlists and test recommender recovery.")
    ap.add_argument("--mpd-dir", required=True, type=Path, help="Directory with mpd.slice.*.json files")
    ap.add_argument("--dataset", required=True, type=Path, help="Path to dataset.csv")
    ap.add_argument("--weights", default=None, type=Path, help="Optional path to feature_weights.json")
    ap.add_argument("--min-overlap", type=int, default=12, help="Minimum playlist size (full overlap required)")
    ap.add_argument("--k-hidden", type=int, default=5, choices=range(1, 51), help="Tracks to hide (and recs to request)")
    ap.add_argument("--pick-top", action="store_true", help="Pick the first K tracks to hide instead of random sampling")
    ap.add_argument("--num-playlists", type=int, default=3, help="How many playlists to find and evaluate")
    ap.add_argument("--save-dir", default="runs/full_overlap", help="Directory for outputs")
    args = ap.parse_args()

    rec = Recommender(
        dataset_file=str(args.dataset.resolve()),
        weights_file=str(args.weights.resolve()) if args.weights else "feature_weights.json"
    )

    playlists = _find_full_overlap_playlists(
        rec, args.mpd_dir.resolve(), min_size=args.min_overlap, n=args.num_playlists
    )
    if not playlists:
        raise RuntimeError(f"No FULL-overlap playlists (size >= {args.min_overlap}) found.")

    print(f"Found {len(playlists)} FULL-overlap playlist(s) (requested {args.num_playlists}).")

    save_root = Path(args.save_dir)
    _ensure_dir(save_root)
    summary: List[Dict[str, Any]] = []

    for i, (pl_name, all_ids) in enumerate(playlists, start=1):
        if len(all_ids) <= args.k_hidden:
            print(f"Skipping '{pl_name}' (size {len(all_ids)} <= k_hidden {args.k_hidden}).")
            continue

        # Choose hidden tracks
        ids_for_sampling = all_ids[:]  # copy
        if args.pick_top:
            hidden = ids_for_sampling[:args.k_hidden]
        else:
            rnd = random.Random(RNG_SEED + i)  # vary per playlist but deterministic
            rnd.shuffle(ids_for_sampling)
            hidden = ids_for_sampling[:args.k_hidden]

        seeds = [t for t in all_ids if t not in hidden]
        target_artists = _artists_of(rec, hidden)

        # Output dir per playlist
        pl_slug = f"{i:02d}_" + _safe_slug(pl_name, f"playlist_{i:02d}")
        out_dir = save_root / pl_slug
        _ensure_dir(out_dir)

        # Save artifacts
        deleted_payload = {
            "playlist_name": pl_name,
            "hidden_track_ids": hidden,
            "hidden_artists": sorted(target_artists),
            "seed_count": len(seeds),
        }
        (out_dir / "deleted_songs.json").write_text(json.dumps(deleted_payload, indent=2), encoding="utf-8")

        test_payload = {
            "playlist_name": pl_name,
            "k_hidden": args.k_hidden,
            "mpd_total_tracks": len(all_ids),
            "mpd_track_ids": all_ids,          # ALL tracks in the MPD playlist (full overlap)
            "dataset_overlap_track_ids": all_ids,  # same as above in FULL mode
            "input_seed_ids": seeds,           # Input to Recommender
            "deleted_ids": hidden,             # Held-out tracks
        }
        (out_dir / "test_playlist.json").write_text(json.dumps(test_payload, indent=2), encoding="utf-8")

        # Run recommender
        print(f"[{i}/{len(playlists)}] '{pl_name}': seeds={len(seeds)} hidden={len(hidden)} → requesting {args.k_hidden} recs")
        recs: List[str] = rec.get_recommendations(seeds, n_recommendations=args.k_hidden, target_artist=target_artists)

        # Evaluate
        ranks: Dict[str, Optional[int]] = {}
        hits = 0
        for t in hidden:
            try:
                r = recs.index(t) + 1
                ranks[t] = r
                hits += 1
            except ValueError:
                ranks[t] = None

        hit_at_k = 1.0 if hits == args.k_hidden else 0.0
        recall_at_k = hits / args.k_hidden
        mrr = sum(1.0 / r for r in ranks.values() if r is not None) / args.k_hidden

        # Save per-playlist results
        results_payload = {
            "playlist_name": pl_name,
            "k_hidden": args.k_hidden,
            "recommendations": recs,
            "ranks": ranks,
            "metrics": {
                f"Hit@{args.k_hidden}": hit_at_k,
                f"Recall@{args.k_hidden}": recall_at_k,
                "MRR": mrr,
            },
        }
        (out_dir / "results.json").write_text(json.dumps(results_payload, indent=2), encoding="utf-8")

        summary.append({
            "playlist_name": pl_name,
            "dir": str(out_dir),
            "size": len(all_ids),
            "k_hidden": args.k_hidden,
            "hits": hits,
            "recommendations": recs,
            f"Hit@{args.k_hidden}": hit_at_k,
            f"Recall@{args.k_hidden}": recall_at_k,
            "MRR": mrr,
        })

        # Console report
        print(f"  → Hidden recovered: {hits}/{args.k_hidden} | Hit@{args.k_hidden}={hit_at_k:.3f} "
              f"| Recall@{args.k_hidden}={recall_at_k:.3f} | MRR={mrr:.3f}")

    # Aggregate summary
    if summary:
        avg_hit = sum(s[f"Hit@{args.k_hidden}"] for s in summary) / len(summary)
        avg_recall = sum(s[f"Recall@{args.k_hidden}"] for s in summary) / len(summary)
        avg_mrr = sum(s["MRR"] for s in summary) / len(summary)
        aggregate = {
            "num_playlists": len(summary),
            "k_hidden": args.k_hidden,
            "averages": {
                f"Hit@{args.k_hidden}": avg_hit,
                f"Recall@{args.k_hidden}": avg_recall,
                "MRR": avg_mrr,
            },
            "playlists": summary,
        }
        (save_root / "summary.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
        print(f"\nSaved aggregate summary to {save_root / 'summary.json'}")
    else:
        print("No eligible playlists evaluated (all skipped).")

if __name__ == "__main__":
    main()
