https://www.kaggle.com/datasets/himanshuwagh/spotify-million


python3 mpd_eval.py \
  --mpd-dir /home/tambu/Downloads/archive/data \
  --dataset ./data/dataset.csv \
  --weights ./data/feature_weights.json \
  --min-overlap 5 \
  --k-hidden 2 \
  --num-playlists 4 \
  --save-dir runs/full_overlap_3
