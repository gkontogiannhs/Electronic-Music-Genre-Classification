import pandas as pd
import os
from utils import parse_audioset_csv, download_tracks


# -------- CONFIG --------
TRAIN_CSV_FILE_PATH = "data/audioset_balanced_train_segments.csv"
EVAL_CSV_FILE_PATH = "data/audioset_eval_segments.csv"
LABELS_FILE_PATH = "data/audioset_class_labels_indices.csv"
TRAIN_OUTPUT_DIR_PATH = "data/ELECTRONIC_MUSIC_V1/Train"
EVAL_OUTPUT_DIR_PATH = "data/ELECTRONIC_MUSIC_V1/Eval"

## Genres of Interest
### House music, Techno, Dubstep, Electro, Drum and bass, Electronica, Electronic dance music, Ambient music, Trance music
TARGET_LABELS = [
    "House music", "Techno", "Electronic music", "Electronica", "Trance music", "Electronic dance music", 
]

if __name__ == "__main__":
    # Load label names
    audioset_labels_df = pd.read_csv(LABELS_FILE_PATH)
    
    # Create a dictionary with keys being the encoded label name and value the actual label name
    code_to_label_map = dict(zip(audioset_labels_df["mid"], audioset_labels_df["display_name"]))

    # Create a same but reversed dictionary for later use
    label_to_code_map = {v: k for k, v in code_to_label_map.items()}

    # Translate to encoded labels to retrieve from the total dataset of only the desired labeled tracks
    target_label_ids = [label_to_code_map[genre] for genre in TARGET_LABELS]

    # Create a dataframe keep all needed metadata to downlaod and parase audio files from the youtube videos
    electronic_metadata_df_train, _ = parse_audioset_csv(TRAIN_CSV_FILE_PATH, target_label_ids)
    electronic_metadata_df_eval, _ = parse_audioset_csv(EVAL_CSV_FILE_PATH, target_label_ids)

    # Create directories Train/Eval for the downloaed clips
    os.makedirs(TRAIN_OUTPUT_DIR_PATH, exist_ok=True)
    os.makedirs(EVAL_OUTPUT_DIR_PATH, exist_ok=True)

    # Download the audio from each youtube vedeo and trip to start stop based on the annotations
    print("Downloading audio files...")
    download_tracks(electronic_metadata_df_train, EVAL_OUTPUT_DIR_PATH)
    download_tracks(electronic_metadata_df_eval, EVAL_OUTPUT_DIR_PATH)
    print("Download completed.")