import pandas as pd
import os
import re
import shutil


TRAIN_DIR_PATH = "data/ELECTRONIC_MUSIC_V1/Train"
EVAL_DIR_PATH = "data/ELECTRONIC_MUSIC_V1/Eval"

# Output directory for organized dataset
OUTPUT_DIR = "data/ELECTRONIC_MUSIC_V2/"

## Genres of Interest to group by or keep
GENRES = ["Techno", "House", "Trance"]

if __name__ == "__main__":

    # Create file structure for the dataset
    for genre in GENRES:
        os.makedirs(os.path.join(OUTPUT_DIR, genre), exist_ok=True)

    # Regex to match categories
    category_patterns = {
        "techno": re.compile(r"\btechno\b", re.IGNORECASE),
        "house": re.compile(r"\bhouse music\b", re.IGNORECASE),
        "trance": re.compile(r"\btrance music\b", re.IGNORECASE)
    }

    # Regex to extract the labels part from filename
    label_extract_pattern = re.compile(r"^(.*?)(?:_[^_]+){2}\.wav$")

    # Process each file
    for input_dir in [TRAIN_DIR_PATH, EVAL_DIR_PATH]:

        # Process each file
        for fname in os.listdir(input_dir):
            if not fname.endswith(".wav"):
                continue

            match = label_extract_pattern.match(fname)
            if not match:
                print(f"Skipped (unmatched pattern): {fname}")
                continue

            label_str = match.group(1)
            label_list = [label.strip() for label in label_str.split(",")]

            matched_categories = set()
            for cat, pattern in category_patterns.items():
                if any(pattern.search(label) for label in label_list):
                    matched_categories.add(cat)

            # Each audio file must belong to only one of the three genres to keep as a seperate class
            if len(matched_categories) == 1:
                category = matched_categories.pop()
                src = os.path.join(input_dir, fname)
                dst = os.path.join(OUTPUT_DIR, category, fname)
                shutil.copy2(src, dst)
                print(f"Moved to {category}: {fname}")
            else:
                print(f"Skipped (ambiguous or irrelevant): {fname}")