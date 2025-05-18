import pandas as pd
import subprocess
import os

def parse_audioset_csv(file_path, target_label_ids=None):
    """
    Parses an AudioSet CSV file and filters rows by target label IDs (if provided).
    
    Args:
        file_path (str): Path to the segments CSV file.
        target_label_ids (list[str], optional): List of label IDs (in 'mid' format) to keep.
    
    Returns:
        pd.DataFrame: A filtered DataFrame with columns: YTID, start_seconds, end_seconds, positive_labels
    """
    rows = []
    actual_rows = 0

    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue  # Skip header/comments
            parts = line.strip().split(",", 3)  # Split only at first 3 commas
            actual_rows += 1
            
            if len(parts) == 4:
                try:
                    ytid = parts[0].strip()
                    start = float(parts[1].strip())
                    end = float(parts[2].strip())
                    labels_str = parts[3].strip().replace('"', '')
                    labels_ids = labels_str.split(",")

                    # Filter: include only if any of the target_label_ids is present
                    if target_label_ids:
                        if not any(lbl in target_label_ids for lbl in labels_ids):
                            continue

                    rows.append({
                        "YTID": ytid,
                        "start_seconds": start,
                        "end_seconds": end,
                        "encoded_labels": labels_str
                    })
                except ValueError:
                    continue  # Skip malformed lines

    return pd.DataFrame(rows), actual_rows


def download_tracks(df, output_dir):
    def _download_from_yt(ytid, start):
        print(f"Downloading {ytid}...")
        subprocess.run(["yt-dlp", f"https://www.youtube.com/watch?v={ytid}", "-f", "bestaudio", "-o", f"{ytid}.webm"], check=True)

        print(f"Trimming {ytid}...")
        subprocess.run([
                "ffmpeg", "-ss", str(start), "-i", f"{ytid}.webm", "-t", "10", "-ar", "16000", "-ac", "1", "-y", out_file
            ], check=True)
        
    for _, row in df.iterrows():
        ytid = row['YTID']
        start = float(row['start_seconds'])
        label_names = row["label_names_list"]

        out_file = os.path.join(output_dir, f"{', '.join(label_names)}_{ytid}_{int(start)}.wav")

        try:
            _download_from_yt(ytid, start)

            os.remove(f"{ytid}.webm")

            print(f"Saved: {out_file} | Labels: {label_names}")

        except subprocess.CalledProcessError as e:
            print(f"Error with {ytid}: {e}")
            continue