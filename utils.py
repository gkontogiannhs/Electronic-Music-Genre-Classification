import pandas as pd
import subprocess
import os
import numpy as np
import librosa
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, f1_score


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


def extract_features(
    filepath,
    sr=16000,
    duration=10,
    segment_duration=5,
    n_fft=2048,
    hop_length=512,
    n_mfcc=13,
    selected_features=None  # dictionary of feature flags
):
    """
    Extracts selected audio features from a file using librosa.

    Parameters:
        filepath (str): Path to the audio file.
        sr (int): Sampling rate.
        duration (int): Total duration of the audio file to load (in seconds).
        segment_duration (int): Duration of each segment (in seconds).
        n_fft (int): FFT window size.
        hop_length (int): Hop length for STFT-based features.
        n_mfcc (int): Number of MFCCs to compute.
        selected_features (dict): Dictionary specifying which features to extract.

    Returns:
        np.ndarray: 3D array (segments, time_steps, features)
    """
    if selected_features is None:
        selected_features = {
            "zcr": True,
            "centroid": True,
            "onset_strength": True,
            "mfcc": True,
            "chroma": True,
            "spectral_bandwidth": True,
            "spectral_rolloff": True,
            "rms": True,
            "tempogram": False
        }

    try:
        y, _ = librosa.load(filepath, sr=sr, duration=duration)
        samples_per_segment = int(sr * segment_duration)
        num_segments = duration // segment_duration
        segments_feats = []

        for s in range(num_segments):
            start = s * samples_per_segment
            end = start + samples_per_segment
            segment = y[start:end]

            if len(segment) < samples_per_segment:
                print("Incomplete segment detected, skipping...")
                continue

            features_list = []

            if selected_features.get("zcr"):
                zcr = librosa.feature.zero_crossing_rate(segment, frame_length=n_fft, hop_length=hop_length)
                features_list.append(zcr)

            if selected_features.get("centroid"):
                centroid = librosa.feature.spectral_centroid(y=segment, sr=sr, n_fft=n_fft, hop_length=hop_length)
                features_list.append(centroid)

            if selected_features.get("onset_strength"):
                onset_env = librosa.onset.onset_strength(y=segment, sr=sr, hop_length=hop_length)[np.newaxis, :]
                features_list.append(onset_env)

            if selected_features.get("mfcc"):
                mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc)
                features_list.append(mfcc)

            if selected_features.get("chroma"):
                chroma = librosa.feature.chroma_stft(y=segment, sr=sr, n_fft=n_fft, hop_length=hop_length)
                features_list.append(chroma)

            if selected_features.get("spectral_bandwidth"):
                bandwidth = librosa.feature.spectral_bandwidth(y=segment, sr=sr, n_fft=n_fft, hop_length=hop_length)
                features_list.append(bandwidth)

            if selected_features.get("spectral_rolloff"):
                rolloff = librosa.feature.spectral_rolloff(y=segment, sr=sr, n_fft=n_fft, hop_length=hop_length)
                features_list.append(rolloff)

            if selected_features.get("rms"):
                rms = librosa.feature.rms(y=segment, frame_length=n_fft, hop_length=hop_length)
                features_list.append(rms)

            if selected_features.get("tempogram"):
                onset_env = librosa.onset.onset_strength(y=segment, sr=sr, hop_length=hop_length)
                tempogram = librosa.feature.fourier_tempogram(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
                tempogram = tempogram[:, :min(tempogram.shape[1], mfcc.shape[1] if selected_features.get("mfcc") else onset_env.shape[0])]
                features_list.append(np.abs(tempogram))

            # Stack features along feature axis and transpose to shape (time_steps, features)
            min_frames = min(f.shape[1] for f in features_list)
            features_stack = np.vstack([f[:, :min_frames] for f in features_list])
            features_stack = features_stack.T
            segments_feats.append(features_stack)
        
        return np.array(segments_feats)  # shape: (num_segments, time_steps, features)

    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return []
    

def get_feature_names(feature_flags, n_mfcc=13, n_tempo=193, n_chroma=12):
    feature_names = []

    if feature_flags.get("zcr"):
        feature_names.append("ZCR")

    if feature_flags.get("centroid"):
        feature_names.append("Centroid")

    if feature_flags.get("onset_strength"):
        feature_names.append("Onset")

    if feature_flags.get("mfcc"):
        feature_names.extend([f"MFCC_{i}" for i in range(n_mfcc)])

    if feature_flags.get("chroma"):
        feature_names.extend([f"Chroma_{i}" for i in range(n_chroma)])

    if feature_flags.get("spectral_bandwidth"):
        feature_names.append("Bandwidth")

    if feature_flags.get("spectral_rolloff"):
        feature_names.append("Rolloff")

    if feature_flags.get("rms"):
        feature_names.append("RMS")

    if feature_flags.get("tempogram"):
        feature_names.extend([f"Tempogram_{i}" for i in range(n_tempo)])

    return feature_names


def report_pca_variance(X, threshold=0.80):
    pca = PCA()
    pca.fit(X)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cum_var >= threshold) + 1
    print(f"{threshold*100:.0f}% variance retained with {n_components} components.")
    return n_components, cum_var


def train_svm_pipelines(X, y, test_size=0.2, random_state=42):
    scalers = {
        "MinMax": MinMaxScaler(),
        "Standard": StandardScaler(),
        "Robust": RobustScaler()
    }

    pca_settings = {
        "no_pca": None,
        "pca_2d": 2,
        "pca_3d": 3,
        "pca_95var": 0.95
    }

    param_grid = {
        "svc__C": [0.1, 1, 10],
        "svc__gamma": ['scale', 'auto']
    }

    results = {}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=test_size, random_state=random_state
    )

    for scaler_name, scaler in scalers.items():
        print(f"\n=== {scaler_name} Scaler ===")
        results[scaler_name] = {}

        for pca_label, pca_val in pca_settings.items():
            steps = [("scaler", scaler)]

            if pca_val is not None:
                steps.append(("pca", PCA(n_components=pca_val)))
            steps.append(("svc", SVC(kernel="rbf", random_state=random_state)))

            pipeline = Pipeline(steps)
            grid = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
            grid.fit(X_train, y_train)

            y_pred = grid.predict(X_test)
            f1 = f1_score(y_test, y_pred, average="weighted")

            print(f"{pca_label}: F1-score = {f1:.3f} | Best Params = {grid.best_params_}")
            results[scaler_name][pca_label] = {
                "f1-score": f1,
                "best_params": grid.best_params_
            }

    return results