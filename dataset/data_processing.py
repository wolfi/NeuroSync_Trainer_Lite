import os
import numpy as np
import pandas as pd

from utils.audio.extraction.extract_features import extract_audio_features
from utils.video.mov_extraction import get_audio, find_files

COLUMNS_TO_DROP = ['Timecode', 'BlendshapeCount']

def load_data(root_dir, sr, processed_folders):
    
    examples = []
    all_facial_data = []
    all_audio_data = []

    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        if os.path.isdir(folder_path) and folder not in processed_folders:
            audio_features, facial_data = process_folder(folder_path, sr)
            if audio_features is not None and facial_data is not None:
                all_facial_data.append(facial_data)
                all_audio_data.append(audio_features)
                examples.append((audio_features, facial_data))
                processed_folders.add(folder)

    return examples

def scale_facial_data(facial_data, scale_factor=1.1):
    
    scaled_data = np.copy(facial_data)

    # Iterate through each value to apply scaling
    for i in range(scaled_data.shape[0]):
        for j in range(scaled_data.shape[1]):
            # Scale the value
            scaled_value = scaled_data[i, j] * scale_factor
            
            # Clip values to keep within the valid range [-1, 1] 
            scaled_data[i, j] = np.clip(scaled_value, -1, 1)

    return scaled_data


def process_folder(folder_path, sr, apply_smoothing=False, apply_over_scale=False):
    
    mov_path, mp4_path, wav_path, facial_csv_path, audio_features_csv_path, _ = find_files(folder_path)
    
    video_path = mov_path or mp4_path
    
    # Check if we have facial CSV and either video/audio paths or the audio features CSV file
    if facial_csv_path and (video_path or wav_path or os.path.exists(audio_features_csv_path)):
        
        # Only try to get audio path and frames if we have video or audio paths
        if video_path or wav_path:
            audio_path = get_audio(video_path, wav_path, folder_path)
        else:
            audio_path = None
        
        # Now check if we have an audio path or the audio features CSV
        if audio_path or os.path.exists(audio_features_csv_path):
            
            # Collect features using either the audio file or 'audio_features.csv'
            audio_features, facial_data = collect_features(audio_path if audio_path else _, audio_features_csv_path, facial_csv_path, sr)

            if apply_over_scale:
                facial_data = scale_facial_data(facial_data)

            facial_data[:, :61] *= 100  

            if apply_smoothing:
                facial_data = smooth_facial_data(facial_data)

            facial_data = zero_specified_columns(facial_data) # this is to remove blinks and neck as they are just noise - add them in later.
          #  facial_data = remove_specified_dimensions(facial_data)
                
            return audio_features, facial_data
    
    return None, None





def interpolate_slower(data):
    """
    Given data of shape (N, F), create an interpolated 'slower' version
    with 2N-1 frames via linear interpolation between consecutive rows.
    
    For frames i and i+1, we insert a new frame (data[i] + data[i+1]) / 2.
    Example:
       input:  [x0, x1, x2, x3]  (4 frames)
       output: [x0, (x0+x1)/2, x1, (x1+x2)/2, x2, (x2+x3)/2, x3]  (7 frames)
    """
    N, F = data.shape
    new_length = 2 * N - 1
    out = np.zeros((new_length, F))

    # Fill in the even indices with the original data
    # and the odd indices with the midpoint of consecutive frames
    for i in range(N - 1):
        out[2*i]   = data[i]
        out[2*i+1] = (data[i] + data[i+1]) / 2.0
    
    # Last frame
    out[-1] = data[-1]
    return out

def collect_features(audio_path, audio_features_csv_path, facial_csv_path, sr, include_fast=True, include_slow=False):
    """
    Loads (or extracts) audio features and facial data, aligns their lengths,
    and optionally augments the data by adding "fast" and/or "slower" versions.
    
    Args:
        audio_path (str): Path to the audio file.
        audio_features_csv_path (str): Path to CSV file storing audio features.
        facial_csv_path (str): Path to the facial data CSV.
        sr (int): Sample rate for audio processing.
        include_fast (bool): If True, include a downsampled (fast) version of the data.
        include_slow (bool): If True, include an interpolated (slower) version of the data.
    
    Returns:
        Tuple of (audio_features, facial_data) as NumPy arrays.
    """
    # Load or extract audio features
    if os.path.exists(audio_features_csv_path):
        print(f"Loading audio features from {audio_features_csv_path}")
        audio_features = pd.read_csv(audio_features_csv_path).values
    else:
        print(f"Extracting audio features from {audio_path}")
        audio_features, _ = extract_audio_features(audio_path, sr)
        if audio_features is not None:
            pd.DataFrame(audio_features).to_csv(audio_features_csv_path, index=False)
            print(f"Audio features saved to {audio_features_csv_path}")

    # Load and process the facial data
    facial_data = pd.read_csv(facial_csv_path).drop(columns=COLUMNS_TO_DROP).values

    # --- Matching lengths logic ---
    if audio_features is not None and facial_data is not None:
        len_audio = len(audio_features)
        len_facial = len(facial_data)

        if len_audio != len_facial:
            # Figure out which is longer and trim accordingly.
            if len_audio > len_facial:
                diff = len_audio - len_facial
                left_trim = diff // 2
                right_trim = diff - left_trim
                audio_features = audio_features[left_trim: len_audio - right_trim]
            else:
                diff = len_facial - len_audio
                left_trim = diff // 2
                right_trim = diff - left_trim
                facial_data = facial_data[left_trim: len_facial - right_trim]

    # Final trimming: ensure both have the same length.
    min_length = min(len(audio_features), len(facial_data))
    audio_features = audio_features[:min_length]
    facial_data = facial_data[:min_length]

    # Build lists of versions starting with the original.
    audio_versions = [audio_features]
    facial_versions = [facial_data]

    # Optionally add a fast version (downsampled by a factor of 2).
    if include_fast:
        facial_copy = facial_data.copy()
        facial_fast_smoothed = smooth_facial_data(facial_copy)
        audio_fast = audio_features[::2].copy()
        facial_fast = facial_fast_smoothed[::2].copy()
        audio_versions.append(audio_fast)
        facial_versions.append(facial_fast)

    # Optionally add a slower version (using linear interpolation).
    if include_slow:
        audio_slower = interpolate_slower(audio_features)
        facial_slower = interpolate_slower(facial_data)
        audio_versions.append(audio_slower)
        facial_versions.append(facial_slower)

    # Vertically stack the selected versions.
    audio_features = np.vstack(audio_versions)
    facial_data = np.vstack(facial_versions)

    return audio_features, facial_data



def smooth_facial_data(facial_data):
    smoothed_data = np.copy(facial_data)
    smoothed_data[1:] = (facial_data[:-1] + facial_data[1:]) / 2
    return smoothed_data



def remove_specified_dimensions(facial_data):
    columns_to_remove = [0, 1, 2 ,3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60]
    modified_data = np.delete(facial_data, columns_to_remove, axis=1)
    return modified_data


def zero_specified_columns(facial_data):
    columns_to_zero = [0, 1, 2 ,3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60]

    facial_data[:, columns_to_zero] = 0
    
    return facial_data

