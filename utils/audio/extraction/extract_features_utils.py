import librosa
import numpy as np


def cepstral_mean_variance_normalization(mfcc):
    mean = np.mean(mfcc, axis=1, keepdims=True)
    std = np.std(mfcc, axis=1, keepdims=True)
    return (mfcc - mean) / (std + 1e-10)


def extract_mfcc_features(y, sr, frame_length, hop_length, num_mfcc=23):
    mfcc_features = extract_overlapping_mfcc(y, sr, num_mfcc, frame_length, hop_length)

    reduced_mfcc_features = reduce_features(mfcc_features)
    return reduced_mfcc_features.T, mfcc_features.shape[1]  # Transpose and return original frame count

def extract_overlapping_mfcc(chunk, sr, num_mfcc, frame_length, hop_length, include_deltas=True, include_cepstral=True):

    mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=num_mfcc, n_fft=frame_length, hop_length=hop_length)
    
    if include_cepstral:
        mfcc = cepstral_mean_variance_normalization(mfcc)

    if include_deltas:
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        combined_mfcc = np.vstack([mfcc, delta_mfcc, delta2_mfcc])  # Stack original MFCCs with deltas
        return combined_mfcc
    else:
        return mfcc


def reduce_features(features):
    num_frames = features.shape[1]
    paired_frames = features[:, :num_frames // 2 * 2].reshape(features.shape[0], -1, 2)
    reduced_frames = paired_frames.mean(axis=2)
    
    if num_frames % 2 == 1:
        last_frame = features[:, -1].reshape(-1, 1)
        reduced_final_features = np.hstack((reduced_frames, last_frame))
    else:
        reduced_final_features = reduced_frames
    
    return reduced_final_features


def smooth_features(features):
    smoothed_features = np.copy(features)
    for i in range(1, len(features)):
        smoothed_features[i] = (features[i - 1] + features[i]) / 2
    return smoothed_features

''' 

# If you know what this is then you should probably use this with mfcc, or without - don't use deltas for auto corr (this is wip) ;)



def extract_overlapping_autocorr(y, sr, frame_length, hop_length, num_autocorr_coeff=138,
                                 pad_signal=True, padding_mode="reflect", trim_padded=True):
    """
    Extract overlapping autocorrelation features from an audio signal.
    
    Parameters:
        y (np.ndarray): Input 1-D audio signal.
        sr (int): Sample rate.
        frame_length (int): Length of each frame in samples.
        hop_length (int): Number of samples between the start of consecutive frames.
        num_autocorr_coeff (int): Number of autocorrelation coefficients to extract.
        pad_signal (bool): If True, pad the signal with half the frame length on each side.
                           This ensures that even edge frames are full-length.
        padding_mode (str): Padding mode passed to np.pad (e.g. "reflect", "constant", etc.).
        trim_padded (bool): If True, remove frames that include padded data from the output.
        
    Returns:
        np.ndarray: Autocorrelation features with shape (num_autocorr_coeff, num_valid_frames),
                    where only frames computed entirely on the original signal are included.
    
    Explanation:
        Padding is necessary to ensure that every frame is the correct length for analysis.
        However, frames at the edges will include padded data and might cause artifacts or skipping.
        By trimming out those frames (using the trim_padded flag), we ensure that only frames
        computed from actual data are returned.
    """
    # Pad the signal if desired.
    if pad_signal:
        pad = frame_length // 2
        y_padded = np.pad(y, pad_width=pad, mode=padding_mode)
    else:
        y_padded = y

    # Frame the (padded) signal.
    frames = librosa.util.frame(y_padded, frame_length=frame_length, hop_length=hop_length)
    
    # Optionally trim frames that come from padded regions.
    if pad_signal and trim_padded:
        num_frames = frames.shape[1]
        # The starting index of each frame in the padded signal.
        start_indices = np.arange(num_frames) * hop_length
        # A frame is fully valid if it lies entirely within the region [pad, len(y)+pad).
        valid_idx = np.where((start_indices >= pad) & (start_indices + frame_length <= len(y) + pad))[0]
        frames = frames[:, valid_idx]
    
    # Remove DC offset per frame.
    frames = frames - np.mean(frames, axis=0, keepdims=True)
    
    # Apply Hann window.
    hann_window = np.hanning(frame_length)
    windowed_frames = frames * hann_window[:, np.newaxis]
    
    autocorr_list = []
    for frame in windowed_frames.T:
        # Compute full autocorrelation for this frame.
        full_corr = np.correlate(frame, frame, mode='full')
        mid = frame_length - 1  # Zero-lag index.
        # Extract the desired number of coefficients (starting from zero-lag).
        wanted = full_corr[mid: mid + num_autocorr_coeff]
        # Normalize by the zero-lag (energy) if nonzero.
        if wanted[0] != 0:
            wanted = wanted / wanted[0]
        autocorr_list.append(wanted)
    
    # Convert list to array and transpose so that shape is (num_autocorr_coeff, num_valid_frames)
    autocorr_features = np.array(autocorr_list).T
    return autocorr_features




def extract_autocorrelation_features(
    y, sr, frame_length, hop_length, include_deltas=False
):
    """
    Extract autocorrelation features, optionally with deltas/delta-deltas,
    then align with the MFCC frame count, reduce, and handle first/last frames.
    """
    autocorr_features = extract_overlapping_autocorr(
        y, sr, frame_length, hop_length
    )
    
    if include_deltas:
        autocorr_features = compute_autocorr_with_deltas(autocorr_features)

    autocorr_features_reduced = reduce_features(autocorr_features)

    return autocorr_features_reduced.T


def compute_autocorr_with_deltas(autocorr_base):
    delta_ac = librosa.feature.delta(autocorr_base)
    delta2_ac = librosa.feature.delta(autocorr_base, order=2)
    combined_autocorr = np.vstack([autocorr_base, delta_ac, delta2_ac])
    return combined_autocorr

'''
