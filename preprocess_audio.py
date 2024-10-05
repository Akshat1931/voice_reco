import librosa
import numpy as np

def preprocess_audio(file_path, n_mfcc=13):
    """
    Extract MFCC features from an audio file.
    
    :param file_path: Path to the .wav file
    :param n_mfcc: Number of MFCC features to extract (default: 13)
    :return: MFCC features as a numpy array
    """
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=16000)  # Sample rate as 16kHz
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        # Take the mean across time frames
        mfcc_mean = np.mean(mfcc.T, axis=0)
        return mfcc_mean
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None
