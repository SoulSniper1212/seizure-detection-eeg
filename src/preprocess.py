import numpy as np

def segment_eeg(eeg_data, raw_objects, window_sec=5, sfreq=256):
    X = []
    segment_length = int(window_sec * sfreq)

    for i, raw in enumerate(raw_objects):
        data = raw.get_data()
        total_samples = data.shape[1]
        for start in range(0, total_samples - segment_length, segment_length):
            segment = data[:, start:start + segment_length]
            X.append(segment)
    
    X = np.array(X)
    return X
