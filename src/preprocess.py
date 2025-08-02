import numpy as np

def segment_eeg(eeg_data, raw_objects, window_sec=5, sfreq=256):
    """
    Segment EEG data into windows
    
    Args:
        eeg_data: List of EEG data arrays
        raw_objects: List of MNE Raw objects
        window_sec: Window size in seconds
        sfreq: Sampling frequency (will be overridden by raw object if available)
    
    Returns:
        X: Array of EEG segments (n_segments, channels, time_points)
    """
    X = []
    
    for i, raw in enumerate(raw_objects):
        # Use sampling frequency from raw object
        sfreq = raw.info['sfreq']
        segment_length = int(window_sec * sfreq)
        
        # Get data from raw object
        data = raw.get_data()
        total_samples = data.shape[1]
        
        # Create segments
        for start in range(0, total_samples - segment_length, segment_length):
            segment = data[:, start:start + segment_length]
            X.append(segment)
    
    if not X:
        raise ValueError("No segments were created. Check if data files are valid.")
    
    X = np.array(X)
    
    # Ensure proper shape: (n_segments, channels, time_points)
    if len(X.shape) != 3:
        raise ValueError(f"Expected 3D array, got shape {X.shape}")
    
    print(f"Created {X.shape[0]} segments with shape {X.shape[1:]} (channels, time_points)")
    return X
