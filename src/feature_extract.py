import numpy as np
import scipy.signal as signal
from scipy.stats import entropy, kurtosis, skew
from sklearn.preprocessing import StandardScaler
import pywt

def extract_time_domain_features(eeg_segment):
    """
    Extract time domain features from EEG segment
    
    Args:
        eeg_segment: EEG data segment (channels, time_points)
    
    Returns:
        Dictionary of time domain features
    """
    features = {}
    
    for i in range(eeg_segment.shape[0]):
        channel_data = eeg_segment[i, :]
        
        # Basic statistical features
        features[f'mean_ch{i}'] = np.mean(channel_data)
        features[f'std_ch{i}'] = np.std(channel_data)
        features[f'var_ch{i}'] = np.var(channel_data)
        features[f'skew_ch{i}'] = skew(channel_data)
        features[f'kurtosis_ch{i}'] = kurtosis(channel_data)
        features[f'min_ch{i}'] = np.min(channel_data)
        features[f'max_ch{i}'] = np.max(channel_data)
        features[f'range_ch{i}'] = np.max(channel_data) - np.min(channel_data)
        
        # Zero crossing rate
        features[f'zcr_ch{i}'] = np.sum(np.diff(np.sign(channel_data)) != 0)
        
        # Root mean square
        features[f'rms_ch{i}'] = np.sqrt(np.mean(channel_data**2))
        
        # Peak-to-peak amplitude
        features[f'pp_ch{i}'] = np.max(channel_data) - np.min(channel_data)
        
        # Entropy
        features[f'entropy_ch{i}'] = entropy(np.histogram(channel_data, bins=20)[0])
    
    return features

def extract_frequency_domain_features(eeg_segment, sfreq=256):
    """
    Extract frequency domain features from EEG segment
    
    Args:
        eeg_segment: EEG data segment (channels, time_points)
        sfreq: Sampling frequency
    
    Returns:
        Dictionary of frequency domain features
    """
    features = {}
    
    for i in range(eeg_segment.shape[0]):
        channel_data = eeg_segment[i, :]
        
        # Compute power spectral density
        freqs, psd = signal.welch(channel_data, fs=sfreq, nperseg=min(256, len(channel_data)//4))
        
        # Band power features
        delta_power = np.trapz(psd[(freqs >= 0.5) & (freqs <= 4)])  # 0.5-4 Hz
        theta_power = np.trapz(psd[(freqs >= 4) & (freqs <= 8)])     # 4-8 Hz
        alpha_power = np.trapz(psd[(freqs >= 8) & (freqs <= 13)])    # 8-13 Hz
        beta_power = np.trapz(psd[(freqs >= 13) & (freqs <= 30)])    # 13-30 Hz
        gamma_power = np.trapz(psd[(freqs >= 30) & (freqs <= 50)])   # 30-50 Hz
        
        features[f'delta_power_ch{i}'] = delta_power
        features[f'theta_power_ch{i}'] = theta_power
        features[f'alpha_power_ch{i}'] = alpha_power
        features[f'beta_power_ch{i}'] = beta_power
        features[f'gamma_power_ch{i}'] = gamma_power
        
        # Total power
        total_power = np.trapz(psd)
        features[f'total_power_ch{i}'] = total_power
        
        # Spectral edge frequency (95% of power)
        cumsum_psd = np.cumsum(psd)
        edge_idx = np.where(cumsum_psd >= 0.95 * cumsum_psd[-1])[0]
        if len(edge_idx) > 0:
            features[f'spectral_edge_ch{i}'] = freqs[edge_idx[0]]
        else:
            features[f'spectral_edge_ch{i}'] = freqs[-1]
        
        # Spectral centroid
        features[f'spectral_centroid_ch{i}'] = np.sum(freqs * psd) / np.sum(psd)
        
        # Spectral entropy
        psd_norm = psd / np.sum(psd)
        features[f'spectral_entropy_ch{i}'] = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
    
    return features

def extract_wavelet_features(eeg_segment, wavelet='db4', levels=4):
    """
    Extract wavelet-based features from EEG segment
    
    Args:
        eeg_segment: EEG data segment (channels, time_points)
        wavelet: Wavelet type
        levels: Number of decomposition levels
    
    Returns:
        Dictionary of wavelet features
    """
    features = {}
    
    for i in range(eeg_segment.shape[0]):
        channel_data = eeg_segment[i, :]
        
        # Wavelet decomposition
        coeffs = pywt.wavedec(channel_data, wavelet, level=levels)
        
        # Energy of each decomposition level
        for j, coeff in enumerate(coeffs):
            energy = np.sum(coeff**2)
            features[f'wavelet_energy_level{j}_ch{i}'] = energy
        
        # Entropy of wavelet coefficients
        for j, coeff in enumerate(coeffs):
            if len(coeff) > 0:
                coeff_hist = np.histogram(coeff, bins=20)[0]
                coeff_entropy = entropy(coeff_hist)
                features[f'wavelet_entropy_level{j}_ch{i}'] = coeff_entropy
    
    return features

def extract_cross_channel_features(eeg_segment):
    """
    Extract cross-channel features (correlation, coherence)
    
    Args:
        eeg_segment: EEG data segment (channels, time_points)
    
    Returns:
        Dictionary of cross-channel features
    """
    features = {}
    n_channels = eeg_segment.shape[0]
    
    # Cross-correlation between channels
    for i in range(n_channels):
        for j in range(i+1, n_channels):
            corr = np.corrcoef(eeg_segment[i, :], eeg_segment[j, :])[0, 1]
            features[f'corr_ch{i}_ch{j}'] = corr
    
    # Average correlation per channel
    for i in range(n_channels):
        correlations = []
        for j in range(n_channels):
            if i != j:
                corr = np.corrcoef(eeg_segment[i, :], eeg_segment[j, :])[0, 1]
                correlations.append(corr)
        features[f'avg_corr_ch{i}'] = np.mean(correlations)
    
    return features

def extract_all_features(eeg_segment, sfreq=256):
    """
    Extract all features from EEG segment
    
    Args:
        eeg_segment: EEG data segment (channels, time_points)
        sfreq: Sampling frequency
    
    Returns:
        Dictionary of all features
    """
    features = {}
    
    # Time domain features
    time_features = extract_time_domain_features(eeg_segment)
    features.update(time_features)
    
    # Frequency domain features
    freq_features = extract_frequency_domain_features(eeg_segment, sfreq)
    features.update(freq_features)
    
    # Wavelet features
    wavelet_features = extract_wavelet_features(eeg_segment)
    features.update(wavelet_features)
    
    # Cross-channel features
    cross_features = extract_cross_channel_features(eeg_segment)
    features.update(cross_features)
    
    return features

def extract_features_batch(eeg_segments, sfreq=256):
    """
    Extract features from a batch of EEG segments
    
    Args:
        eeg_segments: Array of EEG segments (n_segments, channels, time_points)
        sfreq: Sampling frequency
    
    Returns:
        Feature matrix (n_segments, n_features)
    """
    all_features = []
    
    for segment in eeg_segments:
        features = extract_all_features(segment, sfreq)
        all_features.append(list(features.values()))
    
    return np.array(all_features)

def normalize_features(feature_matrix, scaler=None):
    """
    Normalize features using StandardScaler
    
    Args:
        feature_matrix: Feature matrix (n_samples, n_features)
        scaler: Pre-fitted scaler (if None, fit new one)
    
    Returns:
        Normalized feature matrix and fitted scaler
    """
    if scaler is None:
        scaler = StandardScaler()
        feature_matrix_normalized = scaler.fit_transform(feature_matrix)
    else:
        feature_matrix_normalized = scaler.transform(feature_matrix)
    
    return feature_matrix_normalized, scaler
