import numpy as np
import mne
import io
import base64
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from scipy import signal

def process_uploaded_file(file):
    """
    Process an uploaded EEG file
    
    Args:
        file: Flask file object
    
    Returns:
        eeg_data: EEG data array (channels, time_points)
        sfreq: Sampling frequency
        channels: List of channel names
    """
    try:
        # Save file temporarily
        temp_path = '/tmp/uploaded_eeg.edf'
        file.save(temp_path)
        
        # Load with MNE
        raw = mne.io.read_raw_edf(temp_path, preload=True, verbose=False)
        
        # Pick only EEG channels
        raw.pick_types(eeg=True)
        
        # Apply basic filtering
        raw.filter(0.5, 50, fir_design='firwin')
        
        # Get data
        eeg_data = raw.get_data()
        sfreq = raw.info['sfreq']
        channels = raw.ch_names
        
        # Clean up
        import os
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return eeg_data, sfreq, channels
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return None, None, None

def create_eeg_plot(eeg_data, predictions, sfreq, window_sec):
    """
    Create a visualization of EEG data with seizure predictions
    
    Args:
        eeg_data: EEG data array (channels, time_points)
        predictions: Seizure predictions for each segment
        sfreq: Sampling frequency
        window_sec: Window size in seconds
    
    Returns:
        Base64 encoded plot image
    """
    try:
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot EEG data (first few channels)
        n_channels_to_plot = min(5, eeg_data.shape[0])
        time_axis = np.arange(eeg_data.shape[1]) / sfreq
        
        for i in range(n_channels_to_plot):
            # Normalize channel data for better visualization
            channel_data = eeg_data[i, :]
            channel_data_norm = (channel_data - np.mean(channel_data)) / np.std(channel_data)
            ax1.plot(time_axis, channel_data_norm + i * 3, label=f'Channel {i+1}')
        
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Amplitude (normalized)')
        ax1.set_title('EEG Signals')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot seizure predictions
        window_samples = int(window_sec * sfreq)
        n_segments = len(predictions)
        time_segments = np.arange(n_segments) * window_sec
        
        ax2.plot(time_segments, predictions.flatten(), 'b-', linewidth=2, label='Seizure Probability')
        ax2.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Threshold (0.5)')
        ax2.fill_between(time_segments, 0, predictions.flatten(), alpha=0.3, color='blue')
        
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Seizure Probability')
        ax2.set_title('Seizure Detection Results')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # Add statistics text
        seizure_segments = np.sum(predictions > 0.5)
        total_segments = len(predictions)
        avg_probability = np.mean(predictions)
        
        stats_text = f'Seizure Segments: {seizure_segments}/{total_segments} ({100*seizure_segments/total_segments:.1f}%)\n'
        stats_text += f'Average Probability: {avg_probability:.3f}'
        
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return plot_data
        
    except Exception as e:
        print(f"Error creating plot: {e}")
        return None

def extract_features_from_segment(segment, sfreq=256):
    """
    Extract features from an EEG segment
    
    Args:
        segment: EEG segment (channels, time_points)
        sfreq: Sampling frequency
    
    Returns:
        Dictionary of features
    """
    features = {}
    
    for i in range(segment.shape[0]):
        channel_data = segment[i, :]
        
        # Time domain features
        features[f'mean_ch{i}'] = np.mean(channel_data)
        features[f'std_ch{i}'] = np.std(channel_data)
        features[f'var_ch{i}'] = np.var(channel_data)
        features[f'min_ch{i}'] = np.min(channel_data)
        features[f'max_ch{i}'] = np.max(channel_data)
        features[f'range_ch{i}'] = np.max(channel_data) - np.min(channel_data)
        
        # Frequency domain features
        freqs, psd = signal.welch(channel_data, fs=sfreq, nperseg=min(256, len(channel_data)//4))
        
        # Band powers
        delta_power = np.trapz(psd[(freqs >= 0.5) & (freqs <= 4)])
        theta_power = np.trapz(psd[(freqs >= 4) & (freqs <= 8)])
        alpha_power = np.trapz(psd[(freqs >= 8) & (freqs <= 13)])
        beta_power = np.trapz(psd[(freqs >= 13) & (freqs <= 30)])
        gamma_power = np.trapz(psd[(freqs >= 30) & (freqs <= 50)])
        
        features[f'delta_power_ch{i}'] = delta_power
        features[f'theta_power_ch{i}'] = theta_power
        features[f'alpha_power_ch{i}'] = alpha_power
        features[f'beta_power_ch{i}'] = beta_power
        features[f'gamma_power_ch{i}'] = gamma_power
    
    return features

def validate_eeg_file(file):
    """
    Validate uploaded EEG file
    
    Args:
        file: Flask file object
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Check file extension
        if not file.filename.lower().endswith('.edf'):
            return False, "Only .edf files are supported"
        
        # Check file size (max 50MB)
        file.seek(0, 2)  # Seek to end
        file_size = file.tell()
        file.seek(0)  # Reset to beginning
        
        if file_size > 50 * 1024 * 1024:  # 50MB
            return False, "File size too large (max 50MB)"
        
        return True, ""
        
    except Exception as e:
        return False, f"Error validating file: {str(e)}"

def format_time_duration(seconds):
    """
    Format time duration in a human-readable format
    
    Args:
        seconds: Duration in seconds
    
    Returns:
        Formatted string
    """
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} hours"

def calculate_seizure_risk(predictions, threshold=0.5):
    """
    Calculate seizure risk based on predictions
    
    Args:
        predictions: Array of seizure probabilities
        threshold: Probability threshold for seizure detection
    
    Returns:
        Dictionary with risk assessment
    """
    seizure_segments = np.sum(predictions > threshold)
    total_segments = len(predictions)
    seizure_ratio = seizure_segments / total_segments if total_segments > 0 else 0
    max_probability = np.max(predictions)
    avg_probability = np.mean(predictions)
    
    # Risk assessment
    if seizure_ratio == 0:
        risk_level = "Low"
        risk_description = "No seizure activity detected"
    elif seizure_ratio < 0.1:
        risk_level = "Low"
        risk_description = "Minimal seizure activity detected"
    elif seizure_ratio < 0.3:
        risk_level = "Medium"
        risk_description = "Moderate seizure activity detected"
    elif seizure_ratio < 0.5:
        risk_level = "High"
        risk_description = "Significant seizure activity detected"
    else:
        risk_level = "Critical"
        risk_description = "Extensive seizure activity detected"
    
    return {
        'risk_level': risk_level,
        'risk_description': risk_description,
        'seizure_ratio': seizure_ratio,
        'seizure_segments': seizure_segments,
        'total_segments': total_segments,
        'max_probability': max_probability,
        'avg_probability': avg_probability
    }
