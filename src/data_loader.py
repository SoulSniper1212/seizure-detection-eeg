import os
import mne
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict

def load_chbmit_patient(data_dir, patient_id='chb01'):
    """
    Load EEG data for a specific patient
    
    Args:
        data_dir: Directory containing the data
        patient_id: Patient ID (e.g., 'chb01')
    
    Returns:
        eeg_data: List of EEG data arrays
        raw_objects: List of MNE Raw objects
        seizure_info: Dictionary with seizure information
    """
    patient_path = os.path.join(data_dir, patient_id)
    edf_files = [f for f in os.listdir(patient_path) if f.endswith('.edf')]
    
    eeg_data = []
    raw_objects = []
    seizure_info = {}
    
    for edf_file in sorted(edf_files):
        full_path = os.path.join(patient_path, edf_file)
        raw = mne.io.read_raw_edf(full_path, preload=True, verbose=False)
        raw.pick_types(eeg=True)
        raw.filter(0.5, 50, fir_design='firwin')
        eeg_data.append(raw.get_data())
        raw_objects.append(raw)
        
        # Load seizure annotations if available
        seizure_file = full_path + '.seizures'
        if os.path.exists(seizure_file):
            seizure_info[edf_file] = load_seizure_annotations(seizure_file, raw.info['sfreq'])
    
    return eeg_data, raw_objects, seizure_info

def load_seizure_annotations(seizure_file, sfreq):
    """
    Load seizure annotations from file
    
    Args:
        seizure_file: Path to seizure annotation file
        sfreq: Sampling frequency
    
    Returns:
        List of seizure time ranges (start_sample, end_sample)
    """
    seizures = []
    try:
        with open(seizure_file, 'r') as f:
            for line in f:
                if line.strip():
                    start_time, end_time = map(float, line.strip().split())
                    start_sample = int(start_time * sfreq)
                    end_sample = int(end_time * sfreq)
                    seizures.append((start_sample, end_sample))
    except Exception as e:
        print(f"Warning: Could not load seizure annotations from {seizure_file}: {e}")
    
    return seizures

def load_multiple_patients(data_dir, patient_ids=None):
    """
    Load data from multiple patients
    
    Args:
        data_dir: Directory containing the data
        patient_ids: List of patient IDs to load (if None, load all available)
    
    Returns:
        Dictionary with patient data
    """
    if patient_ids is None:
        # Find all patient directories
        patient_ids = []
        for item in os.listdir(data_dir):
            if item.startswith('chb') and os.path.isdir(os.path.join(data_dir, item)):
                patient_ids.append(item)
    
    all_patient_data = {}
    
    for patient_id in patient_ids:
        try:
            eeg_data, raw_objects, seizure_info = load_chbmit_patient(data_dir, patient_id)
            all_patient_data[patient_id] = {
                'eeg_data': eeg_data,
                'raw_objects': raw_objects,
                'seizure_info': seizure_info
            }
            print(f"Loaded {len(eeg_data)} files for {patient_id}")
        except Exception as e:
            print(f"Warning: Could not load data for {patient_id}: {e}")
    
    return all_patient_data

def get_seizure_labels(eeg_data, raw_objects, seizure_info, window_sec=5):
    """
    Generate seizure labels for EEG segments
    
    Args:
        eeg_data: List of EEG data arrays
        raw_objects: List of MNE Raw objects
        seizure_info: Dictionary with seizure information
        window_sec: Window size in seconds
    
    Returns:
        labels: Binary labels for each segment
        segment_info: Additional information about segments
    """
    labels = []
    segment_info = []
    
    for i, (data, raw) in enumerate(zip(eeg_data, raw_objects)):
        sfreq = raw.info['sfreq']
        window_samples = int(window_sec * sfreq)
        total_samples = data.shape[1]
        
        # Get seizure times for this file
        file_name = os.path.basename(raw.filenames[0])
        seizures = seizure_info.get(file_name, [])
        
        # Create segments
        for start in range(0, total_samples - window_samples, window_samples):
            end = start + window_samples
            
            # Check if this segment overlaps with any seizure
            is_seizure = False
            seizure_overlap = 0
            
            for seizure_start, seizure_end in seizures:
                # Calculate overlap
                overlap_start = max(start, seizure_start)
                overlap_end = min(end, seizure_end)
                if overlap_start < overlap_end:
                    overlap = overlap_end - overlap_start
                    seizure_overlap += overlap
                    is_seizure = True
            
            # Label as seizure if more than 50% of segment is seizure
            seizure_ratio = seizure_overlap / window_samples
            label = 1 if seizure_ratio > 0.5 else 0
            
            labels.append(label)
            segment_info.append({
                'file_index': i,
                'start_sample': start,
                'end_sample': end,
                'seizure_ratio': seizure_ratio,
                'is_seizure': is_seizure
            })
    
    return np.array(labels), segment_info

def load_and_prepare_data(data_dir, patient_id='chb01', window_sec=5):
    """
    Load and prepare data for training
    
    Args:
        data_dir: Directory containing the data
        patient_id: Patient ID
        window_sec: Window size in seconds
    
    Returns:
        X: EEG segments (n_segments, time_points, channels) - transposed for Conv1D
        y: Labels (n_segments,)
        segment_info: Additional segment information
    """
    # Load data
    eeg_data, raw_objects, seizure_info = load_chbmit_patient(data_dir, patient_id)
    
    print(f"Loaded {len(eeg_data)} EEG files")
    print(f"Loaded {len(raw_objects)} raw objects")
    print(f"Found seizure info for {len(seizure_info)} files")
    
    # Generate segments and labels
    from src.preprocess import segment_eeg
    X = segment_eeg(eeg_data, raw_objects, window_sec)
    y, segment_info = get_seizure_labels(eeg_data, raw_objects, seizure_info, window_sec)
    
    # Transpose X to (n_segments, time_points, channels) for Conv1D
    X = np.transpose(X, (0, 2, 1))
    
    print(f"Final X shape: {X.shape}")
    print(f"Final y shape: {y.shape}")
    print(f"Number of seizure segments: {np.sum(y)}")
    print(f"Number of non-seizure segments: {np.sum(y == 0)}")
    
    return X, y, segment_info

def get_data_statistics(data_dir, patient_id='chb01'):
    """
    Get statistics about the dataset
    
    Args:
        data_dir: Directory containing the data
        patient_id: Patient ID
    
    Returns:
        Dictionary with dataset statistics
    """
    eeg_data, raw_objects, seizure_info = load_chbmit_patient(data_dir, patient_id)
    
    total_seizures = 0
    total_seizure_time = 0
    
    for file_name, seizures in seizure_info.items():
        total_seizures += len(seizures)
        for start_sample, end_sample in seizures:
            seizure_duration = (end_sample - start_sample) / raw_objects[0].info['sfreq']
            total_seizure_time += seizure_duration
    
    total_recording_time = sum(raw.n_times / raw.info['sfreq'] for raw in raw_objects)
    
    stats = {
        'n_files': len(eeg_data),
        'n_channels': eeg_data[0].shape[0] if eeg_data else 0,
        'total_recording_time_hours': total_recording_time / 3600,
        'total_seizures': total_seizures,
        'total_seizure_time_hours': total_seizure_time / 3600,
        'seizure_frequency_per_hour': total_seizures / (total_recording_time / 3600),
        'seizure_ratio': total_seizure_time / total_recording_time
    }
    
    return stats
