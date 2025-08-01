#!/usr/bin/env python3
"""
Test script to verify the seizure detection system setup
"""

import sys
import os
import numpy as np

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import scipy
        print("✓ SciPy imported successfully")
    except ImportError as e:
        print(f"✗ SciPy import failed: {e}")
        return False
    
    try:
        import mne
        print("✓ MNE imported successfully")
    except ImportError as e:
        print(f"✗ MNE import failed: {e}")
        return False
    
    try:
        import tensorflow as tf
        print("✓ TensorFlow imported successfully")
    except ImportError as e:
        print(f"✗ TensorFlow import failed: {e}")
        return False
    
    try:
        import sklearn
        print("✓ Scikit-learn imported successfully")
    except ImportError as e:
        print(f"✗ Scikit-learn import failed: {e}")
        return False
    
    try:
        import flask
        print("✓ Flask imported successfully")
    except ImportError as e:
        print(f"✗ Flask import failed: {e}")
        return False
    
    try:
        import pywt
        print("✓ PyWavelets imported successfully")
    except ImportError as e:
        print(f"✗ PyWavelets import failed: {e}")
        return False
    
    return True

def test_src_modules():
    """Test if all source modules can be imported"""
    print("\nTesting source modules...")
    
    # Add src to path
    sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
    
    try:
        from data_loader import load_chbmit_patient
        print("✓ data_loader imported successfully")
    except ImportError as e:
        print(f"✗ data_loader import failed: {e}")
        return False
    
    try:
        from model import create_cnn_model, create_lstm_model
        print("✓ model imported successfully")
    except ImportError as e:
        print(f"✗ model import failed: {e}")
        return False
    
    try:
        from feature_extract import extract_time_domain_features
        print("✓ feature_extract imported successfully")
    except ImportError as e:
        print(f"✗ feature_extract import failed: {e}")
        return False
    
    try:
        from preprocess import segment_eeg
        print("✓ preprocess imported successfully")
    except ImportError as e:
        print(f"✗ preprocess import failed: {e}")
        return False
    
    try:
        from train import train_pipeline
        print("✓ train imported successfully")
    except ImportError as e:
        print(f"✗ train import failed: {e}")
        return False
    
    try:
        from evaluate import evaluate_model
        print("✓ evaluate imported successfully")
    except ImportError as e:
        print(f"✗ evaluate import failed: {e}")
        return False
    
    return True

def test_model_creation():
    """Test if models can be created"""
    print("\nTesting model creation...")
    
    try:
        from model import create_cnn_model, create_lstm_model, compile_model
        
        # Test CNN model
        input_shape = (18, 1280)  # 18 channels, 5 seconds at 256 Hz
        cnn_model = create_cnn_model(input_shape)
        cnn_model = compile_model(cnn_model)
        print("✓ CNN model created successfully")
        
        # Test LSTM model
        lstm_model = create_lstm_model(input_shape)
        lstm_model = compile_model(lstm_model)
        print("✓ LSTM model created successfully")
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False
    
    return True

def test_feature_extraction():
    """Test feature extraction"""
    print("\nTesting feature extraction...")
    
    try:
        from feature_extract import extract_time_domain_features, extract_frequency_domain_features
        
        # Create dummy EEG data
        eeg_segment = np.random.randn(18, 1280)  # 18 channels, 5 seconds
        
        # Test time domain features
        time_features = extract_time_domain_features(eeg_segment)
        print(f"✓ Time domain features extracted: {len(time_features)} features")
        
        # Test frequency domain features
        freq_features = extract_frequency_domain_features(eeg_segment)
        print(f"✓ Frequency domain features extracted: {len(freq_features)} features")
        
    except Exception as e:
        print(f"✗ Feature extraction failed: {e}")
        return False
    
    return True

def test_data_structure():
    """Test if data directory structure is correct"""
    print("\nTesting data structure...")
    
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    
    if not os.path.exists(data_dir):
        print("✗ Data directory does not exist")
        return False
    
    # Check for patient directories
    patient_dirs = [d for d in os.listdir(data_dir) if d.startswith('chb') and os.path.isdir(os.path.join(data_dir, d))]
    
    if not patient_dirs:
        print("✗ No patient directories found in data/")
        print("Please download the CHB-MIT dataset and place it in the data/ directory")
        return False
    
    print(f"✓ Found {len(patient_dirs)} patient directories: {patient_dirs}")
    
    # Check for EDF files
    total_edf_files = 0
    for patient in patient_dirs:
        patient_path = os.path.join(data_dir, patient)
        edf_files = [f for f in os.listdir(patient_path) if f.endswith('.edf')]
        total_edf_files += len(edf_files)
        print(f"  {patient}: {len(edf_files)} EDF files")
    
    if total_edf_files == 0:
        print("✗ No EDF files found")
        return False
    
    print(f"✓ Total EDF files: {total_edf_files}")
    return True

def test_app_modules():
    """Test if app modules can be imported"""
    print("\nTesting app modules...")
    
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))
        from utils import process_uploaded_file, create_eeg_plot
        print("✓ app.utils imported successfully")
    except ImportError as e:
        print(f"✗ app.utils import failed: {e}")
        return False
    
    return True

def main():
    """Run all tests"""
    print("=" * 50)
    print("SEIZURE DETECTION SYSTEM SETUP TEST")
    print("=" * 50)
    
    tests = [
        ("Package Imports", test_imports),
        ("Source Modules", test_src_modules),
        ("Model Creation", test_model_creation),
        ("Feature Extraction", test_feature_extraction),
        ("Data Structure", test_data_structure),
        ("App Modules", test_app_modules)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func():
            passed += 1
            print(f"✓ {test_name} PASSED")
        else:
            print(f"✗ {test_name} FAILED")
    
    print("\n" + "=" * 50)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("1. Train a model: cd src && python train.py")
        print("2. Run the web app: cd app && python app.py")
        print("3. Open http://localhost:5000 in your browser")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("\nCommon solutions:")
        print("1. Install missing dependencies: pip install -r requirements.txt")
        print("2. Download the CHB-MIT dataset")
        print("3. Check your Python environment")
    
    print("=" * 50)

if __name__ == "__main__":
    main() 