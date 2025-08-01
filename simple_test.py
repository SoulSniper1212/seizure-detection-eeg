#!/usr/bin/env python3
"""
Simple test script to verify basic functionality
"""

import sys
import os
import numpy as np

def test_basic_imports():
    """Test basic imports without TensorFlow"""
    print("Testing basic imports...")
    
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

def test_src_modules_basic():
    """Test source modules without TensorFlow"""
    print("\nTesting source modules (basic)...")
    
    # Add src to path
    sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
    
    try:
        from data_loader import load_chbmit_patient
        print("✓ data_loader imported successfully")
    except ImportError as e:
        print(f"✗ data_loader import failed: {e}")
        return False
    
    try:
        from preprocess import segment_eeg
        print("✓ preprocess imported successfully")
    except ImportError as e:
        print(f"✗ preprocess import failed: {e}")
        return False
    
    try:
        from feature_extract import extract_time_domain_features
        print("✓ feature_extract imported successfully")
    except ImportError as e:
        print(f"✗ feature_extract import failed: {e}")
        return False
    
    try:
        from evaluate import evaluate_model
        print("✓ evaluate imported successfully")
    except ImportError as e:
        print(f"✗ evaluate import failed: {e}")
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

def test_tensorflow_separately():
    """Test TensorFlow separately to avoid conflicts"""
    print("\nTesting TensorFlow...")
    
    try:
        import tensorflow as tf
        print("✓ TensorFlow imported successfully")
        print(f"  Version: {tf.__version__}")
        
        # Test basic TensorFlow functionality
        a = tf.constant([1, 2, 3])
        b = tf.constant([4, 5, 6])
        c = tf.add(a, b)
        print("✓ TensorFlow basic operations work")
        
    except Exception as e:
        print(f"✗ TensorFlow test failed: {e}")
        return False
    
    return True

def main():
    """Run all tests"""
    print("=" * 50)
    print("SEIZURE DETECTION SYSTEM - BASIC TEST")
    print("=" * 50)
    
    tests = [
        ("Basic Package Imports", test_basic_imports),
        ("Source Modules (Basic)", test_src_modules_basic),
        ("Feature Extraction", test_feature_extraction),
        ("Data Structure", test_data_structure),
        ("App Modules", test_app_modules),
        ("TensorFlow", test_tensorflow_separately)
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
    elif passed >= total - 1:
        print("✅ Most tests passed! The system should work.")
        print("\nNote: TensorFlow compatibility warnings are common and usually don't affect functionality.")
        print("\nNext steps:")
        print("1. Train a model: cd src && python train.py")
        print("2. Run the web app: cd app && python app.py")
        print("3. Open http://localhost:5000 in your browser")
    else:
        print("❌ Some critical tests failed. Please check the errors above.")
        print("\nCommon solutions:")
        print("1. Install missing dependencies: pip install -r requirements.txt")
        print("2. Download the CHB-MIT dataset")
        print("3. Check your Python environment")
    
    print("=" * 50)

if __name__ == "__main__":
    main() 