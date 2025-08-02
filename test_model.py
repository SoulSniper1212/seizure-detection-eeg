import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def test_model():
    """Test the trained seizure detection model"""
    
    # Load the trained model
    print("Loading trained model...")
    model = tf.keras.models.load_model('saved_models/seizure_cnn_chb01.h5')
    print("✅ Model loaded successfully!")
    
    # Create test data with the correct shape
    print("\nCreating test data...")
    test_data = np.random.randn(100, 1280, 23)  # 100 samples, 1280 time points, 23 channels
    print(f"✅ Test data shape: {test_data.shape}")
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = model.predict(test_data)
    print(f"✅ Predictions shape: {predictions.shape}")
    print(f"✅ Sample predictions: {predictions[:5].flatten()}")
    
    # Test with different types of data
    print("\n" + "="*50)
    print("TESTING DIFFERENT DATA TYPES")
    print("="*50)
    
    # 1. Random noise (should be mostly non-seizure)
    print("\n1. Testing with random noise...")
    noise_data = np.random.randn(50, 1280, 23)
    noise_predictions = model.predict(noise_data)
    print(f"   Mean prediction: {np.mean(noise_predictions):.4f}")
    print(f"   Predictions > 0.5: {np.sum(noise_predictions > 0.5)}/{len(noise_predictions)}")
    
    # 2. Low amplitude data (normal EEG-like)
    print("\n2. Testing with low amplitude data...")
    low_amp_data = np.random.randn(50, 1280, 23) * 0.1
    low_amp_predictions = model.predict(low_amp_data)
    print(f"   Mean prediction: {np.mean(low_amp_predictions):.4f}")
    print(f"   Predictions > 0.5: {np.sum(low_amp_predictions > 0.5)}/{len(low_amp_predictions)}")
    
    # 3. High amplitude data (seizure-like)
    print("\n3. Testing with high amplitude data...")
    high_amp_data = np.random.randn(50, 1280, 23) * 3.0
    high_amp_predictions = model.predict(high_amp_data)
    print(f"   Mean prediction: {np.mean(high_amp_predictions):.4f}")
    print(f"   Predictions > 0.5: {np.sum(high_amp_predictions > 0.5)}/{len(high_amp_predictions)}")
    
    # 4. Periodic data (seizure-like patterns)
    print("\n4. Testing with periodic data...")
    t = np.linspace(0, 1280, 1280)
    periodic_data = np.zeros((50, 1280, 23))
    for i in range(50):
        for j in range(23):
            freq = np.random.uniform(1, 10)  # Random frequency
            periodic_data[i, :, j] = np.sin(2 * np.pi * freq * t / 1280) * np.random.uniform(0.5, 2.0)
    periodic_predictions = model.predict(periodic_data)
    print(f"   Mean prediction: {np.mean(periodic_predictions):.4f}")
    print(f"   Predictions > 0.5: {np.sum(periodic_predictions > 0.5)}/{len(periodic_predictions)}")
    
    # 5. Zero data (should be non-seizure)
    print("\n5. Testing with zero data...")
    zero_data = np.zeros((50, 1280, 23))
    zero_predictions = model.predict(zero_data)
    print(f"   Mean prediction: {np.mean(zero_predictions):.4f}")
    print(f"   Predictions > 0.5: {np.sum(zero_predictions > 0.5)}/{len(zero_predictions)}")
    
    # Summary statistics
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    
    all_predictions = np.concatenate([
        noise_predictions, low_amp_predictions, high_amp_predictions, 
        periodic_predictions, zero_predictions
    ])
    
    print(f"📊 Overall Prediction Statistics:")
    print(f"   Total samples: {len(all_predictions)}")
    print(f"   Mean prediction: {np.mean(all_predictions):.4f}")
    print(f"   Std prediction: {np.std(all_predictions):.4f}")
    print(f"   Min prediction: {np.min(all_predictions):.4f}")
    print(f"   Max prediction: {np.max(all_predictions):.4f}")
    print(f"   Predictions > 0.5: {np.sum(all_predictions > 0.5)}/{len(all_predictions)} ({np.mean(all_predictions > 0.5)*100:.1f}%)")
    
    # Test model summary
    print(f"\n📋 Model Summary:")
    model.summary()
    
    print("\n🎉 Model test completed successfully!")
    
    # Return the model for potential further use
    return model

if __name__ == "__main__":
    test_model() 