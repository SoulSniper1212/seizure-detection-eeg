import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import os
import joblib
from datetime import datetime

from src.model import create_cnn_model, create_lstm_model, compile_model
from src.data_loader import load_and_prepare_data, get_data_statistics
from src.evaluate import evaluate_model

def create_model(input_shape, model_type='cnn'):
    """
    Create a model based on the specified type
    
    Args:
        input_shape: Shape of input data
        model_type: Type of model ('cnn' or 'lstm')
    
    Returns:
        Compiled Keras model
    """
    if model_type == 'cnn':
        model = create_cnn_model(input_shape)
    elif model_type == 'lstm':
        model = create_lstm_model(input_shape)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return compile_model(model)

def prepare_data(X, y, test_size=0.2, val_size=0.2, random_state=42):
    """
    Prepare data for training by splitting into train/val/test sets
    
    Args:
        X: Input features
        y: Labels
        test_size: Fraction of data for test set
        val_size: Fraction of remaining data for validation set
        random_state: Random seed
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: train vs val
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def compute_class_weights(y_train):
    """
    Compute class weights to handle imbalanced data
    
    Args:
        y_train: Training labels
    
    Returns:
        Dictionary of class weights
    """
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    
    return dict(zip(np.unique(y_train), class_weights))

def train_model(X_train, y_train, X_val, y_val, model_type='cnn', 
                epochs=50, batch_size=32, learning_rate=0.001, 
                early_stopping=True, class_weights=None):
    """
    Train a seizure detection model
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        model_type: Type of model ('cnn' or 'lstm')
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        early_stopping: Whether to use early stopping
        class_weights: Class weights for imbalanced data
    
    Returns:
        Trained model and training history
    """
    # Debug: Print data shapes
    print(f"X_train shape: {X_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_val shape: {y_val.shape}")
    
    # Check for empty dimensions
    if X_train.shape[1] == 0 or X_train.shape[2] == 0:
        raise ValueError(f"Invalid input shape: {X_train.shape}. Check data preprocessing.")
    
    # Create model
    input_shape = X_train.shape[1:]
    print(f"Creating model with input shape: {input_shape}")
    model = create_model(input_shape, model_type)
    
    # Print model summary
    model.summary()
    
    # Callbacks
    callbacks = []
    
    if early_stopping:
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping_callback)
    
    # Model checkpoint
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        'saved_models/best_model.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    callbacks.append(checkpoint_callback)
    
    # Reduce learning rate on plateau
    lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    callbacks.append(lr_reducer)
    
    # Training
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

def save_model_and_scaler(model, scaler=None, model_name=None):
    """
    Save the trained model and scaler
    
    Args:
        model: Trained Keras model
        scaler: Fitted scaler (optional)
        model_name: Name for the model file
    """
    if model_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"seizure_model_{timestamp}"
    
    # Create saved_models directory if it doesn't exist
    os.makedirs('saved_models', exist_ok=True)
    
    # Save model
    model.save(f'saved_models/{model_name}.h5')
    
    # Save scaler if provided
    if scaler is not None:
        joblib.dump(scaler, f'saved_models/{model_name}_scaler.pkl')
    
    print(f"Model saved as: saved_models/{model_name}.h5")
    if scaler is not None:
        print(f"Scaler saved as: saved_models/{model_name}_scaler.pkl")

def train_pipeline(data_dir, patient_id='chb01', model_type='cnn', 
                   window_sec=5, test_size=0.2, val_size=0.2,
                   epochs=50, batch_size=32, learning_rate=0.001):
    """
    Complete training pipeline
    
    Args:
        data_dir: Directory containing the data
        patient_id: Patient ID
        model_type: Type of model ('cnn' or 'lstm')
        window_sec: Window size in seconds
        test_size: Fraction of data for test set
        val_size: Fraction of remaining data for validation set
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
    
    Returns:
        Trained model, training history, and evaluation results
    """
    print(f"Loading data for patient {patient_id}...")
    
    # Load and prepare data
    X, y, segment_info = load_and_prepare_data(data_dir, patient_id, window_sec)
    
    # Print data statistics
    stats = get_data_statistics(data_dir, patient_id)
    print(f"Dataset statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print(f"Loaded {len(X)} segments with {np.sum(y)} seizure segments")
    print(f"Seizure ratio: {np.mean(y):.3f}")
    
    # Prepare data splits
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(
        X, y, test_size, val_size
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Compute class weights
    class_weights = compute_class_weights(y_train)
    print(f"Class weights: {class_weights}")
    
    # Train model
    print(f"Training {model_type.upper()} model...")
    model, history = train_model(
        X_train, y_train, X_val, y_val,
        model_type=model_type,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        class_weights=class_weights
    )
    
    # Save model
    model_name = f"seizure_{model_type}_{patient_id}"
    save_model_and_scaler(model, model_name=model_name)
    
    # Evaluate model
    print("Evaluating model...")
    evaluate_model(model, X_test, y_test)
    
    return model, history, (X_test, y_test)

if __name__ == "__main__":
    # Example usage
    data_dir = "../data"
    patient_id = "chb01"
    
    model, history, test_data = train_pipeline(
        data_dir=data_dir,
        patient_id=patient_id,
        model_type='cnn',
        window_sec=5,
        epochs=30,
        batch_size=32
    )
