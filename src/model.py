import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

def create_cnn_model(input_shape, num_classes=1):
    """
    Create a CNN model for seizure detection
    
    Args:
        input_shape: Shape of input data (time_steps, channels)
        num_classes: Number of output classes (1 for binary classification)
    
    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        # First convolutional block
        layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv1D(64, kernel_size=3, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.25),
        
        # Second convolutional block
        layers.Conv1D(128, kernel_size=3, activation='relu'),
        layers.BatchNormalization(),
        layers.Conv1D(128, kernel_size=3, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.25),
        
        # Third convolutional block
        layers.Conv1D(256, kernel_size=3, activation='relu'),
        layers.BatchNormalization(),
        layers.Conv1D(256, kernel_size=3, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.25),
        
        # Global pooling and dense layers
        layers.GlobalAveragePooling1D(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='sigmoid')
    ])
    
    return model

def create_lstm_model(input_shape, num_classes=1):
    """
    Create an LSTM model for seizure detection
    
    Args:
        input_shape: Shape of input data (time_steps, channels)
        num_classes: Number of output classes (1 for binary classification)
    
    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        layers.LSTM(128, return_sequences=True, input_shape=input_shape),
        layers.Dropout(0.2),
        layers.LSTM(64, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='sigmoid')
    ])
    
    return model

def compile_model(model, learning_rate=0.001):
    """
    Compile the model with appropriate loss and metrics
    
    Args:
        model: Keras model
        learning_rate: Learning rate for optimizer
    
    Returns:
        Compiled model
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall', 'auc']
    )
    
    return model

def create_ensemble_model(models, weights=None):
    """
    Create an ensemble model from multiple base models
    
    Args:
        models: List of trained models
        weights: Optional weights for each model
    
    Returns:
        Ensemble model
    """
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    
    def ensemble_predict(X):
        predictions = []
        for model in models:
            pred = model.predict(X)
            predictions.append(pred)
        
        # Weighted average
        ensemble_pred = np.zeros_like(predictions[0])
        for i, (pred, weight) in enumerate(zip(predictions, weights)):
            ensemble_pred += weight * pred
        
        return ensemble_pred
    
    return ensemble_predict
