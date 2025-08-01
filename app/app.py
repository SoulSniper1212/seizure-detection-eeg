from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import tensorflow as tf
import os
import sys
import json
import io
import base64
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import load_and_prepare_data, get_data_statistics
from preprocess import segment_eeg
from model import create_cnn_model, compile_model
from utils import process_uploaded_file, create_eeg_plot

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables for model and scaler
model = None
scaler = None
model_loaded = False

def load_model():
    """Load the trained model"""
    global model, model_loaded
    
    try:
        # Look for the most recent model file
        model_dir = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
        if os.path.exists(model_dir):
            model_files = [f for f in os.listdir(model_dir) if f.endswith('.h5')]
            if model_files:
                # Sort by modification time (most recent first)
                model_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
                model_path = os.path.join(model_dir, model_files[0])
                model = tf.keras.models.load_model(model_path)
                model_loaded = True
                print(f"Loaded model: {model_path}")
                return True
        
        print("No trained model found. Please train a model first.")
        return False
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html', model_loaded=model_loaded)

@app.route('/api/load_model', methods=['POST'])
def api_load_model():
    """API endpoint to load model"""
    global model_loaded
    
    if load_model():
        return jsonify({'success': True, 'message': 'Model loaded successfully'})
    else:
        return jsonify({'success': False, 'message': 'Failed to load model'})

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for seizure prediction"""
    global model, model_loaded
    
    if not model_loaded:
        return jsonify({'success': False, 'message': 'Model not loaded'})
    
    try:
        # Get uploaded file
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'message': 'No file selected'})
        
        # Process the uploaded file
        eeg_data, sfreq, channels = process_uploaded_file(file)
        
        if eeg_data is None:
            return jsonify({'success': False, 'message': 'Invalid file format'})
        
        # Segment the EEG data
        window_sec = 5
        window_samples = int(window_sec * sfreq)
        
        segments = []
        for start in range(0, eeg_data.shape[1] - window_samples, window_samples):
            segment = eeg_data[:, start:start + window_samples]
            segments.append(segment)
        
        if not segments:
            return jsonify({'success': False, 'message': 'File too short for analysis'})
        
        X = np.array(segments)
        
        # Make predictions
        predictions = model.predict(X, verbose=0)
        
        # Calculate statistics
        seizure_probability = np.mean(predictions)
        max_probability = np.max(predictions)
        seizure_segments = np.sum(predictions > 0.5)
        total_segments = len(predictions)
        
        # Create visualization
        plot_data = create_eeg_plot(eeg_data, predictions, sfreq, window_sec)
        
        result = {
            'success': True,
            'seizure_probability': float(seizure_probability),
            'max_probability': float(max_probability),
            'seizure_segments': int(seizure_segments),
            'total_segments': int(total_segments),
            'segments_with_seizure': int(seizure_segments),
            'plot_data': plot_data
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error processing file: {str(e)}'})

@app.route('/api/dataset_info', methods=['GET'])
def api_dataset_info():
    """API endpoint to get dataset information"""
    try:
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        
        # Get available patients
        patients = []
        if os.path.exists(data_dir):
            for item in os.listdir(data_dir):
                if item.startswith('chb') and os.path.isdir(os.path.join(data_dir, item)):
                    patients.append(item)
        
        # Get statistics for first patient if available
        stats = None
        if patients:
            try:
                from data_loader import get_data_statistics
                stats = get_data_statistics(data_dir, patients[0])
            except:
                pass
        
        return jsonify({
            'success': True,
            'patients': patients,
            'stats': stats
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error getting dataset info: {str(e)}'})

@app.route('/api/train', methods=['POST'])
def api_train():
    """API endpoint to train a model"""
    try:
        data = request.get_json()
        patient_id = data.get('patient_id', 'chb01')
        model_type = data.get('model_type', 'cnn')
        epochs = data.get('epochs', 30)
        
        # Import training function
        from train import train_pipeline
        
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        
        # Start training (this will run in the main thread for simplicity)
        model, history, test_data = train_pipeline(
            data_dir=data_dir,
            patient_id=patient_id,
            model_type=model_type,
            epochs=epochs
        )
        
        # Reload the model
        global model_loaded
        model_loaded = load_model()
        
        return jsonify({
            'success': True,
            'message': f'Model trained successfully for {patient_id}',
            'model_loaded': model_loaded
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error training model: {str(e)}'})

if __name__ == '__main__':
    # Try to load model on startup
    load_model()
    
    # Create templates directory if it doesn't exist
    templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
    os.makedirs(templates_dir, exist_ok=True)
    
    # Create static directory if it doesn't exist
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    os.makedirs(static_dir, exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
